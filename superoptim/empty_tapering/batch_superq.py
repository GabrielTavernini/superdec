import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import open3d as o3d
import math
from typing import Optional, assert_never
from dataclasses import dataclass
import gc

from ..utils import quat2mat, mat2quat
from superdec.utils.safe_operations import safe_pow, safe_mul
from superdec.utils.predictions_handler_extended import PredictionHandler

class SuperQ(nn.Module):
    def __init__(
        self, 
        pred_handler: PredictionHandler,
        truncation: float = 0.1,
        ply: str = None,
        use_full_pointcloud: bool = False,
        idx: int = 0,
        device: str = "cuda",
        silent: bool = False,
    ):
        self.minE, self.maxE = 0.1, 1.9
        # Anything self.x = nn.Parameter(...) is trainable
        trainable = ["raw_scale", "raw_exponents", "raw_rotation", "raw_tapering", "translation"]

        super().__init__()
        self.idx = idx
        self.mask = (pred_handler.exist[self.idx] > 0.5).reshape(-1)
        self.N = self.mask.sum()

        raw_scale = torch.tensor(pred_handler.scale[self.idx].reshape(-1, 3)[self.mask], dtype=torch.float, device=device)
        self.raw_scale = torch.log(raw_scale)
        raw_exponents = torch.tensor(pred_handler.exponents[self.idx].reshape(-1, 2)[self.mask], dtype=torch.float, device=device)
        self.raw_exponents = torch.logit((raw_exponents - self.minE) / (self.maxE - self.minE))
        rot_mat = torch.tensor(pred_handler.rotation[self.idx].reshape(-1, 3, 3)[self.mask], dtype=torch.float, device=device)
        self.raw_rotation = mat2quat(rot_mat) # Shape (N, 3)
        self.translation = torch.tensor(pred_handler.translation[self.idx].reshape(-1, 3)[self.mask], dtype=torch.float, device=device)
        self.raw_tapering = torch.full((self.N, 2), 1e-4, dtype=torch.float, device=device)

        self.assign_matrix = torch.tensor(pred_handler.assign_matrix[self.idx].T[self.mask], dtype=torch.float, device=device).squeeze() # [N, 4096]
        self.points = torch.tensor(pred_handler.pc[self.idx], dtype=torch.float, device=device) # [4096, 3]
        
        if ply and ply.endswith("npz"):
            try:
                data = np.load(ply)
                ply_points = torch.tensor(np.array(data['points']), dtype=torch.float, device=device) 
                ply_normals = torch.tensor(np.array(data['normals']), dtype=torch.float, device=device) 

                distances = torch.cdist(self.points, ply_points)
                closest_ply_indices = torch.argmin(distances, dim=1)
                self.normals = ply_normals[closest_ply_indices]
            except Exception as e:
                print(f"Warning: Failed to load ply normals from {ply}: {e}")
                self.compute_normals_o3d(device)
        else:
            self.compute_normals_o3d(device)

        og_points = self.points
        if use_full_pointcloud and 'ply_points' in locals():
            distances = torch.cdist(ply_points, self.points)
            nearest_indices = torch.argmin(distances, dim=1)
            full_assign_matrix = self.assign_matrix[:, nearest_indices]
            self.assign_matrix = full_assign_matrix
            self.points = ply_points
            self.normals = ply_normals

        # Generate outside points logic remains similar
        outside_points = []
        self.line_length = 0.01
        tmp_points, tmp_normals = self.points, self.normals
        decay_rate = 0.5
        for i in range(3):
            tmp_points = tmp_points + (tmp_normals * self.line_length)
            distances = torch.cdist(tmp_points, og_points)
            distances = torch.min(distances, dim=1).values
            mask = (distances >= (self.line_length * (i+1)) - 1e-4)
            
            valid_indices = torch.nonzero(mask).squeeze()
            if valid_indices.numel() > 0:
                num_valid = valid_indices.numel()
                num_to_sample = max(1, int(num_valid * (decay_rate if i > 0 else 1)))
                selected_indices = valid_indices[torch.randperm(num_valid)[:num_to_sample]]
                tmp_points, tmp_normals = tmp_points[selected_indices], tmp_normals[selected_indices]
                outside_points.append(tmp_points)
        if len(outside_points) > 0:
            self.outside_points = torch.cat(outside_points, dim=0)
        else:
             self.outside_points = torch.empty((0, 3), device=device)


        self.pred_handler = pred_handler
        self.truncation = truncation
        self.device = device

        for name in trainable:
            val = getattr(self, name)
            setattr(self, name, nn.Parameter(val))
        
        self.trainable_params = self.trainable_params()
        if not silent:
            print(f"Loaded {self.N} superquadircs.")
            print(f"Using {self.outside_points.shape[0]} outside points")
            print("Trainable superq params:", self.trainable_params)

    def compute_normals_o3d(self, device):
        pc_o3d = o3d.geometry.PointCloud()
        pc_o3d.points = o3d.utility.Vector3dVector(self.points.detach().cpu().numpy())
        pc_o3d.estimate_normals()
        self.normals = torch.tensor(np.array(pc_o3d.normals), dtype=torch.float, device=device)

    def scale(self):
        # Add epsilon to scales to prevent division by zero / huge gradients
        return torch.exp(self.raw_scale) + 1e-6

    def exponents(self):
        return (torch.sigmoid(self.raw_exponents) * (self.maxE - self.minE)) + self.minE

    def rotation(self):
        return quat2mat(self.raw_rotation)
    
    def tapering(self):
        return torch.tanh(self.raw_tapering)

    def trainable_params(self):
        names = []
        params = dict(self.named_parameters())
        for k, v in params.items():
            if isinstance(v, torch.Tensor):
                names.append(k)
        return names

    def get_param_groups(self):
        lrs = {
            "raw_scale": 5e-2,
            "raw_exponents": 1e-2,
            "raw_tapering": 5e-4,
        }
        groups = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name in lrs:
                lr = lrs[name]
            else: 
                lr = 1e-3
            groups.append({"params": [param], "lr": lr})
        return groups

    def update_handler(self, compute_meshes=True):
        batch_size = self.pred_handler.scale.shape[1]
        mask = self.mask.reshape(batch_size)
        self.pred_handler.scale[self.idx][mask] = self.scale().detach().cpu().numpy()
        self.pred_handler.exponents[self.idx][mask] = self.exponents().detach().cpu().numpy()
        self.pred_handler.tapering[self.idx][mask] = self.tapering().detach().cpu().numpy()
        self.pred_handler.rotation[self.idx][mask] = self.rotation().detach().cpu().numpy()
        self.pred_handler.translation[self.idx][mask] = self.translation.detach().cpu().numpy()
        if compute_meshes:
            meshes = self.pred_handler.get_meshes(resolution=30)
            return self.pred_handler, meshes
        else:
            return self.pred_handler

    def sdf_batch(self, points):
        """
        Compute SDF for all primitives in parallel using broadcasting.
        points: (3, M) tensor where 3 is (x,y,z) and M is number of points.
        Returns: (N, M) tensor of SDF values where N is number of primitives.
        """
        # Data preparation
        N = self.N
        M = points.shape[1]
        
        # 1. Transform points to local coordinate system
        # points: (1, 3, M)
        points_expanded = points.unsqueeze(0) 
        
        # translation: (N, 3) -> (N, 3, 1)
        t = self.translation.unsqueeze(2)
        
        # points_centered: (N, 3, M)
        points_centered = points_expanded - t
        
        # rotation: (N, 3, 3)
        # R^T: (N, 3, 3) -> transpose last two dims is the same for rotation matrix inverse
        # X = R^T @ (points - t)
        # (N, 3, 3).transpose(-2,-1) @ (N, 3, M) -> (N, 3, M)
        X = torch.matmul(self.rotation().transpose(-2, -1), points_centered)
        
        # 2. Extract parameters
        # exponents: (N, 2)
        e1 = self.exponents()[:, 0].view(N, 1) # (N, 1)
        e2 = self.exponents()[:, 1].view(N, 1) # (N, 1)
        
        # scale: (N, 3)
        sx = self.scale()[:, 0].view(N, 1)
        sy = self.scale()[:, 1].view(N, 1)
        sz = self.scale()[:, 2].view(N, 1)
        
        # Extract coordinates (N, M)
        x = X[:, 0, :]
        y = X[:, 1, :]
        z = X[:, 2, :]
        
        # Fix numerical instability at 0
        eps = 1e-6
        x = ((x > 0).float() * 2 - 1) * torch.max(torch.abs(x), torch.tensor(eps, device=x.device))
        y = ((y > 0).float() * 2 - 1) * torch.max(torch.abs(y), torch.tensor(eps, device=y.device))
        z = ((z > 0).float() * 2 - 1) * torch.max(torch.abs(z), torch.tensor(eps, device=z.device))
        
        # Apply tapering
        # tapering: (N, 2)
        kx = self.tapering()[:, 0].view(N, 1)
        ky = self.tapering()[:, 1].view(N, 1)
        
        fx = safe_mul(kx/sz, z) + 1
        fy = safe_mul(ky/sz, z) + 1
        
        # Avoid division by zero in tapering
        fx = ((fx > 0).float() * 2 - 1) * torch.max(torch.abs(fx), torch.tensor(eps, device=fx.device))
        fy = ((fy > 0).float() * 2 - 1) * torch.max(torch.abs(fy), torch.tensor(eps, device=fy.device))
        x = x / fx
        y = y / fy
        
        # Re-stack X is implicitly handled by component-wise operations below, 
        # but for radial distance we need to stack again or just compute norm
        # X_new = torch.stack([x, y, z], dim=1) # (N, 3, M)
        
        # 3. Calculate radial distance from origin (N, M)
        # r0 = torch.linalg.norm(X_new, dim=1) # This matches axis=0 in original code which was (3, M)
        r0 = torch.sqrt(x**2 + y**2 + z**2)

        # 4. Calculate the Superquadric scaling function
        term1 = safe_pow(safe_pow(x / sx, 2), 1 / e2)
        term2 = safe_pow(safe_pow(y / sy, 2), 1 / e2)
        term3 = safe_pow(safe_pow(z / sz, 2), 1 / e1)
        
        f = safe_pow(safe_pow(term1 + term2, e2 / e1) + term3, -e1 / 2)

        # 5. Compute Signed Distance
        sdf = safe_mul(r0, (1 - f))

        return sdf

    def forward(self):
        split_idx = self.points.shape[0]
        all_points = torch.cat([self.points, self.outside_points], dim=0)
        all_sdfs = self.sdf_batch(all_points.T)
        # Leaky clipping
        all_sdfs_clipped = torch.clip(all_sdfs, -self.truncation, self.truncation)
        all_sdfs_leaky = all_sdfs_clipped + 0.1 * (all_sdfs - all_sdfs_clipped)

        ## Surface values
        sdfs_points = all_sdfs[:, :split_idx]
        
        # Calculate weights for weighted mean
        logits_points = -100.0 * sdfs_points
        weights_points = F.softmax(logits_points, dim=0)
        values_points = torch.sum(weights_points * all_sdfs_leaky[:, :split_idx], dim=0)

        # Calculate counts based on minimum SDF (hard assignment)
        idx_points = torch.argmin(sdfs_points, dim=0)
        counts_points = torch.bincount(idx_points, minlength=self.N).float()

        ## Outside values
        values_outside, idx_outside = torch.min(all_sdfs_leaky[:, split_idx:], dim=0)
        valid_mask = values_outside < 0
        counts_outside = torch.bincount(idx_outside[valid_mask], minlength=self.N).float()
        return values_points, values_outside, counts_points, counts_outside
