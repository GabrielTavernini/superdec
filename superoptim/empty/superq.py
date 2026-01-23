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
from superdec.utils.predictions_handler import PredictionHandler

class SuperQ(nn.Module):
    def __init__(
        self, 
        pred_handler: PredictionHandler,
        truncation: float = 0.1,
        ply: str = None,
        use_full_pointcloud: bool = False,
        idx: int = 0,
        device: str = "cuda",
    ):
        # Anything self.x = nn.Parameter(...) is trainable
        trainable = ["raw_scale", "raw_exponents", "raw_rotation", "translation"]

        super().__init__()
        self.idx = idx
        self.mask = (pred_handler.exist[self.idx] > 0.5).reshape(-1)
        raw_scale = torch.tensor(pred_handler.scale[self.idx].reshape(-1, 3)[self.mask], dtype=torch.float, device=device)
        self.raw_scale = torch.log(raw_scale)
        raw_exponents = torch.tensor(pred_handler.exponents[self.idx].reshape(-1, 2)[self.mask], dtype=torch.float, device=device)
        self.raw_exponents = torch.logit(raw_exponents)
        rot_mat = torch.tensor(pred_handler.rotation[self.idx].reshape(-1, 3, 3)[self.mask], dtype=torch.float, device=device)
        self.raw_rotation = mat2quat(rot_mat) # Shape (N, 3)
        self.translation = torch.tensor(pred_handler.translation[self.idx].reshape(-1, 3)[self.mask], dtype=torch.float, device=device)
        
        self.assign_matrix = torch.tensor(pred_handler.assign_matrix[self.idx].T[self.mask], dtype=torch.float, device=device).squeeze() # [N, 4096]
        self.points = torch.tensor(np.array(pred_handler.get_segmented_pcs()[self.idx].points), dtype=torch.float, device=device) # [4096, 3]
        
        if ply.endswith("ply"):
            pcd = o3d.io.read_point_cloud(ply) 
            ply_points = torch.tensor(np.array(pcd.points), dtype=torch.float, device=device) 
        elif ply.endswith("npz"):
            data = np.load(ply)
            ply_points = torch.tensor(np.array(data['points']), dtype=torch.float, device=device) 
            self.normals = torch.tensor(np.array(data['normals']), dtype=torch.float, device=device) 
        else:
            print("Cannot load pointcloud")
            exit()

        og_points = self.points
        if use_full_pointcloud:
            distances = torch.cdist(ply_points, self.points)
            nearest_indices = torch.argmin(distances, dim=1)
            full_assign_matrix = self.assign_matrix[:, nearest_indices]
            self.assign_matrix = full_assign_matrix
            self.points = ply_points

        distances = torch.cdist(self.points, ply_points)
        closest_ply_indices = torch.argmin(distances, dim=1)
        self.normals = self.normals[closest_ply_indices]

        outside_points = []
        self.line_length = 0.01
        tmp_points, tmp_normals = self.points, self.normals
        decay_rate = 0.5
        for i in range(3):
            tmp_points = tmp_points + (tmp_normals * self.line_length)
            distances = torch.cdist(tmp_points, og_points) # using all points leads to OOM
            distances = torch.min(distances, dim=1).values
            mask = (distances >= (self.line_length * (i+1)) - 1e-4)
            
            valid_indices = torch.nonzero(mask).squeeze()
            num_valid = valid_indices.numel()
            num_to_sample = max(1, int(num_valid * (decay_rate if i > 0 else 1)))
            selected_indices = valid_indices[torch.randperm(num_valid)[:num_to_sample]]
            tmp_points, tmp_normals = tmp_points[selected_indices], tmp_normals[selected_indices]
            outside_points.append(tmp_points)
        self.outside_points = torch.cat(outside_points, dim=0)
        print(f"Using {self.outside_points.shape[0]} outside points")

        self.pred_handler = pred_handler
        self.truncation = truncation
        self.device = device

        self.N = self.mask.sum()
        print(f"Loaded {self.N} superquadircs.")

        # Turn selected attributes into trainable parameters
        for name in trainable:
            val = getattr(self, name)
            setattr(self, name, nn.Parameter(val))
        
        self.trainable_params = self.trainable_params()
        print("Trainable superq params:", self.trainable_params)

    def scale(self):
        # Enforce positive scale
        return torch.exp(self.raw_scale)

    def exponents(self):
        # Enforce exponent contraints for numerical stability
        minE, maxE = 0.1, 1.9
        return (torch.sigmoid(self.raw_exponents) * (maxE - minE)) + minE

    def rotation(self):
        # Convert back to rotation matrix
        return quat2mat(self.raw_rotation)

    def trainable_params(self):
        names = []
        params = dict(self.named_parameters())
        for k, v in params.items():
            if isinstance(v, torch.Tensor):
                names.append(k)
        return names

    def get_param_groups(self):
        groups = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "raw_exponents" in name:
                lr = 1e-2
            elif "raw_scale" in name:
                lr = 5e-2
            else:
                lr =  1e-3
            groups.append({"params": [param], "lr": lr})
        return groups

    def update_handler(self):
        batch_size = self.pred_handler.scale.shape[1]
        mask = self.mask.reshape(batch_size)
        self.pred_handler.scale[self.idx][mask] = self.scale().detach().cpu().numpy()
        self.pred_handler.exponents[self.idx][mask] = self.exponents().detach().cpu().numpy()
        self.pred_handler.rotation[self.idx][mask] = self.rotation().detach().cpu().numpy()
        self.pred_handler.translation[self.idx][mask] = self.translation.detach().cpu().numpy()
        meshes = self.pred_handler.get_meshes(resolution=30)
        return self.pred_handler, meshes

    def sdf(self, idx, points):
        # 1. Transform points to local coordinate system
        # X = R' * (points - t)
        # Note: rotation_matrix.T is equivalent to R'
        points_centered = points - self.translation[idx][:, None]
        X = self.rotation()[idx].T @ points_centered
        
        # Fix numerical instability at 0
        x, y, z = X[0, :], X[1, :], X[2, :]
        x = ((x > 0).float() * 2 - 1) * torch.max(torch.abs(x), x.new_tensor(1e-6))
        y = ((y > 0).float() * 2 - 1) * torch.max(torch.abs(y), x.new_tensor(1e-6))
        z = ((z > 0).float() * 2 - 1) * torch.max(torch.abs(z), x.new_tensor(1e-6))
        X = torch.stack([x, y, z], dim=0)

        # 2. Extract parameters for readability
        e1, e2 = self.exponents()[idx]
        sx, sy, sz = self.scale()[idx]

        # 3. Calculate radial distance from origin
        r0 = torch.linalg.norm(X, axis=0)

        # 4. Calculate the Superquadric scaling function
        # Formula components: (((x/sx)^2)^(1/e2) + ((y/sy)^2)^(1/e2))^(e2/e1) + ((z/sz)^2)^(1/e1)
        term1 = safe_pow(safe_pow(x / sx, 2), 1 / e2)
        term2 = safe_pow(safe_pow(y / sy, 2), 1 / e2)
        term3 = safe_pow(safe_pow(z / sz, 2), 1 / e1)
        f = safe_pow(safe_pow(term1 + term2, e2 / e1) + term3, -e1 / 2)

        # 5. Compute Signed Distance
        sdf = safe_mul(r0, (1 - f))

        # 6. Apply truncation
        if self.truncation != 0:
            sdf = torch.clip(sdf, -self.truncation, self.truncation)

        return sdf


    def forward(self):
        all_points = torch.cat([self.points, self.outside_points], dim=0)
        sdf_values = self.sdf(0, all_points.T)
        for i in range(1, self.N):
            sdf_values = torch.minimum(self.sdf(i, all_points.T), sdf_values)
        return sdf_values[:self.points.shape[0]], sdf_values[self.points.shape[0]:]