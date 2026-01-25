import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import open3d as o3d
import math
from typing import Optional, assert_never
from dataclasses import dataclass

from ..utils import quat2mat, mat2quat
from superdec.utils.predictions_handler import PredictionHandler

class SuperQ(nn.Module):
    def __init__(
        self, 
        pred_handler: PredictionHandler,
        truncation: float = 0.1,
        use_segmentation: bool = False,
        ply: str = None,
        idx: int = 0,
        device: str = "cuda",
    ):
        # Anything self.x = nn.Parameter(...) is trainable
        self.minE, self.maxE = 0.1, 1.9
        trainable = ["raw_scale", "raw_exponents", "raw_rotation", "translation"]

        super().__init__()
        self.idx = idx
        self.mask = (pred_handler.exist[self.idx] > 0.5).reshape(-1)
        raw_scale = torch.tensor(pred_handler.scale[self.idx].reshape(-1, 3)[self.mask], dtype=torch.float, device=device)
        self.raw_scale = torch.log(raw_scale)
        raw_exponents = torch.tensor(pred_handler.exponents[self.idx].reshape(-1, 2)[self.mask], dtype=torch.float, device=device)
        self.raw_exponents = torch.logit((raw_exponents - self.minE) / (self.maxE - self.minE))
        rot_mat = torch.tensor(pred_handler.rotation[self.idx].reshape(-1, 3, 3)[self.mask], dtype=torch.float, device=device)
        self.raw_rotation = mat2quat(rot_mat) # Shape (N, 3)
        self.translation = torch.tensor(pred_handler.translation[self.idx].reshape(-1, 3)[self.mask], dtype=torch.float, device=device)
        
        self.use_segmentation = use_segmentation
        self.assign_matrix = torch.tensor(pred_handler.assign_matrix[self.idx].T[self.mask], dtype=torch.float, device=device).squeeze() # [N, 4096]
        self.points = torch.tensor(np.array(pred_handler.get_segmented_pcs()[self.idx].points), dtype=torch.float, device=device) # [4096, 3]
        
        if ply:
            data = np.load(ply)
            ply_points = torch.tensor(np.array(data['points']), dtype=torch.float, device=device) 
            self.normals = torch.tensor(np.array(data['normals']), dtype=torch.float, device=device) 
            
            distances = torch.cdist(self.points, ply_points)
            closest_ply_indices = torch.argmin(distances, dim=1)
            self.normals = self.normals[closest_ply_indices]

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
        return (torch.sigmoid(self.raw_exponents) * (self.maxE - self.minE)) + self.minE

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
                lr =  5e-2
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

    def sdf(self, idx):
        # 1. Transform points to local coordinate system
        # X = R' * (points - t)
        # Note: rotation_matrix.T is equivalent to R'
        if self.use_segmentation:
            p_mask = (self.assign_matrix[idx] == 1)
            points_raw = self.points[p_mask].T
        else:
            points_raw = self.points.T
            
        points = points_raw.detach().clone().requires_grad_(True)
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
        term1 = ((x / sx)**2)**(1 / e2)
        term2 = ((y / sy)**2)**(1 / e2)
        term3 = ((z / sz)**2)**(1 / e1)
        f = ( (term1 + term2 + 1e-6)**(e2 / e1) + term3 )**(-e1 / 2)

        # 5. Compute Signed Distance
        sdf = r0 * (1 - f)

        d_points = torch.autograd.grad(
            outputs=sdf,
            inputs=points,
            grad_outputs=torch.ones_like(sdf),
            create_graph=False, # Set True if you need higher-order derivatives (Hessian)
            retain_graph=True,
            only_inputs=True
        )[0]
        normals = torch.nn.functional.normalize(d_points, dim=0).T

        # 6. Apply truncation
        if self.truncation != 0:
            sdf = torch.clip(sdf, -self.truncation, self.truncation)

        return sdf, normals


    def forward(self):
        if self.use_segmentation:
            sdf_values = torch.zeros(self.points.shape[0], device=self.device)
            normals = torch.zeros(self.points.shape, device=self.device)
            for i in range(self.N):
                p_mask = (self.assign_matrix[i] == 1)
                sdf_values[p_mask], normals[p_mask] = self.sdf(i)
        else:
            sdf_values, normals = self.sdf(0)
            for i in range(1, self.N):
                curr_sdf, curr_normals = self.sdf(i)
                mask = curr_sdf < sdf_values
                sdf_values = torch.where(mask, curr_sdf, sdf_values)
                normals = torch.where(mask.unsqueeze(1), curr_normals, normals)
        return sdf_values, normals
        