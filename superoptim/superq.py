import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import open3d as o3d
import math
from typing import Optional, assert_never
from dataclasses import dataclass

from superdec.utils.predictions_handler import PredictionHandler

class SuperQ(nn.Module):
    def __init__(
        self, 
        pred_handler: PredictionHandler,
        truncation: float = 0.1,
        ply: str = None,
        device: str = "cuda",
    ):
        # Anything self.x = nn.Parameter(...) is trainable
        # trainable = ["sqscale", "exponents", "translation", "rotation"]
        trainable = ["raw_sqscale", "raw_exponents", "raw_rotation", "translation"]

        super().__init__()
        self.mask = (pred_handler.exist > 0.5).reshape(-1)
        self.raw_sqscale = torch.tensor(pred_handler.scale.reshape(-1, 3)[self.mask], dtype=torch.float, device=device)
        self.raw_exponents = torch.tensor(pred_handler.exponents.reshape(-1, 2)[self.mask], dtype=torch.float, device=device)
        # On the Continuity of Rotation Representations in Neural Networks (Zhou et al.)
        rot_mat = torch.tensor(pred_handler.rotation.reshape(-1, 3, 3)[self.mask], dtype=torch.float, device=device)
        self.raw_rotation = rot_mat[:, :, :2].clone() # Shape (N, 3, 2)
        self.translation = torch.tensor(pred_handler.translation.reshape(-1, 3)[self.mask], dtype=torch.float, device=device)
        
        self.assign_matrix = torch.tensor(pred_handler.assign_matrix.T[self.mask], dtype=torch.float, device=device).squeeze() # [N, 4096]
        self.points = torch.tensor(np.array(pred_handler.get_segmented_pcs()[0].points), dtype=torch.float, device=device) # [4096, 3]
        
        if ply:
            pcd = o3d.io.read_point_cloud(ply) 
            ply_points = torch.tensor(np.array(pcd.points), dtype=torch.float, device=device) 
            distances = torch.cdist(ply_points, self.points)
            nearest_indices = torch.argmin(distances, dim=1)
            full_assign_matrix = self.assign_matrix[:, nearest_indices]
            self.assign_matrix = full_assign_matrix
            self.points = ply_points

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

    def sqscale(self):
        # Enforce positive scales > 1e-5
        return torch.clamp(self.raw_sqscale, min=1e-5)

    def exponents(self):
        # Enforce exponent contraints for numerical stability
        return torch.clamp(self.raw_exponents, min=0.01)

    def rotation(self):
        """
        Converts the stored 6D vectors back to a valid 3x3 rotation matrix 
        using Gram-Schmidt orthogonalization.
        """
        a1 = self.raw_rotation[:, :, 0]
        a2 = self.raw_rotation[:, :, 1]

        # 1. Normalize the first vector
        b1 = F.normalize(a1, dim=1)

        # 2. Project second vector onto the first to make it orthogonal
        # b2 = a2 - (b1 . a2) * b1
        dot_prod = torch.sum(b1 * a2, dim=1, keepdim=True)
        b2 = a2 - dot_prod * b1
        b2 = F.normalize(b2, dim=1)

        # 3. Compute the third vector using cross product
        b3 = torch.cross(b1, b2, dim=1)

        # Stack columns to form rotation matrix
        return torch.stack([b1, b2, b3], dim=-1)

    def trainable_params(self):
        names = []
        params = dict(self.named_parameters())
        for k, v in params.items():
            if isinstance(v, torch.Tensor):
                names.append(k)
        return names

    def update_handler(self):
        batch_size = self.pred_handler.scale.shape[1]
        mask = self.mask.reshape(-1, batch_size)
        self.pred_handler.scale[mask] = self.sqscale().detach().cpu().numpy()
        self.pred_handler.exponents[mask] = self.exponents().detach().cpu().numpy()
        self.pred_handler.rotation[mask] = self.rotation().detach().cpu().numpy()
        self.pred_handler.translation[mask] = self.translation.detach().cpu().numpy()
        meshes = self.pred_handler.get_meshes(resolution=30)
        return self.pred_handler, meshes

    def sdf(self, idx):
        """
        Computes the Signed Distance Function for a Superquadric.
        
        Args:
            points: (3, N) array of query points.
            scale_vec: (3,) array [sx, sy, sz].
            exponents: (2,) array [e1, e2].
            translation: (3,) array [tx, ty, tz].
            rotation_matrix: (3, 3) rotation matrix.
            truncation: float, distance limit (0 to disable).
        """
        # 1. Transform points to local coordinate system
        # X = R' * (points - t)
        # Note: rotation_matrix.T is equivalent to R'
        p_mask = (self.assign_matrix[idx] == 1)
        points = self.points[p_mask].T
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
        sx, sy, sz = self.sqscale()[idx]

        # 3. Calculate radial distance from origin
        r0 = torch.linalg.norm(X, axis=0)

        # 4. Calculate the Superquadric scaling function
        # Formula components: (((x/sx)^2)^(1/e2) + ((y/sy)^2)^(1/e2))^(e2/e1) + ((z/sz)^2)^(1/e1)
        term1 = ((x / sx)**2)**(1 / e2)
        term2 = ((y / sy)**2)**(1 / e2)
        term3 = ((z / sz)**2)**(1 / e1)
        f = ( (term1 + term2)**(e2 / e1) + term3 )**(-e1 / 2)

        # 5. Compute Signed Distance
        sdf = r0 * (1 - f)

        # 6. Apply truncation
        if self.truncation != 0:
            sdf = torch.clip(sdf, -self.truncation, self.truncation)

        return sdf


    def forward(self):
        sdf_values = torch.zeros(self.points.shape[0], device=self.device)
        for i in range(self.N):
            v = self.sdf(i)
            p_mask = (self.assign_matrix[i] == 1)
            sdf_values[p_mask] = v
        
        return sdf_values
        