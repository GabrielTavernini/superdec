import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import open3d as o3d
import math
import os

# Enable TensorFloat32 for better performance on Ampere+ GPUs
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')

from typing import Optional, assert_never
from dataclasses import dataclass
import gc

from ..utils import quat2mat, mat2quat
from superdec.utils.safe_operations import safe_pow, safe_mul
from superdec.utils.predictions_handler_extended import PredictionHandler

class BatchSuperQMulti(nn.Module):
    def __init__(
        self, 
        pred_handler: PredictionHandler,
        indices: list[int],
        ply_paths: list[str] = None,
        truncation: float = 0.05,
        device: str = "cuda",
    ):
        super().__init__()
        self.indices = indices
        self.device = device
        self.truncation = truncation
        self.pred_handler = pred_handler
        self.minS = 0.01
        self.minE, self.maxE = 0.1, 1.9

        B = len(indices)
        self.N_max = pred_handler.scale.shape[1]
        self.M_points = 100_000

        scale_list = []
        exp_list = []
        rot_list = []
        trans_list = []
        self.masks = [] 

        self.points = torch.zeros(B, self.M_points, 3, device=device)
        self.occupancies = torch.zeros(B, self.M_points, dtype=torch.bool, device=device)
        
        for i, idx in enumerate(indices):
            # --- Params ---
            mask = (pred_handler.exist[idx] > 0.5)
            self.masks.append(torch.tensor(mask, dtype=torch.bool, device=device).reshape(-1))

            s = torch.tensor(pred_handler.scale[idx], dtype=torch.float, device=device).reshape(-1, 3)
            e = torch.tensor(pred_handler.exponents[idx], dtype=torch.float, device=device).reshape(-1, 2)
            r = torch.tensor(pred_handler.rotation[idx], dtype=torch.float, device=device).reshape(-1, 3, 3)
            t = torch.tensor(pred_handler.translation[idx], dtype=torch.float, device=device).reshape(-1, 3)
            
            s[~mask.reshape(-1)] = 1.0 
            scale_list.append(torch.log(torch.clamp(s - self.minS, min=1e-6)))
            e = torch.clamp(e, self.minE + 1e-4, self.maxE - 1e-4)
            exp_list.append(torch.logit((e - self.minE) / (self.maxE - self.minE)))
            rot_list.append(mat2quat(r))
            trans_list.append(t)
            
            # --- Points ---
            try:
                ply = ply_paths[i] if ply_paths else None
                points_file = ply.replace("pointcloud.npz", "points.npz")
                points_dict = np.load(points_file)
                points_iou = points_dict['points']
                occ_tgt = points_dict['occupancies']
                if np.issubdtype(occ_tgt.dtype, np.uint8):
                    occ_tgt = np.unpackbits(occ_tgt)[:points_iou.shape[0]]
                pts = torch.tensor(points_iou, dtype=torch.float, device=device)
                occ = torch.tensor(occ_tgt, dtype=torch.bool, device=device)
            except Exception as e:
                print(f"Error loading {points_file}: {e}")
                exit()
            
            # Ensure pts has correct shape if not loaded from ply
            if pts.shape[0] != self.M_points and occ.shape[0] != self.M_points:
                print(f"Error points shape missmatch")
                exit()
            
            self.points[i] = pts
            self.occupancies[i] = occ
            
        self.raw_scale = nn.Parameter(torch.stack(scale_list)) # (B, N, 3)
        self.raw_exponents = nn.Parameter(torch.stack(exp_list)) # (B, N, 2)
        self.raw_rotation = nn.Parameter(torch.stack(rot_list)) # (B, N, 4)
        self.translation = nn.Parameter(torch.stack(trans_list)) # (B, N, 3)
        self.raw_tapering = nn.Parameter(torch.full((B, self.N_max, 2), 1e-4, dtype=torch.float, device=device))
        
        self.exist_mask = torch.stack(self.masks) # (B, N)

    @torch.compile
    def scale(self):
        return torch.exp(self.raw_scale) + self.minS

    @torch.compile   
    def exponents(self):
        return (torch.sigmoid(self.raw_exponents) * (self.maxE - self.minE)) + self.minE

    @torch.compile  
    def rotation(self):
        return quat2mat(self.raw_rotation)

    @torch.compile 
    def tapering(self):
        return torch.tanh(self.raw_tapering)

    def get_param_groups(self):
        lrs = {
            "raw_scale": 2e-2,
            "raw_exponents": 1e-2,
            "raw_tapering": 5e-4,
        }
        groups = []
        for name, param in self.named_parameters():
            if not param.requires_grad: continue
            lr = lrs.get(name, 2e-3)
            groups.append({"params": [param], "lr": lr})
        return groups
    
    @torch.compile
    def sdf_batch(self, points):
        B, _, M = points.shape
        N = self.N_max
        
        points_expanded = points.unsqueeze(1) # (B, 1, 3, M)
        t = self.translation.unsqueeze(-1) # (B, N, 3, 1)
        points_centered = points_expanded - t # (B, N, 3, M)
        
        X = torch.matmul(self.rotation().transpose(-2, -1), points_centered) # (B, N, 3, M)
        
        e1 = self.exponents()[..., 0].unsqueeze(-1)
        e2 = self.exponents()[..., 1].unsqueeze(-1)
        sx = self.scale()[..., 0].unsqueeze(-1)
        sy = self.scale()[..., 1].unsqueeze(-1)
        sz = self.scale()[..., 2].unsqueeze(-1)
        
        x = X[:, :, 0, :]
        y = X[:, :, 1, :]
        z = X[:, :, 2, :]
        
        eps = 1e-6
        # Use torch.where and clamp for speed and avoiding tensor creation
        # Original: ((x > 0).float() * 2 - 1) maps >0 to 1, <=0 to -1
        x = torch.where(x > 0, 1.0, -1.0) * torch.clamp(torch.abs(x), min=eps)
        y = torch.where(y > 0, 1.0, -1.0) * torch.clamp(torch.abs(y), min=eps)
        z = torch.where(z > 0, 1.0, -1.0) * torch.clamp(torch.abs(z), min=eps)
        
        kx = self.tapering()[..., 0].unsqueeze(-1)
        ky = self.tapering()[..., 1].unsqueeze(-1)
        
        fx = safe_mul(kx/sz, z) + 1
        fy = safe_mul(ky/sz, z) + 1
        
        fx = torch.where(fx > 0, 1.0, -1.0) * torch.clamp(torch.abs(fx), min=eps)
        fy = torch.where(fy > 0, 1.0, -1.0) * torch.clamp(torch.abs(fy), min=eps)
        
        x = x / fx
        y = y / fy
        
        # r0 = torch.sqrt(x**2 + y**2 + z**2)
        
        term1 = safe_pow(safe_pow(x / sx, 2), 1 / e2)
        term2 = safe_pow(safe_pow(y / sy, 2), 1 / e2)
        term3 = safe_pow(safe_pow(z / sz, 2), 1 / e1)
        
        f_func = safe_pow(safe_pow(term1 + term2, e2 / e1) + term3, -e1 / 2)
        
        # sdf = safe_mul(r0, (1 - f_func))
        sdf = (1 - f_func)
        return sdf
    
    def compute_losses(self, forward_out):
        sdfs = forward_out.get('sdfs')
        
        temperature = 1e-3
        pred_occ = torch.sigmoid(-sdfs / temperature)
        gt_occ = self.occupancies.float()

        intersection = (pred_occ * gt_occ).sum(dim=1)
        union = (pred_occ + gt_occ - pred_occ * gt_occ).sum(dim=1)
        iou = intersection / torch.clamp(union, min=1.0)

        # Regularization term per primitive
        # mask = self.exist_mask.float()
        # Lreg_b = torch.norm(self.scale(), p=2, dim=2) # (B, N)
        # Lreg = 2e-2 * (Lreg_b * mask).mean(dim=1) # (B,)

        mask = self.exist_mask.float()
        Ltap_b = torch.norm(torch.abs(self.tapering()), p=1, dim=2) # (B, N)
        Ltap = 2e-1 * (Ltap_b * mask).mean(dim=1) # (B,)
        
        loss = -torch.log(iou) + Ltap #+ Lreg
        return loss, {"iou": iou, "tap": Ltap}
    
    def forward(self):
        all_sdfs = self.sdf_batch(self.points.transpose(1, 2)) # (B, N, M_total)
        
        # mask out non-existing primitives
        mask = self.exist_mask.unsqueeze(-1).expand_as(all_sdfs)
        all_sdfs[~mask] = float('inf')

        # Union is min(sdf)
        min_sdf, _ = torch.min(all_sdfs, dim=1) # (B, M)
        return {
            'sdfs': min_sdf,
        }

    def update_handler(self, compute_meshes=True):
         for i, idx in enumerate(self.indices):
             mask = self.exist_mask[i].cpu().numpy()
             if not np.any(mask): continue
             
             self.pred_handler.scale[idx][mask] = self.scale()[i][mask].detach().cpu().numpy()
             self.pred_handler.exponents[idx][mask] = self.exponents()[i][mask].detach().cpu().numpy()
             self.pred_handler.tapering[idx][mask] = self.tapering()[i][mask].detach().cpu().numpy()
             self.pred_handler.rotation[idx][mask] = self.rotation()[i][mask].detach().cpu().numpy()
             self.pred_handler.translation[idx][mask] = self.translation[i][mask].detach().cpu().numpy()
             
         if compute_meshes:
             return self.pred_handler, self.pred_handler.get_meshes(resolution=30)
         else:
             return self.pred_handler

