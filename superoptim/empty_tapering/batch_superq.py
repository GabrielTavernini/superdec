import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import open3d as o3d
import math
import os
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
        truncation: float = 0.1,
        ply_paths: list[str] = None,
        device: str = "cuda",
    ):
        super().__init__()
        self.indices = indices
        self.device = device
        self.truncation = truncation
        self.pred_handler = pred_handler
        self.minE, self.maxE = 0.1, 1.9

        B = len(indices)
        self.N_max = pred_handler.scale.shape[1]
        
        self.points_list = []
        self.normals_list = []
        self.outside_points_list = []
        
        scale_list = []
        exp_list = []
        rot_list = []
        trans_list = []
        self.masks = [] 

        max_M = 0
        max_K = 0

        for i, idx in enumerate(indices):
            # --- Params ---
            mask = (pred_handler.exist[idx] > 0.5)
            self.masks.append(torch.tensor(mask, dtype=torch.bool, device=device).reshape(-1))

            s = torch.tensor(pred_handler.scale[idx], dtype=torch.float, device=device).reshape(-1, 3)
            e = torch.tensor(pred_handler.exponents[idx], dtype=torch.float, device=device).reshape(-1, 2)
            r = torch.tensor(pred_handler.rotation[idx], dtype=torch.float, device=device).reshape(-1, 3, 3)
            t = torch.tensor(pred_handler.translation[idx], dtype=torch.float, device=device).reshape(-1, 3)
            
            s[~mask.reshape(-1)] = 1.0 
            scale_list.append(torch.log(s))
            e = torch.clamp(e, self.minE + 1e-4, self.maxE - 1e-4)
            exp_list.append(torch.logit((e - self.minE) / (self.maxE - self.minE)))
            rot_list.append(mat2quat(r))
            trans_list.append(t)
            
            # --- Points ---
            ply = ply_paths[i] if ply_paths else None
            pts_default = torch.tensor(pred_handler.pc[idx], dtype=torch.float, device=device)
            pts = pts_default
            nrms = None

            if ply and ply.endswith("npz") and os.path.exists(ply):
                try:
                    data = np.load(ply)
                    pts_ply = torch.tensor(np.array(data['points']), dtype=torch.float, device=device) 
                    ply_normals = torch.tensor(np.array(data['normals']), dtype=torch.float, device=device) 
                    distances = torch.cdist(pts_default, pts_ply)
                    closest = torch.argmin(distances, dim=1)
                    pts = pts_default
                    nrms = ply_normals[closest]
                except Exception as e:
                    print(f"Error loading {ply}: {e}")
            
            if nrms is None:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts.detach().cpu().numpy())
                pcd.estimate_normals()
                nrms = torch.tensor(np.array(pcd.normals), dtype=torch.float, device=device)
            
            self.points_list.append(pts)
            self.normals_list.append(nrms)
            max_M = max(max_M, pts.shape[0])

            # --- Outside Points ---
            out_pts_list_local = []
            line_length = 0.01
            decay_rate = 0.5
            tmp_pts, tmp_nrms = pts, nrms
            og_pts = pts
            
            for step in range(3):
                tmp_pts = tmp_pts + (tmp_nrms * line_length)
                dists = torch.cdist(tmp_pts, og_pts)
                dists = torch.min(dists, dim=1).values
                m = (dists >= (line_length * (step+1)) - 1e-4)
                valid_idx = torch.nonzero(m).squeeze()
                
                if valid_idx.numel() > 0:
                    num_valid = valid_idx.numel()
                    num_to_sample = max(1, int(num_valid * (decay_rate if step > 0 else 1)))
                    sel = valid_idx[torch.randperm(num_valid)[:num_to_sample]]
                    tmp_pts, tmp_nrms = tmp_pts[sel], tmp_nrms[sel]
                    out_pts_list_local.append(tmp_pts)
            
            if out_pts_list_local:
                out_pts = torch.cat(out_pts_list_local, dim=0)
            else:
                out_pts = torch.empty((0, 3), device=device)
            
            self.outside_points_list.append(out_pts)
            max_K = max(max_K, out_pts.shape[0])

        self.raw_scale = nn.Parameter(torch.stack(scale_list)) # (B, N, 3)
        self.raw_exponents = nn.Parameter(torch.stack(exp_list)) # (B, N, 2)
        self.raw_rotation = nn.Parameter(torch.stack(rot_list)) # (B, N, 4)
        self.translation = nn.Parameter(torch.stack(trans_list)) # (B, N, 3)
        self.raw_tapering = nn.Parameter(torch.full((B, self.N_max, 2), 1e-4, dtype=torch.float, device=device))
        
        self.exist_mask = torch.stack(self.masks) # (B, N)
        
        self.points = torch.zeros(B, max_M, 3, device=device)
        self.outside_points = torch.zeros(B, max_K, 3, device=device)
        self.points_valid_mask = torch.zeros(B, max_M, dtype=torch.bool, device=device)
        self.outside_valid_mask = torch.zeros(B, max_K, dtype=torch.bool, device=device)

        for i in range(B):
            n = self.points_list[i].shape[0]
            self.points[i, :n] = self.points_list[i]
            self.points_valid_mask[i, :n] = True
            
            n = self.outside_points_list[i].shape[0]
            if n > 0:
                self.outside_points[i, :n] = self.outside_points_list[i]
                self.outside_valid_mask[i, :n] = True

    def scale(self):
        return torch.exp(self.raw_scale) + 1e-6
        
    def exponents(self):
        return (torch.sigmoid(self.raw_exponents) * (self.maxE - self.minE)) + self.minE
        
    def rotation(self):
        return quat2mat(self.raw_rotation)
        
    def tapering(self):
        return torch.tanh(self.raw_tapering)

    def get_param_groups(self):
        lrs = {
            "raw_scale": 5e-2,
            "raw_exponents": 1e-2,
            "raw_tapering": 5e-4,
        }
        groups = []
        for name, param in self.named_parameters():
            if not param.requires_grad: continue
            lr = lrs.get(name, 1e-3)
            groups.append({"params": [param], "lr": lr})
        return groups
        
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
        x = ((x > 0).float() * 2 - 1) * torch.max(torch.abs(x), torch.tensor(eps, device=x.device))
        y = ((y > 0).float() * 2 - 1) * torch.max(torch.abs(y), torch.tensor(eps, device=y.device))
        z = ((z > 0).float() * 2 - 1) * torch.max(torch.abs(z), torch.tensor(eps, device=z.device))
        
        kx = self.tapering()[..., 0].unsqueeze(-1)
        ky = self.tapering()[..., 1].unsqueeze(-1)
        
        fx = safe_mul(kx/sz, z) + 1
        fy = safe_mul(ky/sz, z) + 1
        
        fx = ((fx > 0).float() * 2 - 1) * torch.max(torch.abs(fx), torch.tensor(eps, device=fx.device))
        fy = ((fy > 0).float() * 2 - 1) * torch.max(torch.abs(fy), torch.tensor(eps, device=fy.device))
        
        x = x / fx
        y = y / fy
        
        r0 = torch.sqrt(x**2 + y**2 + z**2)
        
        term1 = safe_pow(safe_pow(x / sx, 2), 1 / e2)
        term2 = safe_pow(safe_pow(y / sy, 2), 1 / e2)
        term3 = safe_pow(safe_pow(z / sz, 2), 1 / e1)
        
        f_func = safe_pow(safe_pow(term1 + term2, e2 / e1) + term3, -e1 / 2)
        
        sdf = safe_mul(r0, (1 - f_func))
        return sdf
        
    def forward(self):
        split_idx = self.points.shape[1]
        all_points = torch.cat([self.points, self.outside_points], dim=1) 
        all_sdfs = self.sdf_batch(all_points.transpose(1, 2)) # (B, N, M_total)
        
        # Use a large finite number instead of inf to avoid NaN in 0 * inf during weighted sum
        large_val = 1e6
        mask_expanded = self.exist_mask.unsqueeze(-1)
        all_sdfs = torch.where(mask_expanded, all_sdfs, torch.tensor(large_val, device=all_sdfs.device))
        
        all_sdfs_clipped = torch.clip(all_sdfs, -self.truncation, self.truncation)
        all_sdfs_leaky = all_sdfs_clipped + 0.1 * (all_sdfs - all_sdfs_clipped)
        
        sdfs_points = all_sdfs[:, :, :split_idx] 
        leaky_points = all_sdfs_leaky[:, :, :split_idx]
        
        logits = -100.0 * sdfs_points
        weights = F.softmax(logits, dim=1) 
        values_points = torch.sum(weights * leaky_points, dim=1) 
        
        idx_points = torch.argmin(sdfs_points, dim=1) 
        counts_points = torch.zeros(all_sdfs.shape[0], all_sdfs.shape[1], device=all_sdfs.device)
        for b in range(all_sdfs.shape[0]):
            valid_p = self.points_valid_mask[b]
            if valid_p.any():
                counts_points[b] = torch.bincount(idx_points[b][valid_p], minlength=self.N_max).float()
        
        sdfs_outside_leaky = all_sdfs_leaky[:, :, split_idx:]
        values_outside, idx_outside = torch.min(sdfs_outside_leaky, dim=1) 
        
        counts_outside = torch.zeros_like(counts_points)
        for b in range(all_sdfs.shape[0]):
            valid_o = self.outside_valid_mask[b]
            if valid_o.any():
                 v_out = values_outside[b][valid_o]
                 idx_out = idx_outside[b][valid_o]
                 neg_mask = v_out < 0
                 if neg_mask.any():
                     counts_outside[b] = torch.bincount(idx_out[neg_mask], minlength=self.N_max).float()
                     
        return values_points, values_outside, counts_points, counts_outside

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

