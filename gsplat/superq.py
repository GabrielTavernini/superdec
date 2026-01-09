import torch
import torch.nn as nn
import numpy as np
import open3d as o3d
import math
from typing import Optional, assert_never
from dataclasses import dataclass

from utils import estimate_sq_surface_areas
from gsplat.strategy.ops import remove

from superdec.loss.sampler import EqualDistanceSamplerSQ
from superdec.utils.predictions_handler import PredictionHandler

@dataclass
class SuperQConfig():
    # LR for 3D point positions of bg
    means_lr: float = .8e-4
    # LR for 3D offsets
    offsets_lr: float = 1.6e-4
    # LR for superq parameters
    superq_lr: float = .8e-4
    # LR for superq eta omegas
    superq_pos_lr: float = 5e-4
    # Allow gaussians to move on the surface
    move_on_sq: bool = False
    # Max offset from attachement point
    max_offset: float = 0.2

class SuperQ(nn.Module):
    def __init__(
        self, 
        cfg: SuperQConfig,
        pred_handler: PredictionHandler,
        target_pts_density: int = 2000,
        min_pts_per_sq: int = 1000,
        num_pts_background: int = -1,
        background_ply: Optional[str] = None,
        device: str = "cuda",
    ):
        # Anything self.x = nn.Parameter(...) is trainable
        trainable = ["background", "offsets", "sqscale", "exponents", "translation", "rotation"]
        # trainable = ["background", "offsets", "sqscale", "translation", "rotation"]
        # trainable = ["background", "offsets"]

        super().__init__()
        self.mask = (pred_handler.exist > 0.5).reshape(-1)
        self.sqscale = torch.tensor(pred_handler.scale.reshape(-1, 3)[self.mask], dtype=torch.float, device=device)
        self.exponents = torch.tensor(pred_handler.exponents.reshape(-1, 2)[self.mask], dtype=torch.float, device=device)
        self.translation = torch.tensor(pred_handler.translation.reshape(-1, 3)[self.mask], dtype=torch.float, device=device)
        self.rotation = torch.tensor(pred_handler.rotation.reshape(-1, 3, 3)[self.mask], dtype=torch.float, device=device)
        self.pred_handler = pred_handler
        self.cfg = cfg

        N = self.mask.sum()
        print(f"Loaded {N} superquadircs.")

        # Calculate specific points per SQ
        areas = estimate_sq_surface_areas(self).detach().cpu().numpy()  # [S]
        num_pts_per_sq_area_based = np.round(areas * target_pts_density).astype(int)
        num_pts_per_sq_area_based[num_pts_per_sq_area_based < min_pts_per_sq] = min_pts_per_sq
        print(f"Sampling {np.sum(num_pts_per_sq_area_based)} total points. Min: {num_pts_per_sq_area_based.min()}, Max: {num_pts_per_sq_area_based.max()}")

        max_samples = int(np.max(num_pts_per_sq_area_based))
        sampler = EqualDistanceSamplerSQ(n_samples=max_samples, D_eta=0.05, D_omega=0.05)
        raw_etas, raw_omegas = sampler.sample_on_batch(
            pred_handler.scale.astype(np.float32),
            pred_handler.exponents.astype(np.float32)
        )
        raw_etas = raw_etas.reshape(-1, max_samples)[self.mask]
        raw_omegas = raw_omegas.reshape(-1, max_samples)[self.mask]
        
        # filter out excess points
        col_indices = np.arange(max_samples)[None, :]
        required_counts = num_pts_per_sq_area_based[:, None]
        valid_mask = col_indices < required_counts # [S, max_samples]
        etas = raw_etas[valid_mask]
        omegas = raw_omegas[valid_mask]

        # Make sure we don't get nan for gradients
        etas[etas == 0] += 1e-6
        omegas[omegas == 0] += 1e-6

        # etas and betas need to be in the state_dict to allow loading from checkpoint
        self.register_buffer("etas", torch.tensor(etas, device=device))
        self.register_buffer("omegas", torch.tensor(omegas, device=device))
        # sq_idx tells us which Superquadric each point belongs to.
        sq_idx_tensor = torch.repeat_interleave(
            torch.arange(self.sqscale.shape[0], device=device), 
            torch.tensor(num_pts_per_sq_area_based, device=device)
        )
        self.register_buffer("sq_idx", sq_idx_tensor)
        self.offsets = torch.zeros(etas.shape[0], 3, device=device)
        if self.cfg.move_on_sq:
            self.surface_offsets = torch.zeros((etas.shape[0], 2), device=device)
            # self.offsets_e = torch.zeros(etas.shape[0], device=device)
            # self.offsets_o = torch.zeros(etas.shape[0], device=device)
            # trainable.extend(["offsets_e", "offsets_o"])
            trainable.extend(["surface_offsets"])

        if background_ply is not None:
            pcd = o3d.io.read_point_cloud(background_ply)
            points_np = np.asarray(pcd.points)
            if num_pts_background != -1:
                idx = np.random.choice(points_np.shape[0], num_pts_background, replace=False)
                points_np = points_np[idx]
            self.background = torch.tensor(points_np, device=device).float()
        else:
            self.background = self._create_background(num_pts_background)

        self._init_heads(N)
        self._forward_params()
        # Turn selected attributes into trainable parameters
        for name in trainable:
            val = getattr(self, name)
            setattr(self, name, nn.Parameter(val))
        
        self.trainable_params_sq = self.trainable_params(False)
        self.trainable_params_bg = self.trainable_params(True)
        print("Trainable superq-tied params:", self.trainable_params_sq)
        print("Trainable background params:", self.trainable_params_bg)

    def get_lr(self, param_name):
        if param_name in ["background"]:
            return self.cfg.means_lr
        elif param_name in ["offsets"]:
            return self.cfg.offsets_lr
        elif param_name in ["sqscale", "exponents", "translation", "rotation"]:
            return self.cfg.superq_lr
        elif param_name in ["offsets_e", "offsets_o", "surface_offsets"]:
            return self.cfg.superq_pos_lr
        else:
            return self.cfg.superq_lr
            assert_never()

    def _init_heads(self, N):
        pass        
        # feature_dim = 32
        # hidden_dim = feature_dim * 2

        # self.features = nn.Parameter(torch.randn(N, feature_dim, device=self.etas.device))
        # self.head_exp = nn.Sequential(
        #     nn.Linear(feature_dim, hidden_dim, device=self.etas.device),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, 2, device=self.etas.device),
        #     nn.Sigmoid(),
        # )
        # self.head_scale = nn.Linear(feature_dim, 3, device=device)
        # self.head_trans = nn.Linear(feature_dim, 3, device=device)
        # self.head_rot = nn.Linear(feature_dim, 9, device=device)

    def _forward_params(self):
        pass
        # self.sqscale = self.head_scale(self.features)
        # self.exponents = self.head_exp(self.features) * 2
        # self.translation = self.head_trans(self.features)
        # rot_flat = self.head_rot(self.features)
        # self.rotation = rot_flat.view(-1, 3, 3)

    def _compute_normals(self):
        current_scale = self.sqscale[self.sq_idx]       # [Ng, 3]
        current_exps = self.exponents[self.sq_idx]      # [Ng, 2]

        # 2. Calculate Local Geometry
        a1, a2, a3 = current_scale[:, 0], current_scale[:, 1], current_scale[:, 2]
        e1, e2 = current_exps[:, 0], current_exps[:, 1]

        cos_eta, sin_eta = torch.cos(self.etas), torch.sin(self.etas)
        cos_omega, sin_omega = torch.cos(self.omegas), torch.sin(self.omegas)

        t1 = self.fexp(cos_eta, e1)
        t2 = self.fexp(sin_eta, e1)
        
        x = a1 * t1 * self.fexp(cos_omega, e2)
        y = a2 * t1 * self.fexp(sin_omega, e2)
        z = a3 * t2
        
        # Fix numerical instability at 0
        x = ((x > 0).float() * 2 - 1) * torch.max(torch.abs(x), x.new_tensor(1e-6))
        y = ((y > 0).float() * 2 - 1) * torch.max(torch.abs(y), x.new_tensor(1e-6))
        z = ((z > 0).float() * 2 - 1) * torch.max(torch.abs(z), x.new_tensor(1e-6))

        # Compute the normals of the SQs
        nx = (torch.cos(self.etas)**2) * (torch.cos(self.omegas)**2) / x
        ny = (torch.cos(self.etas)**2) * (torch.sin(self.omegas)**2) / y
        nz = (torch.sin(self.etas)**2) / z

        normals = torch.stack([nx, ny, nz], -1)
        return normals / torch.linalg.norm(normals)

    def load_dynamic_checkpoint(self, state_dict):
        if "offsets" not in state_dict:
            raise RuntimeError("Checkpoint missing 'offsets' key.")
            
        saved_Ng = state_dict["offsets"].shape[0]
        saved_background_Ng = state_dict["background"].shape[0]
        current_Ng = self.offsets.shape[0]
        current_background_Ng = self.background.shape[0]

        print(f"Resizing model from {current_Ng} points to {saved_Ng} points to match checkpoint.")
        print(f"Resizing background from {current_background_Ng} points to {saved_background_Ng} points to match checkpoint.")

        self.background = nn.Parameter(torch.zeros(saved_background_Ng, 3, device=self.background.device))
        self.offsets = nn.Parameter(torch.zeros(saved_Ng, 3, device=self.offsets.device))
        self.sq_idx = torch.zeros(saved_Ng, dtype=torch.long, device=self.sq_idx.device)
        self.etas = torch.zeros(saved_Ng, device=self.etas.device)
        self.omegas = torch.zeros(saved_Ng, device=self.omegas.device)
        
        self.load_state_dict(state_dict, strict=True)
        
    def update(self, params):
        for k, v in params.items():
            if isinstance(v, torch.Tensor) and "weight" not in k and "bias" not in k:
                setattr(self, k, params[k])

    def trainable_params(self, background):
        names = []
        base = self.background if background else self.etas
        params = dict(self.named_parameters())
        for k, v in params.items():
            if isinstance(v, torch.Tensor) and v.shape[0] == base.shape[0]:
                names.append(k)
        return names

    def densify_buffers(self, mask, split = False):
        """
        Duplicates buffers for SuperQ points based on the mask.
        n_clones: 1 for duplication, 2 for splitting.
        """
        n_clones = 2 if split else 1
        new_etas = self.etas[mask].repeat(n_clones)
        new_omegas = self.omegas[mask].repeat(n_clones)
        new_sq_idx = self.sq_idx[mask].repeat(n_clones)
        
        if split:
            rest = ~mask
        else:
            rest = torch.ones(self.etas.shape[0], dtype=torch.bool)
        self.etas = torch.cat([self.etas[rest], new_etas])
        self.omegas = torch.cat([self.omegas[rest], new_omegas])
        self.sq_idx = torch.cat([self.sq_idx[rest], new_sq_idx])

    def prune_buffers(self, is_prune):
        self.etas = self.etas[~is_prune]
        self.omegas = self.omegas[~is_prune]
        self.sq_idx = self.sq_idx[~is_prune]

    def get_counts(self):
        Ng = self.offsets.shape[0]
        Nbg = self.background.shape[0]
        return Ng, Nbg

    def update_handler(self):
        batch_size = self.pred_handler.scale.shape[1]
        mask = self.mask.reshape(-1, batch_size)
        self.pred_handler.scale[mask] = self.sqscale.detach().cpu().numpy()
        self.pred_handler.exponents[mask] = self.exponents.detach().cpu().numpy()
        self.pred_handler.translation[mask] = self.translation.detach().cpu().numpy()
        self.pred_handler.rotation[mask] = self.rotation.detach().cpu().numpy()
        meshes = self.pred_handler.get_meshes(resolution=30)
        return self.pred_handler, meshes

    def fexp(self, x, p):
        return torch.sign(x)*(torch.abs(x)**p)

    def _create_background(self, num_samples):
        with torch.no_grad():
            points = self._compute_means()
            mean = torch.mean(points, dim=0)
            centered_points = points - mean

            # Eigenvectors represent the rotation matrix of the obb
            covariance = torch.matmul(centered_points.T, centered_points) / (points.shape[0] - 1)
            eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
            transformed_points = torch.matmul(centered_points, eigenvectors)

            min_rot = torch.min(transformed_points, dim=0).values
            max_rot = torch.max(transformed_points, dim=0).values
            lengths = max_rot - min_rot
            
            # Calculate face areas for weighting
            areas = torch.tensor([
                lengths[1] * lengths[2], # Face perp to Local X
                lengths[0] * lengths[2], # Face perp to Local Y
                lengths[0] * lengths[1]  # Face perp to Local Z
            ])
            
            # Pick axes based on area weights
            probs = areas / areas.sum()
            fixed_axes = torch.multinomial(probs, num_samples, replacement=True)
            
            # Generate random points [0,1]
            u = torch.rand(num_samples, 3, device=self.etas.device)
            
            # Snap fixed axis to 0 or 1
            is_max = torch.randint(0, 2, (num_samples,), device=self.etas.device).bool()
            rows = torch.arange(num_samples, device=self.etas.device)
            u[rows, fixed_axes] = is_max.float()
            
            points_local = u * lengths + min_rot
            points_world = torch.matmul(points_local, eigenvectors.T) + mean
            return points_world

    def _compute_means(self):
        self._forward_params()

        # Make sure that all tensors have the right shape
        current_scale = self.sqscale[self.sq_idx]       # [Ng, 3]
        current_exps = self.exponents[self.sq_idx]      # [Ng, 2]
        current_rot = self.rotation[self.sq_idx]        # [Ng, 3, 3]
        current_trans = self.translation[self.sq_idx]   # [Ng, 3]

        # 2. Calculate Local Geometry
        a1, a2, a3 = current_scale[:, 0], current_scale[:, 1], current_scale[:, 2]
        e1, e2 = current_exps[:, 0], current_exps[:, 1]

        cos_eta, sin_eta = torch.cos(self.etas), torch.sin(self.etas)
        cos_omega, sin_omega = torch.cos(self.omegas), torch.sin(self.omegas)

        t1 = self.fexp(cos_eta, e1)
        t2 = self.fexp(sin_eta, e1)
        
        x = a1 * t1 * self.fexp(cos_omega, e2)
        y = a2 * t1 * self.fexp(sin_omega, e2)
        z = a3 * t2
        
        # Fix numerical instability at 0
        x = ((x > 0).float() * 2 - 1) * torch.max(torch.abs(x), x.new_tensor(1e-6))
        y = ((y > 0).float() * 2 - 1) * torch.max(torch.abs(y), x.new_tensor(1e-6))
        z = ((z > 0).float() * 2 - 1) * torch.max(torch.abs(z), x.new_tensor(1e-6))
        local_pos = torch.stack([x, y, z], dim=-1)

        if self.cfg.move_on_sq:
            normals = self._compute_normals()

            # Construct a tangent basis (u, v) from the normal vector n
            # We need an arbitrary vector 'ref' that is not parallel to n
            ref = torch.zeros_like(normals)
            ref[..., 0] = 1.0
            
            # If normal is roughly parallel to X, use Y axis as reference to avoid singularity
            parallel_mask = torch.abs(normals[..., 0]) > 0.9
            ref[parallel_mask] = torch.tensor([0.0, 1.0, 0.0], device=normals.device)

            # tangent vectors (u)
            tangent1 = torch.cross(normals, ref)
            tangent1 = tangent1 / (torch.linalg.norm(tangent1, dim=-1, keepdim=True) + 1e-6)
            tangent2 = torch.cross(normals, tangent1)
            
            # Project the 2D learnable surface_offsets into 3D world space
            # surface_offsets: [N, 2] -> offset_3d: [N, 3]
            surface_offsets = torch.tanh(self.surface_offsets) * 0.2
            offset_3d = (
                tangent1 * surface_offsets[..., 0:1] + 
                tangent2 * surface_offsets[..., 1:2]
            )
            local_pos = local_pos + offset_3d

        rotated_pos = torch.bmm(current_rot, local_pos.unsqueeze(-1)).squeeze(-1)        
        global_pos = rotated_pos + current_trans
        return global_pos

    def forward(self):
        global_pos = self._compute_means()
        global_pos += torch.tanh(self.offsets) * self.cfg.max_offset
        return torch.cat((global_pos, self.background), 0)