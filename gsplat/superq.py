import torch
import torch.nn as nn
import numpy as np
import open3d as o3d
import math
from typing import Optional

from gsplat.strategy.ops import remove

from superdec.loss.sampler import EqualDistanceSamplerSQ
from superdec.utils.predictions_handler import PredictionHandler

class SuperQ(nn.Module):
    def __init__(
        self, 
        pred_handler: PredictionHandler,
        target_pts_density: int = 2000,
        num_pts_background: int = -1,
        negative_offset: bool = True,
        max_offset: float = 0.05,
        background_ply: Optional[str] = None,
        device: str = "cuda",
    ):
        # Anything self.x = nn.Parameter(...) is trainable
        trainable = ["background", "offsets", "sqscale", "exponents", "translation", "rotation"]
        # trainable = ["background", "offsets"]

        super().__init__()
        self.mask = (pred_handler.exist > 0.5).reshape(-1)
        self.sqscale = torch.tensor(pred_handler.scale.reshape(-1, 3)[self.mask], dtype=torch.float, device=device)
        self.exponents = torch.tensor(pred_handler.exponents.reshape(-1, 2)[self.mask], dtype=torch.float, device=device)
        self.translation = torch.tensor(pred_handler.translation.reshape(-1, 3)[self.mask], dtype=torch.float, device=device)
        self.rotation = torch.tensor(pred_handler.rotation.reshape(-1, 3, 3)[self.mask], dtype=torch.float, device=device)
        self.pred_handler = pred_handler

        print(f"Loaded {self.sqscale.shape[0]} superquadircs.")

        self.negative_offset = negative_offset
        self.max_offset = max_offset

        # Calculate specific points per SQ
        areas = self._estimate_surface_areas().detach().cpu().numpy()  # [S]
        num_pts_per_sq_area_based = np.round(areas * target_pts_density).astype(int)
        num_pts_per_sq_area_based[num_pts_per_sq_area_based < 1000] = 1000
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

        if background_ply is not None:
            pcd = o3d.io.read_point_cloud(background_ply)
            points_np = np.asarray(pcd.points)
            if num_pts_background != -1:
                idx = np.random.choice(points_np.shape[0], num_pts_background, replace=False)
                points_np = points_np[idx]
            self.background = torch.tensor(points_np, device=device).float()
        else:
            self.background = self._create_background(num_pts_background)

        # Turn selected attributes into trainable parameters
        for name in trainable:
            val = getattr(self, name)
            setattr(self, name, nn.Parameter(val))

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
    
    def _prune_parameter(self, is_prune, optimizers, param_name):
        tmp_dict = {param_name: getattr(self, param_name)}
        remove(
            params=tmp_dict, 
            optimizers={param_name: optimizers[param_name]}, 
            state={}, 
            mask=is_prune
        )
        setattr(self, param_name, tmp_dict[param_name])

    def prune_gs(self, is_prune, optimizers):
        Ng = self.offsets.shape[0]
        self._prune_parameter(is_prune[:Ng], optimizers, "offsets")
        self._prune_parameter(is_prune[Ng:], optimizers, "background")

        self.etas = self.etas[~is_prune[:Ng]]
        self.omegas = self.omegas[~is_prune[:Ng]]
        self.sq_idx = self.sq_idx[~is_prune[:Ng]]

    def update_handler(self):
        batch_size = self.pred_handler.scale.shape[1]
        mask = self.mask.reshape(-1, batch_size)
        self.pred_handler.scale[mask] = self.sqscale.detach().cpu().numpy()
        self.pred_handler.exponents[mask] = self.exponents.detach().cpu().numpy()
        self.pred_handler.translation[mask] = self.translation.detach().cpu().numpy()
        self.pred_handler.rotation[mask] = self.rotation.detach().cpu().numpy()
        meshes = self.pred_handler.get_meshes(resolution=30)
        return self.pred_handler

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

        rotated_pos = torch.bmm(current_rot, local_pos.unsqueeze(-1)).squeeze(-1)        
        global_pos = rotated_pos + current_trans
        return global_pos

    def forward(self):
        global_pos = self._compute_means()

        if self.negative_offset:
            offset_val = torch.tanh(self.offsets) * self.max_offset
        else:
            offset_val = torch.sigmoid(self.offsets) * self.max_offset  
        global_pos += offset_val
        return torch.cat((global_pos, self.background), 0)

    def _estimate_surface_areas(self, n_samples: int = 2048) -> torch.Tensor:
        """
        Numerically estimate surface area for each SQ by sampling (eta, omega)
        and computing mean |dX/deta x dX/domega| * domain_area.

        Domain: eta in [-pi/2, pi/2] (length pi), omega in [-pi, pi] (length 2pi).
        Domain area = pi * 2pi = 2 * pi^2
        """
        device = self.sqscale.device
        S = self.sqscale.shape[0]

        # random param samples (uniform over domain)
        eta = (torch.rand(n_samples, device=device) - 0.5) * math.pi      # in [-pi/2, pi/2]
        omega = (torch.rand(n_samples, device=device) - 0.5) * 2 * math.pi  # in [-pi, pi]

        # Expand to (S, n_samples)
        eta = eta.unsqueeze(0).expand(S, n_samples)     # [S, n_samples]
        omega = omega.unsqueeze(0).expand(S, n_samples) # [S, n_samples]

        a1 = self.sqscale[:, 0].unsqueeze(1)  # [S,1]
        a2 = self.sqscale[:, 1].unsqueeze(1)
        a3 = self.sqscale[:, 2].unsqueeze(1)
        e1 = self.exponents[:, 0].unsqueeze(1)
        e2 = self.exponents[:, 1].unsqueeze(1)

        cos_eta = torch.cos(eta)
        sin_eta = torch.sin(eta)
        cos_omega = torch.cos(omega)
        sin_omega = torch.sin(omega)

        # base param functions
        t1 = self.fexp(cos_eta, e1)   # [S, n]
        t2 = self.fexp(sin_eta, e1)
        fcos = self.fexp(cos_omega, e2)
        fsin = self.fexp(sin_omega, e2)

        # derivatives of fexp:
        # df/dx for f(x)=sign(x)*|x|^p is p * |x|^(p-1)
        # then dx/deta for cos_eta is -sin_eta, for sin_eta is cos_eta, for cos_omega -> -sin_omega, for sin_omega -> cos_omega
        # note shapes broadcast correctly
        df_dcos_eta = e1 * (torch.abs(cos_eta) ** (e1 - 1))
        dt1_deta = df_dcos_eta * (-sin_eta)

        df_dsin_eta = e1 * (torch.abs(sin_eta) ** (e1 - 1))
        dt2_deta = df_dsin_eta * cos_eta

        df_dcos_omega = e2 * (torch.abs(cos_omega) ** (e2 - 1))
        dfcos_domega = df_dcos_omega * (-sin_omega)

        df_dsin_omega = e2 * (torch.abs(sin_omega) ** (e2 - 1))
        dfsin_domega = df_dsin_omega * cos_omega

        # partial derivatives of param surface
        # x = a1 * t1 * fcos
        # y = a2 * t1 * fsin
        # z = a3 * t2

        rx_eta = a1 * dt1_deta * fcos
        rx_omega = a1 * t1 * dfcos_domega

        ry_eta = a2 * dt1_deta * fsin
        ry_omega = a2 * t1 * dfsin_domega

        rz_eta = a3 * dt2_deta
        rz_omega = torch.zeros_like(rz_eta)

        # dX/deta = (rx_eta, ry_eta, rz_eta)
        # dX/domega = (rx_omega, ry_omega, 0)
        # cross product:
        cx = ry_eta * rz_omega - rz_eta * ry_omega
        cy = rz_eta * rx_omega - rx_eta * rz_omega
        cz = rx_eta * ry_omega - ry_eta * rx_omega

        # magnitude
        mag = torch.sqrt(cx * cx + cy * cy + cz * cz + 1e-12)  # [S, n_samples]

        mean_mag = torch.mean(mag, dim=1)  # [S]
        domain_area = 2.0 * (math.pi ** 2)  # pi * 2pi
        areas = mean_mag * domain_area
        return areas