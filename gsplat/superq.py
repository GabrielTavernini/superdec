import torch
import torch.nn as nn
import numpy as np

from superdec.loss.sampler import EqualDistanceSamplerSQ
from superdec.utils.predictions_handler import PredictionHandler

class SuperQ(nn.Module):
    def __init__(
        self, 
        pred_handler: PredictionHandler,
        num_pts_per_sq: int = 400,
        negative_offset: bool = True,
        max_offset: float = 0.05,
        device: str = "cuda",
    ):
        # Anything self.x = nn.Parameter(...) is trainable
        trainable = ["offsets", "sqscale", "exponents", "translation", "rotation"]
        # trainable = ["offsets"]

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

        S = self.sqscale.shape[0]
        Ng = S * num_pts_per_sq
        sampler = EqualDistanceSamplerSQ(n_samples=num_pts_per_sq, D_eta=0.05, D_omega=0.05)
        etas, omegas = sampler.sample_on_batch(
            pred_handler.scale.astype(np.float32),
            pred_handler.exponents.astype(np.float32)
        )
        etas = etas.reshape(-1, num_pts_per_sq)[self.mask].reshape(-1)
        omegas = omegas.reshape(-1, num_pts_per_sq)[self.mask].reshape(-1)

        # Make sure we don't get nan for gradients
        etas[etas == 0] += 1e-6
        omegas[omegas == 0] += 1e-6

        # etas and betas need to be in the state_dict to allow loading from checkpoint
        self.register_buffer("etas", torch.tensor(etas, device=device))
        self.register_buffer("omegas", torch.tensor(omegas, device=device))
        # sq_idx tells us which Superquadric each point belongs to.
        self.register_buffer("sq_idx", torch.arange(S, device=device).repeat_interleave(num_pts_per_sq))
        self.offsets = torch.rand(Ng, 3, device=device)

        # Turn selected attributes into trainable parameters
        for name in trainable:
            val = getattr(self, name)
            setattr(self, name, nn.Parameter(val))

    def load_dynamic_checkpoint(self, state_dict):
        if "offsets" not in state_dict:
            raise RuntimeError("Checkpoint missing 'offsets' key.")
            
        saved_Ng = state_dict["offsets"].shape[0]
        current_Ng = self.offsets.shape[0]

        print(f"Resizing model from {current_Ng} points to {saved_Ng} points to match checkpoint.")

        self.offsets = nn.Parameter(torch.zeros(saved_Ng, 3, device=self.offsets.device))
        self.sq_idx = torch.zeros(saved_Ng, dtype=torch.long, device=self.sq_idx.device)
        self.etas = torch.zeros(saved_Ng, device=self.etas.device)
        self.omegas = torch.zeros(saved_Ng, device=self.omegas.device)
        
        self.load_state_dict(state_dict, strict=True)

    def prune_gs(self, is_prune):
        # offsets are pruned in the strat
        self.etas = self.etas[~is_prune]
        self.omegas = self.omegas[~is_prune]
        self.sq_idx = self.sq_idx[~is_prune]

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

    def forward(self):
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

        if self.negative_offset:
            offset_val = torch.tanh(self.offsets) * self.max_offset
        else:
            offset_val = torch.sigmoid(self.offsets) * self.max_offset        
        global_pos += offset_val
        return global_pos
