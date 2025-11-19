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

        self.negative_offset = negative_offset
        self.max_offset = max_offset

        S = self.sqscale.shape[0]
        Ng = S * num_pts_per_sq
        sampler = EqualDistanceSamplerSQ(n_samples=num_pts_per_sq, D_eta=0.05, D_omega=0.05)
        etas, omegas = sampler.sample_on_batch(
            pred_handler.scale.astype(np.float32),
            pred_handler.exponents.astype(np.float32)
        )
        etas = etas.reshape(-1, num_pts_per_sq)[self.mask]
        omegas = omegas.reshape(-1, num_pts_per_sq)[self.mask]

        # Make sure we don't get nan for gradients
        etas[etas == 0] += 1e-6
        omegas[omegas == 0] += 1e-6


        self.etas = torch.tensor(etas, device=device)
        self.omegas = torch.tensor(omegas, device=device)
        self.offsets = torch.rand(Ng, 3, device=device)
        self.is_prune = torch.zeros(etas.shape, dtype=torch.bool, device=device)

        # Turn selected attributes into trainable parameters
        for name in trainable:
            val = getattr(self, name)
            setattr(self, name, nn.Parameter(val))

    def prune_gs(self, is_prune):
        tmp = self.is_prune.reshape(-1)
        sel = torch.where(~tmp)
        tmp[sel] |= is_prune
        self.is_prune |= tmp.reshape(self.etas.shape)

    def update_handler(self):
        # TODO: fix to work with multiple objects
        self.pred_handler.scale[0][self.mask] = self.sqscale.detach().cpu().numpy()
        self.pred_handler.exponents[0][self.mask] = self.exponents.detach().cpu().numpy()
        self.pred_handler.translation[0][self.mask] = self.translation.detach().cpu().numpy()
        self.pred_handler.rotation[0][self.mask] = self.rotation.detach().cpu().numpy()
        meshes = self.pred_handler.get_meshes(resolution=30)
        return self.pred_handler

    def fexp(self, x, p):
        return torch.sign(x)*(torch.abs(x)**p)

    def forward(self):
        # Make sure that all tensors have the right shape
        a1 = self.sqscale[:, 0].unsqueeze(-1) # Sx1
        a2 = self.sqscale[:, 1].unsqueeze(-1) # Sx1
        a3 = self.sqscale[:, 2].unsqueeze(-1) # Sx1
        e1 = self.exponents[:, 0].unsqueeze(-1) # Sx1
        e2 = self.exponents[:, 1].unsqueeze(-1) # Sx1

        x = a1 * self.fexp(torch.cos(self.etas), e1) * self.fexp(torch.cos(self.omegas), e2)
        y = a2 * self.fexp(torch.cos(self.etas), e1) * self.fexp(torch.sin(self.omegas), e2)
        z = a3 * self.fexp(torch.sin(self.etas), e1)
        
        x = ((x > 0).float() * 2 - 1) * torch.max(torch.abs(x), x.new_tensor(1e-6))
        y = ((y > 0).float() * 2 - 1) * torch.max(torch.abs(y), x.new_tensor(1e-6))
        z = ((z > 0).float() * 2 - 1) * torch.max(torch.abs(z), x.new_tensor(1e-6))
        pos = torch.stack([x, y, z], -1)

        global_pos = []
        for s in range(self.sqscale.shape[0]):
            p = pos[s] @ self.rotation[s].T
            p += self.translation[s]
            global_pos.append(p)
        global_pos = torch.stack(global_pos, dim=0)
        global_pos = global_pos.reshape(-1, 3)

        if self.negative_offset:
            global_pos += torch.tanh(self.offsets) * self.max_offset
        else:
            global_pos += torch.sigmoid(self.offsets) * self.max_offset
        
        global_pos = global_pos[~self.is_prune.reshape(-1)]
        return global_pos
