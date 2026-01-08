import torch
from torch import Tensor
from typing import Dict, Union, Callable, List, Tuple
import torch.nn.functional as F

from superq import SuperQ
from gsplat.strategy.ops import _update_param_with_optimizer, remove, normalized_quat_to_rotmat

@torch.no_grad()
def duplicate(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    superq: SuperQ,
    superq_optimizers: Dict[str, torch.optim.Optimizer],
    mask: Tensor,
    state: Dict[str, Tensor]
):
    Ng, _ = superq.get_counts()
    superq_params = dict(superq.named_parameters())
    device = mask.device
    
    mask_sq, mask_bg = mask[:Ng], mask[Ng:]
    n_dup_sq = mask_sq.sum().item()
    n_dup_bg = mask_bg.sum().item()

    def _duplicate_with_mask(mask, params, optimizers, names=None):
        def p_fn(n, p): return torch.nn.Parameter(torch.cat([p, p[mask]]), requires_grad=True)
        def o_fn(k, v): return torch.cat([v, torch.zeros((mask.sum().item(), *v.shape[1:]), device=device)])
        _update_param_with_optimizer(p_fn, o_fn, params, optimizers, names=names)

    if n_dup_sq > 0:
        superq.densify_buffers(mask_sq)
        _duplicate_with_mask(mask_sq, superq_params, superq_optimizers, names=superq.trainable_params_sq)

    if n_dup_bg > 0:
        _duplicate_with_mask(mask_bg, superq_params, superq_optimizers, names=superq.trainable_params_bg)
    
    superq.update(superq_params)

    def param_fn_general(name: str, p: Tensor) -> Tensor:
        p_sq, p_bg = p[:Ng], p[Ng:]
        p_sq_new = torch.cat([p_sq, p_sq[mask_sq]])
        p_bg_new = torch.cat([p_bg, p_bg[mask_bg]])
        return torch.nn.Parameter(torch.cat([p_sq_new, p_bg_new]), requires_grad=p.requires_grad)

    def optimizer_fn_general(key: str, v: Tensor) -> Tensor:
        v_sq_new = torch.cat([v[:Ng], torch.zeros((n_dup_sq, *v.shape[1:]), device=device)])
        v_bg_new = torch.cat([v[Ng:], torch.zeros((n_dup_bg, *v.shape[1:]), device=device)])
        return torch.cat([v_sq_new, v_bg_new])

    _update_param_with_optimizer(param_fn_general, optimizer_fn_general, params, optimizers)

    # update the extra running state
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            v_sq, v_bg = v[:Ng], v[Ng:]
            v_sq_new = torch.cat([v_sq, v_sq[mask_sq]])
            v_bg_new = torch.cat([v_bg, v_bg[mask_bg]])
            state[k] = torch.cat([v_sq_new, v_bg_new])


@torch.no_grad()
def split(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Tensor],
    mask: Tensor,
    superq_optimizers: Dict[str, torch.optim.Optimizer],
    superq,
    revised_opacity: bool = False,
):
    Ng, _ = superq.get_counts()
    superq_params = dict(superq.named_parameters())
    device = mask.device
    
    mask_sq, mask_bg = mask[:Ng], mask[Ng:]
    n_split_sq = mask_sq.sum().item()
    n_split_bg = mask_bg.sum().item()

    def get_samples(param_scales, param_quats, mask):
        scales = torch.exp(param_scales[mask])
        quats = F.normalize(param_quats[mask], dim=-1)
        rotmats = normalized_quat_to_rotmat(quats)  # [N, 3, 3]
        samples = torch.einsum(
            "nij,nj,bnj->bni",
            rotmats,
            scales,
            torch.randn(2, len(scales), 3, device=device),
        )  # [2, N, 3]
        return samples

    def _split_with_mask(mask, samples, params, optimizers, names=None):
        def p_fn(n, p):
            if len(p.shape) > 1 and p.shape[1] == 3:
                p_split = (p[mask] + samples).reshape(-1, 3) # [2*N, 3]
            else:
                p_split = p[mask].repeat(2, *[1] * (p.dim() - 1))
            return torch.nn.Parameter(torch.cat([p[~mask], p_split]), requires_grad=True)
        def o_fn(k, v):
            return torch.cat([v[~mask], torch.zeros((2 * mask.sum().item(), *v.shape[1:]), device=device)])
        _update_param_with_optimizer(p_fn, o_fn, params, optimizers, names=names)

    if n_split_sq > 0:
        superq.densify_buffers(mask_sq, split=True)
        sq_samples = get_samples(params['scales'][:Ng], params['quats'][:Ng], mask_sq)
        _split_with_mask(mask_sq, sq_samples, superq_params, superq_optimizers, names=superq.trainable_params_sq)

    if n_split_bg > 0:
        bg_samples = get_samples(params['scales'][Ng:], params['quats'][Ng:], mask_bg)
        _split_with_mask(mask_bg, bg_samples, superq_params, superq_optimizers, names=superq.trainable_params_bg)

    superq.update(superq_params)

    # general parameter 
    def param_fn_gen(name: str, p: Tensor) -> Tensor:
        p_sq, p_bg = p[:Ng], p[Ng:]
        repeats = [2] + [1] * (p.dim() - 1)

        # sq part
        if name == "scales":
            p_sq_split = torch.log(torch.exp(p_sq[mask_sq]) / 1.6).repeat(2, 1)
        elif name == "opacities" and revised_opacity:
            new_op = 1.0 - torch.sqrt(1.0 - torch.sigmoid(p_sq[mask_sq]))
            p_sq_split = torch.logit(new_op).repeat(repeats)
        else:
            p_sq_split = p_sq[mask_sq].repeat(repeats)
        p_sq_new = torch.cat([p_sq[~mask_sq], p_sq_split])

        # bg part
        if name == "scales":
            p_bg_split = torch.log(torch.exp(p_bg[mask_bg]) / 1.6).repeat(2, 1)
        elif name == "opacities" and revised_opacity:
            new_op = 1.0 - torch.sqrt(1.0 - torch.sigmoid(p_bg[mask_bg]))
            p_bg_split = torch.logit(new_op).repeat(repeats)
        else:
            p_bg_split = p_bg[mask_bg].repeat(repeats)
        p_bg_new = torch.cat([p_bg[~mask_bg], p_bg_split])
        return torch.nn.Parameter(torch.cat([p_sq_new, p_bg_new]), requires_grad=p.requires_grad)

    def optimizer_fn_gen(key: str, v: Tensor) -> Tensor:
        v_sq_split = torch.zeros((2 * n_split_sq, *v.shape[1:]), device=device)
        v_bg_split = torch.zeros((2 * n_split_bg, *v.shape[1:]), device=device)
        return torch.cat([v[:Ng][~mask_sq], v_sq_split, v[Ng:][~mask_bg], v_bg_split])

    _update_param_with_optimizer(param_fn_gen, optimizer_fn_gen, params, optimizers)

    for k, v in state.items():
      if isinstance(v, torch.Tensor):
            v_sq, v_bg = v[:Ng], v[Ng:]
            repeats = [2] + [1] * (v.dim() - 1)
            v_sq_new = torch.cat([v_sq[~mask_sq], v_sq[mask_sq].repeat(repeats)])
            v_bg_new = torch.cat([v_bg[~mask_bg], v_bg[mask_bg].repeat(repeats)])
            state[k] = torch.cat([v_sq_new, v_bg_new])


@torch.no_grad()
def remove(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    superq: SuperQ,
    superq_optimizers: Dict[str, torch.optim.Optimizer],
    mask: Tensor,
    state: Dict[str, Tensor]
):
    Ng, _ = superq.get_counts()
    superq_params = dict(superq.named_parameters())

    def _prune_with_mask(mask, params, optimizers, names=None):
        def param_fn(name: str, p: Tensor) -> Tensor:
            return torch.nn.Parameter(p[~mask], requires_grad=p.requires_grad)
        def optimizer_fn(key: str, v: Tensor) -> Tensor:
            return v[~mask]
        _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers, names)

    # prune general parameters
    _prune_with_mask(mask, params, optimizers)

    # prune superq parameters
    superq.prune_buffers(mask[:Ng])
    _prune_with_mask(mask[:Ng], superq_params, superq_optimizers, names=superq.trainable_params_sq)
    _prune_with_mask(mask[Ng:], superq_params, superq_optimizers, names=superq.trainable_params_bg)
    superq.update(superq_params)
    
    # update the extra running state
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v[~mask]