import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from superdec.loss.sampler import EqualDistanceSamplerSQ
from superdec.utils.transforms import transform_to_primitive_frame, quat2mat, mat2quat
from superdec.utils.safe_operations import safe_pow, safe_mul

def sampling_from_parametric_space_to_equivalent_points(
    shape_params,
    epsilons,
    sq_sampler
):
    """
    Given the sampling steps in the parametric space, we want to ge the actual
    3D points.
    """
    def fexp(x, p):
        return torch.sign(x)*(torch.abs(x)**p)
    B = shape_params.shape[0]  # batch size
    M = shape_params.shape[1]  # number of primitives
    S = sq_sampler.n_samples

    etas, omegas = sq_sampler.sample_on_batch(
        shape_params.detach().cpu().numpy(),
        epsilons.detach().cpu().numpy()
    )
    # Make sure we don't get nan for gradients
    etas[etas == 0] += 1e-6
    omegas[omegas == 0] += 1e-6

    # Move to tensors
    etas = shape_params.new_tensor(etas)
    omegas = shape_params.new_tensor(omegas)

    # Make sure that all tensors have the right shape
    a1 = shape_params[:, :, 0].unsqueeze(-1)  # size BxMx1
    a2 = shape_params[:, :, 1].unsqueeze(-1)  # size BxMx1
    a3 = shape_params[:, :, 2].unsqueeze(-1)  # size BxMx1
    e1 = epsilons[:, :, 0].unsqueeze(-1)  # size BxMx1
    e2 = epsilons[:, :, 1].unsqueeze(-1)  # size BxMx1

    x = a1 * fexp(torch.cos(etas), e1) * fexp(torch.cos(omegas), e2)
    y = a2 * fexp(torch.cos(etas), e1) * fexp(torch.sin(omegas), e2)
    z = a3 * fexp(torch.sin(etas), e1)

    x = ((x > 0).float() * 2 - 1) * torch.max(torch.abs(x), x.new_tensor(1e-6))
    y = ((y > 0).float() * 2 - 1) * torch.max(torch.abs(y), x.new_tensor(1e-6))
    z = ((z > 0).float() * 2 - 1) * torch.max(torch.abs(z), x.new_tensor(1e-6))

    # Compute the normals of the SQs
    nx = (torch.cos(etas)**2) * (torch.cos(omegas)**2) / x
    ny = (torch.cos(etas)**2) * (torch.sin(omegas)**2) / y
    nz = (torch.sin(etas)**2) / z

    return torch.stack([x, y, z], -1), torch.stack([nx, ny, nz], -1)

class BaseLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.w_sps = getattr(cfg, 'w_sps', 0.0)
        self.w_ext = getattr(cfg, 'w_ext', 0.0)

    def compute_existence_loss(self, assign_matrix, exist):
        thred = 24
        loss = nn.BCELoss().cuda()
        gt = (assign_matrix.sum(1) > thred).to(torch.float32).detach()
        entropy = loss(exist.squeeze(-1), gt)
        return entropy

    def get_sparsity_loss(self, assign_matrix):
        num_points = assign_matrix.shape[1]
        norm_05 = (assign_matrix.sum(1)/num_points + 0.01).sqrt().mean(1).pow(2)
        norm_05 = torch.mean(norm_05)
        return norm_05

    def compute_common_losses(self, out_dict, loss_dict):
        loss = 0
        loss_dict['expected_prim_num'] = out_dict['exist'].squeeze(-1).sum(-1).mean().data.detach().item()

        if self.w_ext > 0:
            exist_loss = self.compute_existence_loss(out_dict['assign_matrix'], out_dict['exist'])
            loss += self.w_ext * exist_loss
            loss_dict['exist_loss'] = exist_loss.item()

        if self.w_sps > 0:
            sparsity_loss = self.get_sparsity_loss(out_dict['assign_matrix'])
            loss += self.w_sps * sparsity_loss
            loss_dict['sparsity_loss'] = sparsity_loss.item()
            
        return loss

class SuperDecLoss(BaseLoss):
    def __init__(self, cfg):
        super().__init__(cfg)
        self._init_buffers()
        self.sampler = EqualDistanceSamplerSQ(n_samples=cfg.n_samples, D_eta=0.05, D_omega=0.05)
        
        self.w_cub = cfg.w_cub
        self.w_cd = cfg.w_cd

        
        self.cos_sim_cubes = nn.CosineSimilarity(dim=4, eps=1e-4) 

    def _init_buffers(self):
        self.register_buffer('mask_project', torch.FloatTensor([[0,1,1],[0,1,1],[1,0,1],[1,0,1],[1,1,0],[1,1,0]]).unsqueeze(0).unsqueeze(0))
        self.register_buffer('mask_plane', torch.FloatTensor([[1,0,0],[1,0,0],[0,1,0],[0,1,0],[0,0,1],[0,0,1]]).unsqueeze(0).unsqueeze(0))
        self.register_buffer('cube_normal', torch.FloatTensor([[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]]).unsqueeze(0).unsqueeze(0))
        self.register_buffer('cube_planes', torch.FloatTensor([[-1,-1,-1],[1,1,1]]).unsqueeze(0).unsqueeze(0))

    def compute_cuboid_loss(self, pc_inver, normals_inver, out_dict):
        B, P = out_dict['scale'].shape[:2]
        N = pc_inver.shape[2]

        planes_scaled = self.cube_planes.repeat(B, P, 1, 1) * out_dict['scale'].unsqueeze(2).repeat(1,1,2,1)
        planes_scaled = planes_scaled.unsqueeze(1).repeat(1,N,1,3,1).reshape(B,N,P*6,3)
        mask_project = self.mask_project.repeat(B,N,P,1)
        mask_plane = self.mask_plane.repeat(B,N,P,1)
        cube_normal = self.cube_normal.unsqueeze(2).repeat(B,N,P,1,1)
        scale_reshaped = out_dict['scale'].unsqueeze(1).repeat(1,N,1,6).reshape(B,N,P*6,3)
        
        normals_inver_reshaped = normals_inver.permute(0,2,1,3).unsqueeze(3).repeat(1,1,1,6,1)
        _, idx_normals_sim_max = torch.max(self.cos_sim_cubes(normals_inver_reshaped, cube_normal),dim=-1,keepdim=True)

        pc_project = pc_inver.permute(0,2,1,3).repeat(1,1,1,6).reshape(B,N,P*6,3) * mask_project + planes_scaled * mask_plane
        pc_project = torch.max(torch.min(pc_project, scale_reshaped), -scale_reshaped).view(B, N, P, 6, 3)  
        pc_project = torch.gather(pc_project, dim=3, index = idx_normals_sim_max.unsqueeze(-1).repeat(1,1,1,1,3)).squeeze(3).permute(0,2,1,3)

        diff = ((pc_project - pc_inver) ** 2).sum(-1).permute(0,2,1)
        diff = torch.mean(torch.mean(torch.sum(diff * out_dict['assign_matrix'], -1), 1))

        return diff

    def compute_cd_loss(self, pc_inver, out_dict):
        weights = out_dict['assign_matrix']  # [B, P, N]
        scale = out_dict['scale']             # [B, P, 3]
        shape = out_dict['shape']             # [B, P, 2]
        exist = out_dict['exist']             # [B, P, 1]

        # Sample points and normals on superquadrics
        X_SQ, normals = sampling_from_parametric_space_to_equivalent_points(scale, shape, self.sampler)
        normals = normals.detach()        # [B, P, S, 3]
        
        # Compute squared distances
        diff = X_SQ.unsqueeze(3) - pc_inver.unsqueeze(2)  # [B, P, S, N, 3]
        D = (diff ** 2).sum(-1)                            # [B, P, S, N]

        # Point-to-Primitive Chamfer
        pcl_to_prim_loss = D.min(dim=2)[0]      
        pcl_to_prim_loss = (pcl_to_prim_loss.transpose(-1,-2) * weights).sum(-1).mean()

        # Primitive-to-Point Chamfer
        distances_bis = D.min(dim=3)[0]                     # [B, P, S]
        prim_to_pcl = distances_bis.mean(dim=-1)            # [B, P]
        prim_to_pcl_loss = (prim_to_pcl * exist.squeeze(-1)).sum(dim=-1)
        prim_to_pcl_loss = (prim_to_pcl_loss / (exist.squeeze(-1).sum(dim=-1) + 1e-6)).mean()

        return pcl_to_prim_loss, prim_to_pcl_loss

    def forward(self, batch, out_dict):
        pc, normals = batch['points'].cuda().float(), batch['normals'].cuda().float()
        pc_inver = transform_to_primitive_frame(pc, out_dict['trans'], out_dict['rotate'])
        normals_inver = transform_to_primitive_frame(normals, out_dict['trans'], out_dict['rotate'])

        loss = 0
        loss_dict = {}

        if self.w_cub > 0:
            cub_loss = self.compute_cuboid_loss(pc_inver, normals_inver, out_dict)
            loss += self.w_cub * cub_loss
            loss_dict['cub_loss'] = cub_loss.item()

        if self.w_cd > 0:
            pcl_to_prim_loss, prim_to_pcl_loss = self.compute_cd_loss(pc_inver, out_dict)
            cd_loss = pcl_to_prim_loss + prim_to_pcl_loss
            loss += self.w_cd * cd_loss
            loss_dict['cd_loss'] = cd_loss.item()

        loss += self.compute_common_losses(out_dict, loss_dict)
        
        loss_dict['all'] = loss.item()
        return loss, loss_dict

class ParamLoss(BaseLoss):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.w_param = getattr(cfg, 'w_param', 1.0)

    def forward(self, batch, out_dict):
        gt_scale = batch['gt_scale'].cuda().float()
        gt_shape = batch['gt_shape'].cuda().float()
        gt_trans = batch['gt_trans'].cuda().float()
        gt_rotate = batch['gt_rotate'].cuda().float()
        gt_exist = batch['gt_exist'].cuda().float()
        
        loss = 0
        loss_dict = {}            
        mask = (gt_exist > 0.5).float()
        
        # Scale
        scale_loss = nn.MSELoss(reduction='none')(out_dict['scale'], gt_scale) # [B, P, 3]
        scale_loss = (scale_loss * mask).sum() / (mask.sum() * 3 + 1e-6)
        loss += scale_loss
        loss_dict['param_scale'] = scale_loss.item()
        
        # Shape
        shape_loss = nn.MSELoss(reduction='none')(out_dict['shape'], gt_shape) # [B, P, 2]
        shape_loss = (shape_loss * mask).sum() / (mask.sum() * 2 + 1e-6)
        loss += shape_loss
        loss_dict['param_shape'] = shape_loss.item()
        
        # Translation
        trans_loss = nn.MSELoss(reduction='none')(out_dict['trans'], gt_trans) # [B, P, 3]
        trans_loss = (trans_loss * mask).sum() / (mask.sum() * 3 + 1e-6)
        loss += trans_loss
        loss_dict['param_trans'] = trans_loss.item()
        
        # Rotation
        rot_loss = nn.MSELoss(reduction='none')(out_dict['rotate'], gt_rotate) # [B, P, 3, 3]
        rot_loss = (rot_loss * mask.unsqueeze(-1)).sum() / (mask.sum() * 9 + 1e-6)
        loss += rot_loss
        loss_dict['param_rot'] = rot_loss.item()
        
        total_loss = self.w_param * loss
        loss_dict['param_loss'] = loss.item()
        
        total_loss += self.compute_common_losses(out_dict, loss_dict)
        loss_dict['all'] = total_loss.item()
        return total_loss, loss_dict

class IoULoss(BaseLoss):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.truncation = getattr(cfg, 'truncation', 0.05)
        self.w_iou = getattr(cfg, 'w_iou', 1.0)
        self.w_sdf = getattr(cfg, 'w_sdf', 1.0)
        self.w_bbox = getattr(cfg, 'w_bbox', 1.0)
    
    def poles(self, scale, rotate, trans):
        B, N, _ = scale.shape
        local = torch.zeros(B, N, 6, 3, device=scale.device)
        local[:,:,0,0] =  scale[:,:,0]  # +x
        local[:,:,1,0] = -scale[:,:,0]  # -x
        local[:,:,2,1] =  scale[:,:,1]  # +y
        local[:,:,3,1] = -scale[:,:,1]  # -y
        local[:,:,4,2] =  scale[:,:,2]  # +z
        local[:,:,5,2] = -scale[:,:,2]  # -z

        # rotate
        poles_world = torch.matmul(
            rotate.unsqueeze(2),     # (B,N,1,3,3)
            local.unsqueeze(-1) # (B,N,6,3,1)
        ).squeeze(-1)

        # translate
        poles_world = poles_world + trans.unsqueeze(2)
        return poles_world

    def sdf_batch(self, points, out_dict):
        # points: (B, M, 3) -> expects transpose for math -> (B, 3, M) roughly
        points = points.transpose(1, 2) # (B, 3, M)
        
        scale = out_dict['scale'] # (B, N, 3)
        exponents = out_dict['shape'] # (B, N, 2)
        rotate = out_dict['rotate'] # (B, N, 3, 3)
        trans = out_dict['trans'] # (B, N, 3)
        tapering = out_dict.get('tapering', torch.zeros_like(exponents)) # (B, N, 2)

        B, _, M = points.shape
        N = scale.shape[1]
        
        points_expanded = points.unsqueeze(1) # (B, 1, 3, M)
        t = trans.unsqueeze(-1) # (B, N, 3, 1)
        points_centered = points_expanded - t # (B, N, 3, M)
        
        X = torch.matmul(rotate.transpose(-2, -1), points_centered) # (B, N, 3, M)
        
        e1 = exponents[..., 0].unsqueeze(-1)
        e2 = exponents[..., 1].unsqueeze(-1)
        sx = scale[..., 0].unsqueeze(-1)
        sy = scale[..., 1].unsqueeze(-1)
        sz = scale[..., 2].unsqueeze(-1)
        
        x = X[:, :, 0, :]
        y = X[:, :, 1, :]
        z = X[:, :, 2, :]
        
        eps = 1e-6
        x = torch.where(x > 0, 1.0, -1.0) * torch.clamp(torch.abs(x), min=eps)
        y = torch.where(y > 0, 1.0, -1.0) * torch.clamp(torch.abs(y), min=eps)
        z = torch.where(z > 0, 1.0, -1.0) * torch.clamp(torch.abs(z), min=eps)
        
        kx = tapering[..., 0].unsqueeze(-1)
        ky = tapering[..., 1].unsqueeze(-1)
        
        fx = safe_mul(kx/sz, z) + 1
        fy = safe_mul(ky/sz, z) + 1
        
        # Check tapering signs
        fx = torch.where(fx > 0, 1.0, -1.0) * torch.clamp(torch.abs(fx), min=eps)
        fy = torch.where(fy > 0, 1.0, -1.0) * torch.clamp(torch.abs(fy), min=eps)
        
        x = x / fx
        y = y / fy
        
        r0 = torch.sqrt(x**2 + y**2 + z**2)
        
        term1 = safe_pow(safe_pow(x / sx, 2), 1 / e2)
        term2 = safe_pow(safe_pow(y / sy, 2), 1 / e2)
        term3 = safe_pow(safe_pow(z / sz, 2), 1 / e1)
        
        f_func = safe_pow(safe_pow(term1 + term2, e2 / e1) + term3, -e1 / 2)
        
        sdf = safe_mul(r0, (1 - f_func))
        return sdf # (B, N, M)

    def forward(self, batch, out_dict):
        pc, normals = batch['points'].cuda().float(), batch['normals'].cuda().float()
        points_iou, gt_occ = batch['points_iou'].cuda().float(), batch['occupancies'].cuda().bool()

        # pc: (B, M_surf, 3)
        loss = 0
        loss_dict = {}
        
        # 1. SDF on Surface Points
        all_sdfs_surf = self.sdf_batch(pc, out_dict) #(B, N, M_surf)
        
        # Mask out non-existing primitives
        exist_mask = (out_dict['exist'] > 0.5).reshape(all_sdfs_surf.shape[0], all_sdfs_surf.shape[1]) # (B, N)
        mask = exist_mask.unsqueeze(-1).expand_as(all_sdfs_surf)
        all_sdfs_surf[~mask] = float('inf') # So they don't contribute to min
        
        # Union SDF (min)
        tau = 0.01
        min_sdf_surf = -tau * torch.logsumexp(-all_sdfs_surf / tau, dim=1) # (B, M_surf)
        
        # SDF Loss (surface points should have SDF=0)
        # Using truncated SDF logic from batch_superq
        temperature = 1e-2
        sdfs_surf_trunc = (torch.sigmoid(min_sdf_surf / temperature) - 0.5) * 2 * self.truncation
        # Since we want SDF=0, we minimize abs(sdf)
        L_sdf = 8 * torch.mean(torch.abs(sdfs_surf_trunc))
        loss += self.w_sdf * L_sdf
        loss_dict['sdf'] = L_sdf.item()
        
        # 2. BBox Loss
        # Compute bbox from PC
        bbox_min = torch.min(pc, dim=1).values.unsqueeze(1).unsqueeze(2) # (B, 1, 1, 3)
        bbox_max = torch.max(pc, dim=1).values.unsqueeze(1).unsqueeze(2)
        # Add margin as in batch_superq
        margin = (bbox_max - bbox_min) * .025
        bbox_min -= margin
        bbox_max += margin
        
        poles = self.poles(out_dict['scale'], out_dict['rotate'], out_dict['trans']) # (B, N, 6, 3)
        mask_bbox = exist_mask.unsqueeze(-1) # (B, N, 1)
        
        violation = torch.relu(bbox_min - poles) + torch.relu(poles - bbox_max) # (B, N, 6, 3)
        dist = torch.linalg.norm(violation, dim=-1) # (B, N, 6)
        L_bbox = (dist * mask_bbox).sum(dim=2).mean() # Mean over batch
        
        loss += self.w_bbox * L_bbox
        loss_dict['bbox'] = L_bbox.item()

        # 3. IoU Loss
        all_sdfs_iou = self.sdf_batch(points_iou, out_dict) # (B, N, M_iou)
        
        # Mask
        mask_iou = exist_mask.unsqueeze(-1).expand_as(all_sdfs_iou)
        all_sdfs_iou[~mask_iou] = float('inf')
        
        # Min SDF
        min_sdf_iou = -tau * torch.logsumexp(-all_sdfs_iou / tau, dim=1) # (B, M_iou)
        
        temperature_iou = 1e-3
        pred_occ = torch.sigmoid(-min_sdf_iou / temperature_iou)
        
        intersection = (pred_occ * gt_occ).sum(dim=1)
        union = (pred_occ + gt_occ - pred_occ * gt_occ).sum(dim=1)
        iou = intersection / torch.clamp(union, min=1.0)
        
        L_iou = -torch.log(iou).mean()
        loss += self.w_iou * L_iou
        loss_dict['iou'] = iou.mean().item()
        loss_dict['iou_loss'] = L_iou.item()

        loss += self.compute_common_losses(out_dict, loss_dict)
        loss_dict['all'] = loss.item()
        return loss, loss_dict


class Loss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        loss_type = getattr(cfg, 'type', 'original')
        print(f"Initializing Loss: {loss_type}")
        
        if loss_type == 'original':
            self.impl = SuperDecLoss(cfg)
        elif loss_type == 'param':
            self.impl = ParamLoss(cfg)
        elif loss_type == 'iou':
            self.impl = IoULoss(cfg)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(self, batch, out_dict):
        return self.impl(batch, out_dict)

