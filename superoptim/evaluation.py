import torch.nn as nn
import trimesh
import numpy as np
from scipy.spatial import KDTree
from superdec.data.dataloader import ShapeNet, ScenesDataset
from torch.utils.data import DataLoader

def build_dataloader(cfg):
    if cfg.dataloader.dataset == 'shapenet':
        ds = ShapeNet(split=cfg.shapenet.split, cfg=cfg)
    elif cfg.dataloader.dataset == 'scenes_dataset':
        ds = ScenesDataset(cfg=cfg)
    else:
        raise ValueError(f"Unsupported dataset {cfg.dataloader.dataset}")

    dl = DataLoader(
        ds, batch_size=cfg.dataloader.batch_size, shuffle=False,
        num_workers=cfg.dataloader.num_workers, pin_memory=True
    )
    return dl

def check_mesh_contains(mesh, points):
    return mesh.contains(points)

def compute_iou(occ1, occ2):
    ''' Computes the Intersection over Union (IoU) value for two sets of
    occupancy values.

    Args:
        occ1 (tensor): first set of occupancy values
        occ2 (tensor): second set of occupancy values
    '''
    # occ1 = np.asarray(occ1)
    # occ2 = np.asarray(occ2)

    # Put all data in second dimension
    # Also works for 1-dimensional data
    if occ1.ndim >= 2:
        occ1 = occ1.reshape(occ1.shape[0], -1)
    if occ2.ndim >= 2:
        occ2 = occ2.reshape(occ2.shape[0], -1)

    # Convert to boolean values
    occ1 = (occ1 >= 0.5)
    occ2 = (occ2 >= 0.5)

    # Compute IOU
    area_union = (occ1 | occ2).astype(np.float32).sum(axis=-1)
    # Avoid division by zero
    area_union = np.maximum(area_union, 1e-6)
    
    area_intersect = (occ1 & occ2).astype(np.float32).sum(axis=-1)

    iou = (area_intersect / area_union)

    return iou

def eval_mesh(mesh, pointcloud_tgt, normals_tgt,
              points_iou=None, occ_tgt=None):
    ''' Evaluates a mesh.

    Args:
        mesh (trimesh): mesh which should be evaluated
        pointcloud_tgt (numpy array): target point cloud
        normals_tgt (numpy array): target normals
        points_iou (numpy_array): points tensor for IoU evaluation
        occ_tgt (numpy_array): GT occupancy values for IoU points
    '''
    if len(mesh.vertices) != 0 and len(mesh.faces) != 0:
        pointcloud, idx = mesh.sample(pointcloud_tgt.shape[0], return_index=True)
        pointcloud = pointcloud.astype(np.float32)
        normals = mesh.face_normals[idx]
    else:
        pointcloud = np.empty((0, 3))
        normals = np.empty((0, 3))

    out_dict = get_outdict(pointcloud_tgt, normals_tgt, pointcloud, normals)

    if points_iou is not None and occ_tgt is not None and len(mesh.vertices) != 0 and len(mesh.faces) != 0:
        occ = check_mesh_contains(mesh, points_iou)
        out_dict['iou'] = compute_iou(occ, occ_tgt)
    else:
        out_dict['iou'] = 0.

    return out_dict

def get_threshold_percentage(dist, thresholds):
    ''' Evaluates a point cloud.

    Args:
        dist (numpy array): calculated distance
        thresholds (numpy array): threshold values for the F-score calculation
    '''
    in_threshold = [
        (dist <= t).mean() for t in thresholds
    ]
    return in_threshold

def distance_p2p(points_src, normals_src, points_tgt, normals_tgt): # from convolutional occupancy network
    ''' Computes minimal distances of each point in points_src to points_tgt.

    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    '''
    kdtree = KDTree(points_tgt)
    dist, idx = kdtree.query(points_src)

    if normals_src is not None and normals_tgt is not None:
        normals_src = \
            normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = \
            normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
        # Handle normals that point into wrong direction gracefully
        # (mostly due to mehtod not caring about this in generation)
        normals_dot_product = np.abs(normals_dot_product)
    else:
        normals_dot_product = np.array(
            [np.nan] * points_src.shape[0], dtype=np.float32)
    return dist, normals_dot_product

def get_outdict(pc_gt, normals_gt, pc_pred, normals_pred):
    thresholds = np.linspace(1./1000, 1, 1000)
    completeness, completeness_normals = distance_p2p(pc_gt, normals_gt, pc_pred, normals_pred)
    recall = get_threshold_percentage(completeness, thresholds)
    completeness2 = completeness**2
    
    completeness = completeness.mean()
    completeness2 = completeness2.mean()
    completeness_normals = completeness_normals.mean()
    
    # Accuracy: how far are th points of the predicted pointcloud
    # from the target pointcloud
    accuracy, accuracy_normals = distance_p2p(pc_pred, normals_pred, pc_gt, normals_gt)
    precision = get_threshold_percentage(accuracy, thresholds)
    accuracy2 = accuracy**2
    
    accuracy = accuracy.mean()
    accuracy2 = accuracy2.mean()
    accuracy_normals = accuracy_normals.mean()
    
    # Chamfer distance
    chamferL2 = 0.5 * (completeness2 + accuracy2)
    normals_correctness = (
        0.5 * completeness_normals + 0.5 * accuracy_normals
    )
    chamferL1 = 0.5 * (completeness + accuracy)
    F = [
        2 * precision[i] * recall[i] / (precision[i] + precision[i] + 0.000001)
        for i in range(len(precision))
    ]

    
    out_dict_cur = {
            'completeness': completeness,
            'accuracy': accuracy,
            'normals completeness': completeness_normals,
            'normals accuracy': accuracy_normals,
            'normals': normals_correctness,
            'completeness2': completeness2,
            'accuracy2': accuracy2,
            'chamfer-L2': chamferL2,
            'chamfer-L1': chamferL1,
            'f-score': F[9], # threshold = 1.0%
            'f-score-15': F[14], # threshold = 1.5%
            'f-score-20': F[19], # threshold = 2.0%
        }
    return out_dict_cur