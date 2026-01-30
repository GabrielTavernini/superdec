import os
import sys
import torch
import numpy as np
from omegaconf import OmegaConf
from superdec.superdec import SuperDec
from superdec.utils.predictions_handler_extended import PredictionHandler
from superdec.data.dataloader import denormalize_outdict, denormalize_points
import open3d as o3d
import viser
import random
from superdec.data.dataloader import normalize_points, denormalize_outdict
from superdec.data.transform import rotate_around_axis
import time
from superdec.utils.visualizations import generate_ncolors
from tqdm import tqdm
import matplotlib.pyplot as plt

from .batch_superq import BatchSuperQMulti
from ..utils import plot_pred_handler

def visualize_handler(server, superq, sdf_values, plot = False):
    # Expect batched tensors from BatchSuperQMulti; use batch 0
    sdf_values = sdf_values.detach().cpu()

    pred_handler, meshes = superq.update_handler()
    if plot:
        plot_pred_handler(pred_handler, superq.truncation)

    mesh = meshes[superq.indices[0]]
    server.scene.add_mesh_trimesh("superquadrics", mesh=mesh, visible=True)

    points = superq.points[0].detach().cpu().numpy()
    cmap = plt.get_cmap('RdBu')
    norm = plt.Normalize(vmin=-superq.truncation, vmax=superq.truncation)
    sdf_arr = sdf_values[0].numpy()
    sdf_colors = cmap(norm(sdf_arr))[:, :3]
    server.scene.add_point_cloud(
        name="/sdf_pointcloud",
        points=points,
        colors=sdf_colors,
        point_size=0.005,
    )

def main():
    if len(sys.argv) > 1:
        object_name = sys.argv[1]
    else:
        object_name = "round_table"
    pred_handler = PredictionHandler.from_npz(f"data/output_npz/objects/{object_name}.npz")
    print(f"Optimizing {pred_handler.names[0]}")
    
    superq = BatchSuperQMulti(
        pred_handler=pred_handler,
        indices=[0],
        ply_paths=[f"data/ShapeNet/04379243/{pred_handler.names[0]}/pointcloud.npz"],
    )
    param_groups = superq.get_param_groups()
    optimizer = torch.optim.Adam(param_groups)

    pred_handler, meshes = superq.update_handler()
    orig_mesh = meshes[superq.indices[0]]
    plot_pred_handler(pred_handler, superq.truncation, filename="superq_plot_orig.png")

    server = viser.ViserServer()
    server.scene.set_up_direction([0.0, 1.0, 0.0])
    server.scene.add_mesh_trimesh("original_superquadrics", mesh=orig_mesh, visible=False)

    # Segmented pointcloud for batch 0
    points = pred_handler.pc[superq.indices[0]]
    assign_matrix = pred_handler.assign_matrix[superq.indices[0]]
    colors = generate_ncolors(assign_matrix.shape[1])
    segmentation = np.argmax(assign_matrix, axis=1)
    colored_pc = colors[segmentation]
    server.scene.add_point_cloud(
        name="/segmented_pointcloud",
        points=points,
        colors=colored_pc,
        point_size=0.005,
        visible=False,
    )

    points = superq.points[0].detach().cpu().numpy()
    color = np.array([1.0, 0.0, 0.0])  # red
    colored_pc = np.tile(color, (points.shape[0], 1))
    server.scene.add_point_cloud(
        name="/iou_points",
        points=points,
        colors=colored_pc,
        point_size=0.005,
        visible=False,
    )

    # torch.autograd.set_detect_anomaly(True)
    num_epochs = 5_000
    pbar = tqdm(range(num_epochs), desc="Fitting Superquadrics")
    best_loss = float('inf')
    best_params = None
    for epoch in pbar:
        optimizer.zero_grad()
        forward_out = superq.forward()
        sdf_vals = forward_out.get('sdfs')

        # Use shared loss computation (per-batch)
        loss, losses = superq.compute_losses(forward_out)
        loss = loss[0]

        if torch.isnan(loss):
            print("Failed optimization with nan values")
            exit()
        
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(superq.parameters(), max_norm=1.0)
        optimizer.step()

        # Save best parameters (based on Total Loss)
        with torch.no_grad():
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_params = {
                    "raw_scale": superq.raw_scale.clone(),
                    "raw_exponents": superq.raw_exponents.clone(),
                    "raw_rotation": superq.raw_rotation.clone(),
                    "raw_tapering": superq.raw_tapering.clone(),
                    "translation": superq.translation.clone(),
                    "iou": losses['iou'][0].item(),
                    "epoch": epoch,
                }

        if epoch % 50 == 0:
            visualize_handler(server, superq, sdf_vals)
        
        pbar.set_postfix({
            "IoU": f"{losses['iou'][0].item():.6f}",
            # "Reg": f"{losses['reg'][0].item():.6f}",
            "Tap": f"{losses['tap'][0].item():.6f}",
            "Loss": f"{loss.item():.6f}"
        })

    # Restore best parameters
    with torch.no_grad():
            if best_params is not None:
                superq.raw_scale.copy_(best_params["raw_scale"])
                superq.raw_exponents.copy_(best_params["raw_exponents"])
                superq.raw_rotation.copy_(best_params["raw_rotation"])
                superq.raw_tapering.copy_(best_params["raw_tapering"])
                superq.translation.copy_(best_params["translation"])

    forward_out = superq.forward()
    sdf_vals = forward_out.get('sdfs')
    visualize_handler(server, superq, sdf_vals, plot=True)
    print(best_params["iou"])
    print(best_params["epoch"])

    while True:
        time.sleep(10.0)

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    main()