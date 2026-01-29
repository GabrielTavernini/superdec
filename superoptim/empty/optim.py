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

def visualize_handler(server, superq, sdf_values, outside_values, plot = False):
    # Expect batched tensors from BatchSuperQMulti; use batch 0
    sdf_values = sdf_values.detach().cpu()
    outside_values = outside_values.detach().cpu()

    pred_handler, meshes = superq.update_handler()
    batch_idx = 0
    obj_idx = superq.indices[batch_idx]
    if plot:
        plot_pred_handler(pred_handler, superq.truncation)

    mesh = meshes[obj_idx]
    server.scene.add_mesh_trimesh("superquadrics", mesh=mesh, visible=True)

    points = superq.points[batch_idx].detach().cpu().numpy()

    cmap = plt.get_cmap('RdBu')
    norm = plt.Normalize(vmin=-superq.truncation, vmax=superq.truncation)
    sdf_arr = sdf_values[batch_idx].numpy()
    sdf_colors = cmap(norm(sdf_arr))[:, :3]
    server.scene.add_point_cloud(
        name="/sdf_pointcloud",
        points=points,
        colors=sdf_colors,
        point_size=0.005,
    )

    # Add outside points
    p2 = superq.outside_points[batch_idx].detach().cpu().numpy()
    outside_arr = outside_values[batch_idx].numpy()
    outside_colors = cmap(norm(outside_arr))[:, :3]
    server.scene.add_point_cloud(
        name="/outside_pointcloud",
        points=p2,
        colors=outside_colors,
        point_size=0.005,
    )

def main():
    if len(sys.argv) > 1:
        object_name = sys.argv[1]
    else:
        object_name = "round_table"
    pred_handler = PredictionHandler.from_npz(f"data/output_npz/objects/{object_name}.npz")
    print(f"Optimizing {pred_handler.names[0]}")
    
    truncation = 0.05
    superq = BatchSuperQMulti(
        pred_handler=pred_handler,
        truncation=truncation,
        indices=[0],
        ply_paths=[f"data/ShapeNet/04379243/{pred_handler.names[0]}/pointcloud.npz"],
    )
    param_groups = superq.get_param_groups()
    optimizer = torch.optim.Adam(param_groups)
    
    # center the object (batch 0)
    with torch.no_grad():
        b = 0
        center = torch.mean(superq.points[b], dim=0)
        superq.points[b] -= center
        superq.outside_points[b] -= center
        superq.translation.data[b] -= center

    pred_handler, meshes = superq.update_handler()
    orig_mesh = meshes[superq.indices[0]]
    plot_pred_handler(pred_handler, truncation, filename="superq_plot_orig.png")

    server = viser.ViserServer()
    server.scene.set_up_direction([0.0, 1.0, 0.0])
    server.scene.add_mesh_trimesh("original_superquadrics", mesh=orig_mesh, visible=False)

    # Segmented pointcloud for batch 0
    points = superq.points[0].detach().cpu().numpy()
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

    # torch.autograd.set_detect_anomaly(True)
    num_epochs = 1000
    weight_pos = 2.0
    weight_neg = 1.0
    pbar = tqdm(range(num_epochs), desc="Fitting Superquadrics")
    for epoch in pbar:
        optimizer.zero_grad()
        forward_out = superq.forward()
        sdf_vals = forward_out.get('sdf_values')
        outside_vals = forward_out.get('outside_values')

        # Use shared loss computation (per-batch)
        loss, losses = superq.compute_losses(forward_out, weight_pos=weight_pos, weight_neg=weight_neg)
        Lsdf, Lreg, Lempty = losses['Lsdf'][0], losses['Lreg'][0], losses['Lempty'][0]
        loss = loss[0]

        if torch.isnan(loss):
            print("Failed optimization with nan values")
            exit()
        
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(superq.parameters(), max_norm=1.0)
        optimizer.step()

        if epoch % 20 == 0:
            visualize_handler(server, superq, sdf_vals, outside_vals)
        pbar.set_postfix({"Lsdf": f"{Lsdf.item():.6f}", "Lempty": f"{Lempty.item():.6f}", "Loss": f"{loss.item():.6f}"})
    visualize_handler(server, superq, sdf_vals, outside_vals, plot=True)

    while True:
        time.sleep(10.0)

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    main()