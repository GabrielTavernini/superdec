import os
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

from .batch_superq import SuperQ
from ..utils import plot_pred_handler

def visualize_handler(server, superq, sdf_values, outside_values, plot = False):
    sdf_values = sdf_values.detach().cpu().numpy()

    points = superq.points.detach().cpu().numpy()
    pred_handler, meshes = superq.update_handler()
    if plot:
        plot_pred_handler(pred_handler, superq.truncation)

    mesh = meshes[superq.idx]
    server.scene.add_mesh_trimesh("superquadrics", mesh=mesh, visible=True)

    cmap = plt.get_cmap('RdBu')
    norm = plt.Normalize(vmin=-superq.truncation, vmax=superq.truncation)
    sdf_colors = cmap(norm(sdf_values))
    sdf_colors = sdf_colors[:, :3]
    server.scene.add_point_cloud(
        name="/sdf_pointcloud",
        points=points,
        colors=sdf_colors,
        point_size=0.005,
    )

    # 3. Add Normals as Line Segments
    if hasattr(superq, 'normals'):
        outside_values = outside_values.detach().cpu().numpy()
        p1 = points
        p2 = superq.outside_points.detach().cpu().numpy()

        outside_colors = cmap(norm(outside_values))
        outside_colors = outside_colors[:, :3]
        server.scene.add_point_cloud(
            name="/outside_pointcloud",
            points=p2,
            colors=outside_colors,
            point_size=0.005,
        )

def main():
    truncation = 0.05
    # pred_handler = PredictionHandler.from_npz("data/output_npz/sq.npz")
    pred_handler = PredictionHandler.from_npz("data/output_npz/objects/round_table6.npz")
    print(f"Optimizing {pred_handler.names[0]}")
    superq = SuperQ(
        pred_handler=pred_handler,
        truncation=truncation,
        # idx=4,
        # use_full_pointcloud=True,
        ply=f"data/ShapeNet/04379243/{pred_handler.names[0]}/pointcloud.npz",
    )
    param_groups = superq.get_param_groups()
    optimizer = torch.optim.Adam(param_groups)
    
    # center the object
    with torch.no_grad():
        center = torch.mean(superq.points, axis=0)
        superq.translation -= center
        superq.points -= center
        superq.outside_points -= center

    pred_handler, meshes = superq.update_handler()
    orig_mesh = meshes[superq.idx]
    plot_pred_handler(pred_handler, truncation, filename="superq_plot_orig.png")

    server = viser.ViserServer()
    server.scene.set_up_direction([0.0, 1.0, 0.0])
    server.scene.add_mesh_trimesh("original_superquadrics", mesh=orig_mesh, visible=False)

    points = superq.points.detach().cpu().numpy()
    assign_matrix = superq.assign_matrix.detach().cpu().numpy()
    colors = generate_ncolors(assign_matrix.shape[0])
    colored_pc = colors[np.argmax(assign_matrix, axis=0)]
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
        sdf_values, outside_values, counts_points, counts_outside = superq.forward()

        pos_part = torch.clamp(sdf_values, min=0)
        neg_part = torch.clamp(sdf_values, max=0)
        Lsdf = weight_pos * torch.mean(pos_part) + weight_neg * torch.mean(torch.abs(neg_part))
        Lsdf /= weight_pos + weight_neg
        
        outside_ratio = counts_outside / (counts_points + counts_outside + 1e-6)
        scale_weights = 1 + 10.0 * outside_ratio
        Lreg = 0.005 * torch.mean(scale_weights * torch.norm(superq.scale(), p=1, dim=1))

        Lempty = 0.5 * torch.relu(-outside_values).mean()
        
        loss = Lsdf + Lreg + Lempty
        if torch.isnan(loss):
            print("Failed optimization with nan values")
            exit()
        
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(superq.parameters(), max_norm=1.0)
        optimizer.step()

        if epoch % 20 == 0:    
            visualize_handler(server, superq, sdf_values, outside_values)
        pbar.set_postfix({"Lsdf": f"{Lsdf.item():.6f}", "Lempty": f"{Lempty.item():.6f}", "Loss": f"{loss.item():.6f}"})
    visualize_handler(server, superq, sdf_values, outside_values, plot=True)

    while True:
        time.sleep(10.0)

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    main()