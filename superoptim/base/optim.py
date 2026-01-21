import os
import torch
import numpy as np
from omegaconf import OmegaConf
from superdec.superdec import SuperDec
from superdec.utils.predictions_handler import PredictionHandler
from superdec.data.dataloader import denormalize_outdict, denormalize_points
import open3d as o3d
import viser
from superdec.data.dataloader import normalize_points, denormalize_outdict
from superdec.data.transform import rotate_around_axis
import time
from superdec.utils.visualizations import generate_ncolors
from tqdm import tqdm
import matplotlib.pyplot as plt

from .superq import SuperQ
from ..utils import plot_pred_handler


def main():
    truncation = 0.05
    pred_handler = PredictionHandler.from_npz("data/output_npz/objects/rect_table.npz")
    superq = SuperQ(
        pred_handler=pred_handler,
        truncation=truncation,
        # ply="examples/chair.ply"
    )
    param_groups = superq.get_param_groups()
    optimizer = torch.optim.Adam(param_groups)
    
    pred_handler, meshes = superq.update_handler()
    orig_mesh = meshes[0]
    plot_pred_handler(pred_handler, truncation, filename="superq_plot_orig.png")

    # torch.autograd.set_detect_anomaly(True)
    num_epochs = 1000
    weight_pos = 2.0
    weight_neg = 1.0
    weight_scale = 0.01
    pbar = tqdm(range(num_epochs), desc="Fitting Superquadrics")
    for epoch in pbar:
        optimizer.zero_grad()
        
        sdf_values = superq.forward()
        # loss = torch.mean(torch.abs(sdf_values))

        pos_part = torch.clamp(sdf_values, min=0)
        neg_part = torch.clamp(sdf_values, max=0)
        Lsdf = weight_pos * torch.mean(pos_part) + weight_neg * torch.mean(torch.abs(neg_part))
        Lsdf /= weight_pos + weight_neg
        
        # volumes = torch.prod(superq.scale(), dim=1)
        # Lreg = weight_scale * torch.mean(volumes)
        Lreg = weight_scale * torch.norm(superq.scale(), p=1, dim=1).mean()
        
        loss = Lsdf + Lreg
        if torch.isnan(loss):
            print("Failed optimization with nan values")
            exit()
        
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(superq.parameters(), max_norm=1.0)
        optimizer.step()
        pbar.set_postfix({"Lsdf": f"{Lsdf.item():.6f}", "Loss": f"{loss.item():.6f}"})
    sdf_values = sdf_values.detach().cpu().numpy()
    
    pred_handler, meshes = superq.update_handler()
    mesh = meshes[0]
    plot_pred_handler(pred_handler, truncation)

    cmap = plt.get_cmap('RdBu')
    norm = plt.Normalize(vmin=-truncation, vmax=truncation)
    sdf_colors = cmap(norm(sdf_values))
    sdf_colors = sdf_colors[:, :3]

    server = viser.ViserServer()
    server.scene.add_mesh_trimesh("superquadrics_orig", mesh=orig_mesh, visible=False)

    points = superq.points.detach().cpu().numpy()
    assign_matrix = superq.assign_matrix.detach().cpu().numpy()
    colors = generate_ncolors(assign_matrix.shape[0])
    colored_pc = colors[np.argmax(assign_matrix, axis=0)]
    server.scene.add_point_cloud(
        name="/segmented_pointcloud_orig",
        points=points,
        colors=colored_pc,
        point_size=0.005,
        visible=False,
    )

    server.scene.add_mesh_trimesh("superquadrics", mesh=mesh, visible=True)
    server.scene.add_point_cloud(
        name="/segmented_pointcloud",
        points=points,
        colors=sdf_colors,
        point_size=0.005,
    )
    server.scene.set_up_direction([0.0, 1.0, 0.0])

    while True:
        time.sleep(10.0)

if __name__ == "__main__":
    main()