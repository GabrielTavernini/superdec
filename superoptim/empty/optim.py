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

def visualize_handler(server, superq, sdf_values, outside_values, plot = False):
    sdf_values = sdf_values.detach().cpu().numpy()

    points = superq.points.detach().cpu().numpy()
    pred_handler, meshes = superq.update_handler()
    if plot:
        plot_pred_handler(pred_handler, superq.truncation)

    mesh = meshes[0]
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
        # line_segments = np.stack([p1, p2[:p1.shape[0],:]], axis=1)

        # server.scene.add_line_segments(
        #     name="/normals",
        #     points=line_segments,
        #     colors=np.array([0.0, 1.0, 0.0]),
        #     line_width=0.75,
        # )

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
    pred_handler = PredictionHandler.from_npz("data/output_npz/objects/rect_table.npz")
    print(f"Optimizing {pred_handler.names[0]}")
    superq = SuperQ(
        pred_handler=pred_handler,
        truncation=truncation,
        use_full_pointcloud=True,
        ply="data/ShapeNet/04379243/d1296da8d3a45ac88aaae3d020f5ddf8/pointcloud.npz",
    )
    param_groups = superq.get_param_groups()
    optimizer = torch.optim.Adam(param_groups)
    
    pred_handler, meshes = superq.update_handler()
    orig_mesh = meshes[0]
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
    num_epochs = 5000
    weight_pos = 2.0
    weight_neg = 1.0
    pbar = tqdm(range(num_epochs), desc="Fitting Superquadrics")
    for epoch in pbar:
        optimizer.zero_grad()
        
        sdf_values, outside_values = superq.forward()
        # loss = torch.mean(torch.abs(sdf_values))

        pos_part = torch.clamp(sdf_values, min=0)
        neg_part = torch.clamp(sdf_values, max=0)
        Lsdf = weight_pos * torch.mean(pos_part) + weight_neg * torch.mean(torch.abs(neg_part))
        Lsdf /= weight_pos + weight_neg
        
        # volumes = torch.prod(superq.scale(), dim=1)
        # Lreg = 2 * torch.mean(volumes)
        Lreg = 0.005 * torch.norm(superq.scale(), p=1, dim=1).mean()

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
    main()