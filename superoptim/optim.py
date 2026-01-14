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
from superq import SuperQ

from scipy.spatial.transform import Rotation   

def sdf_superquadric(points, scale_vec, exponents, translation, rotation_matrix, truncation=1):
    """
    Computes the Signed Distance Function for a Superquadric.
    
    Args:
        points: (3, N) array of query points.
        scale_vec: (3,) array [sx, sy, sz].
        exponents: (2,) array [e1, e2].
        translation: (3,) array [tx, ty, tz].
        rotation_matrix: (3, 3) rotation matrix.
        truncation: float, distance limit (0 to disable).
    """
    # 1. Transform points to local coordinate system
    # X = R' * (points - t)
    # Note: rotation_matrix.T is equivalent to R'
    points_centered = points - translation[:, np.newaxis]
    X = rotation_matrix.T @ points_centered

    # 2. Extract parameters for readability
    e1, e2 = exponents
    sx, sy, sz = scale_vec

    # 3. Calculate radial distance from origin
    r0 = np.linalg.norm(X, axis=0)

    # 4. Calculate the Superquadric scaling function
    # Formula components: (((x/sx)^2)^(1/e2) + ((y/sy)^2)^(1/e2))^(e2/e1) + ((z/sz)^2)^(1/e1)
    term1 = ((X[0, :] / sx)**2)**(1 / e2)
    term2 = ((X[1, :] / sy)**2)**(1 / e2)
    term3 = ((X[2, :] / sz)**2)**(1 / e1)
    
    f = ( (term1 + term2)**(e2 / e1) + term3 )**(-e1 / 2)

    # 5. Compute Signed Distance
    sdf = r0 * (1 - f)

    # 6. Apply truncation
    if truncation != 0:
        sdf = np.clip(sdf, -truncation, truncation)

    return sdf

def plot_sdf_multi_slice(y_world, limit, scale, exp, trans, rot, filename="superq_plot.png"):
    res = 200
    grid_range = 0.75
    x_range = np.linspace(-grid_range, grid_range, res)
    z_range = np.linspace(-grid_range, grid_range, res)
    X_grid, Z_grid = np.meshgrid(x_range, z_range)
    
    # Reshape grid into (3, N) query points
    Y_grid = np.full_like(X_grid, y_world)
    points = np.stack([X_grid.ravel(), Y_grid.ravel(), Z_grid.ravel()], axis=0)

    sdf_values = sdf_superquadric(points, scale[0], exp[0], trans[0], rot[0], truncation=limit)
    mixed_values = sdf_values.reshape(res, res)
    for i in range(1, len(scale)):
        sdf_values = sdf_superquadric(points, scale[i], exp[i], trans[i], rot[i], truncation=limit)
        values = sdf_values.reshape(res, res)
        mixed_values = np.minimum(mixed_values, values)

    # Save to picture
    plt.figure(figsize=(8, 6))
    mesh = plt.pcolormesh(x_range, z_range, mixed_values, 
                          shading='auto', 
                          cmap='RdBu', 
                          vmin=-limit, 
                          vmax=limit)
    
    # Add a contour line at 0 to show the actual surface boundary
    plt.contour(x_range, z_range, mixed_values, levels=[0], colors='black', linewidths=2)
    
    plt.colorbar(mesh, label='Signed Distance')
    plt.xlabel('X (World)')
    plt.ylabel('Z (World)')
    plt.title(f'Superquadric SDF Slice at Y={y_world}')
    
    # plt.axis('equal')
    plt.savefig(filename)
    plt.close()
    print(f"Plot saved as {filename}")

def plot_pred_handler(pred_handler, truncation, wolrd_y=0.15, filename="superq_plot.png"):
    mask = (pred_handler.exist > 0.5).reshape(-1)
    sqscale = np.array(pred_handler.scale.reshape(-1, 3)[mask])
    exponents = np.array(pred_handler.exponents.reshape(-1, 2)[mask])
    translation = np.array(pred_handler.translation.reshape(-1, 3)[mask])
    rotation = np.array(pred_handler.rotation.reshape(-1, 3, 3)[mask])
    plot_sdf_multi_slice(wolrd_y, truncation, sqscale, exponents, translation, rotation, filename=filename)

def main():
    truncation = 0.05
    pred_handler = PredictionHandler.from_npz("test.npz")
    superq = SuperQ(
        pred_handler=pred_handler,
        truncation=truncation,
        # ply="examples/chair.ply"
    )
    optimizer = torch.optim.Adam(superq.parameters(), lr=1e-4)
    
    pred_handler, meshes = superq.update_handler()
    orig_mesh = meshes[0]
    plot_pred_handler(pred_handler, truncation, filename="superq_plot_orig.png")

    num_epochs = 1000
    weight_pos = 2.0
    weight_neg = 1.0
    pbar = tqdm(range(num_epochs), desc="Fitting Superquadrics")
    for epoch in pbar:
        optimizer.zero_grad()
        
        sdf_values = superq.forward()
        # loss = torch.mean(torch.abs(sdf_values))

        pos_part = torch.clamp(sdf_values, min=0)
        neg_part = torch.clamp(sdf_values, max=0)
        loss = weight_pos * torch.mean(pos_part) + weight_neg * torch.mean(torch.abs(neg_part))
        loss /= weight_pos + weight_neg
        if torch.isnan(loss):
            print("Failed optimization with nan values")
            exit()
        
        loss.backward()
        optimizer.step()
        pbar.set_postfix({"Loss": f"{loss.item():.6f}"})

    sdf_values = superq.forward().detach().cpu().numpy()
    
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