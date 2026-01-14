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

import matplotlib.pyplot as plt

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

def plot_sdf_slice(y_world, scale, exp, trans, rot, filename="superq_plot.png"):
    limit = 0.5

    res = 200
    x_range = np.linspace(-2, 2, res)
    z_range = np.linspace(-2, 2, res)
    X_grid, Z_grid = np.meshgrid(x_range, z_range)
    
    # Reshape grid into (3, N) query points
    Y_grid = np.full_like(X_grid, y_world)
    points = np.stack([X_grid.ravel(), Y_grid.ravel(), Z_grid.ravel()], axis=0)

    sdf_values = sdf_superquadric(points, scale, exp, trans, rot, truncation=limit)
    values = sdf_values.reshape(res, res)

    # Save to picture
    plt.figure(figsize=(8, 6))
    mesh = plt.pcolormesh(x_range, z_range, values, 
                          shading='auto', 
                          cmap='RdBu', 
                          vmin=-limit, 
                          vmax=limit)
    
    # Add a contour line at 0 to show the actual surface boundary
    plt.contour(x_range, z_range, values, levels=[0], colors='black', linewidths=2)
    
    plt.colorbar(mesh, label='Signed Distance')
    plt.xlabel('X (World)')
    plt.ylabel('Z (World)')
    plt.title(f'Superquadric SDF Slice at Y={y_world}')
    
    # plt.axis('equal')
    plt.savefig(filename)
    plt.close()
    print(f"Plot saved as {filename}")

def plot_sdf_multi_slice(y_world, scale, exp, trans, rot, filename="superq_plot.png"):
    limit = 0.1

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

def main():
    pred_handler = PredictionHandler.from_npz("test.npz")
    mesh = pred_handler.get_meshes(resolution=30)[0]
    pcs = pred_handler.get_segmented_pcs()[0]

    mask = (pred_handler.exist > 0.5).reshape(-1)
    sqscale = np.array(pred_handler.scale.reshape(-1, 3)[mask])
    exponents = np.array(pred_handler.exponents.reshape(-1, 2)[mask])
    translation = np.array(pred_handler.translation.reshape(-1, 3)[mask])
    rotation = np.array(pred_handler.rotation.reshape(-1, 3, 3)[mask])
    assign_matrix = np.array(pred_handler.assign_matrix.T[mask]).squeeze()

    # Extract the first superquadric's parameters
    # idx = 0
    # s = sqscale[idx]
    # e = exponents[idx]
    # t = translation[idx]
    # r = rotation[idx]
    # plot_sdf_slice(-0.2, s, e, t, r)

    # Plot a slice
    plot_sdf_multi_slice(-0.19, sqscale, exponents, translation, rotation)

    limit = 0.1
    points = np.array(pcs.points)
    sdf_values = np.zeros((4096))
    for i in range(len(sqscale)):
        p_mask = (assign_matrix[i] == 1)
        v = sdf_superquadric(points[p_mask].T, sqscale[i], exponents[i], translation[i], rotation[i], truncation=limit)
        sdf_values[p_mask] = v 

    # all sqs together
    # points = np.array(pcs.points).T
    # sdf_values = sdf_superquadric(points, sqscale[0], exponents[0], translation[0], rotation[0], truncation=limit)
    # for i in range(1, len(sqscale)):
    #     sdf_values = np.minimum(
    #         sdf_superquadric(points, sqscale[i], exponents[i], translation[i], rotation[i], truncation=limit),
    #         sdf_values
    #     )

    cmap = plt.get_cmap('RdBu')
    norm = plt.Normalize(vmin=-limit, vmax=limit)
    sdf_colors = cmap(norm(sdf_values))
    sdf_colors = sdf_colors[:, :3]

    server = viser.ViserServer()
    server.scene.add_mesh_trimesh("superquadrics", mesh=mesh, visible=True)
    
    server.scene.add_point_cloud(
        name="/segmented_pointcloud",
        points=np.array(pcs.points),
        # colors=np.array(pcs.colors),
        colors=sdf_colors,
        point_size=0.005,
    )
    server.scene.set_up_direction([0.0, 1.0, 0.0])

    while True:
        time.sleep(10.0)

if __name__ == "__main__":
    main()