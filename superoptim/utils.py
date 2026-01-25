import os
import torch
import numpy as np
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

# https://behavior.stanford.edu/reference/utils/transform_utils.html#utils.transform_utils.mat2quat
def mat2quat(rmat: torch.Tensor) -> torch.Tensor:
    """
    Converts given rotation matrix to quaternion.
    Args:
        rmat (torch.Tensor): (3, 3) or (..., 3, 3) rotation matrix
    Returns:
        torch.Tensor: (4,) or (..., 4) (x,y,z,w) float quaternion angles
    """
    assert torch.allclose(torch.linalg.det(rmat), torch.tensor(1.0)), "Rotation matrix must not be scaled"

    # Check if input is a single matrix or a batch
    is_single = rmat.dim() == 2
    if is_single:
        rmat = rmat.unsqueeze(0)

    batch_shape = rmat.shape[:-2]
    mat_flat = rmat.reshape(-1, 3, 3)

    m00, m01, m02 = mat_flat[:, 0, 0], mat_flat[:, 0, 1], mat_flat[:, 0, 2]
    m10, m11, m12 = mat_flat[:, 1, 0], mat_flat[:, 1, 1], mat_flat[:, 1, 2]
    m20, m21, m22 = mat_flat[:, 2, 0], mat_flat[:, 2, 1], mat_flat[:, 2, 2]

    trace = m00 + m11 + m22

    trace_positive = trace > 0
    cond1 = (m00 > m11) & (m00 > m22) & ~trace_positive
    cond2 = (m11 > m22) & ~(trace_positive | cond1)
    cond3 = ~(trace_positive | cond1 | cond2)

    # Trace positive condition
    sq = torch.where(trace_positive, torch.sqrt(trace + 1.0) * 2.0, torch.zeros_like(trace))
    qw = torch.where(trace_positive, 0.25 * sq, torch.zeros_like(trace))
    qx = torch.where(trace_positive, (m21 - m12) / sq, torch.zeros_like(trace))
    qy = torch.where(trace_positive, (m02 - m20) / sq, torch.zeros_like(trace))
    qz = torch.where(trace_positive, (m10 - m01) / sq, torch.zeros_like(trace))

    # Condition 1
    sq = torch.where(cond1, torch.sqrt(1.0 + m00 - m11 - m22) * 2.0, sq)
    qw = torch.where(cond1, (m21 - m12) / sq, qw)
    qx = torch.where(cond1, 0.25 * sq, qx)
    qy = torch.where(cond1, (m01 + m10) / sq, qy)
    qz = torch.where(cond1, (m02 + m20) / sq, qz)

    # Condition 2
    sq = torch.where(cond2, torch.sqrt(1.0 + m11 - m00 - m22) * 2.0, sq)
    qw = torch.where(cond2, (m02 - m20) / sq, qw)
    qx = torch.where(cond2, (m01 + m10) / sq, qx)
    qy = torch.where(cond2, 0.25 * sq, qy)
    qz = torch.where(cond2, (m12 + m21) / sq, qz)

    # Condition 3
    sq = torch.where(cond3, torch.sqrt(1.0 + m22 - m00 - m11) * 2.0, sq)
    qw = torch.where(cond3, (m10 - m01) / sq, qw)
    qx = torch.where(cond3, (m02 + m20) / sq, qx)
    qy = torch.where(cond3, (m12 + m21) / sq, qy)
    qz = torch.where(cond3, 0.25 * sq, qz)

    quat = torch.stack([qx, qy, qz, qw], dim=-1)

    # Normalize the quaternion
    quat = quat / torch.norm(quat, dim=-1, keepdim=True)

    # Reshape to match input batch shape
    quat = quat.reshape(batch_shape + (4,))

    # If input was a single matrix, remove the batch dimension
    if is_single:
        quat = quat.squeeze(0)

    return quat

# https://behavior.stanford.edu/reference/utils/transform_utils.html#utils.transform_utils.quat2mat
def quat2mat(quaternion):
    """
    Convert quaternions into rotation matrices.

    Args:
        quaternion (torch.Tensor): A tensor of shape (..., 4) representing batches of quaternions (x, y, z, w).

    Returns:
        torch.Tensor: A tensor of shape (..., 3, 3) representing batches of rotation matrices.
    """
    quaternion = quaternion / torch.norm(quaternion, dim=-1, keepdim=True)

    outer = quaternion.unsqueeze(-1) * quaternion.unsqueeze(-2)

    # Extract the necessary components
    xx = outer[..., 0, 0]
    yy = outer[..., 1, 1]
    zz = outer[..., 2, 2]
    xy = outer[..., 0, 1]
    xz = outer[..., 0, 2]
    yz = outer[..., 1, 2]
    xw = outer[..., 0, 3]
    yw = outer[..., 1, 3]
    zw = outer[..., 2, 3]

    rmat = torch.empty(quaternion.shape[:-1] + (3, 3), dtype=quaternion.dtype, device=quaternion.device)

    rmat[..., 0, 0] = 1 - 2 * (yy + zz)
    rmat[..., 0, 1] = 2 * (xy - zw)
    rmat[..., 0, 2] = 2 * (xz + yw)

    rmat[..., 1, 0] = 2 * (xy + zw)
    rmat[..., 1, 1] = 1 - 2 * (xx + zz)
    rmat[..., 1, 2] = 2 * (yz - xw)

    rmat[..., 2, 0] = 2 * (xz - yw)
    rmat[..., 2, 1] = 2 * (yz + xw)
    rmat[..., 2, 2] = 1 - 2 * (xx + yy)

    return rmat