from scipy.spatial.transform import Rotation as R
import open3d as o3d
import open3d.visualization.rendering as rendering
import numpy as np
import open3d as o3d
import viser
import time
import math

def camera_to_world(location, up):
    """
    Computes the camera-to-world transformation matrix.
    Handles the edge case where the camera is looking straight up or down.
    """
    # Ensure inputs are numpy arrays
    location = np.array(location, dtype=float)
    up = np.array(up, dtype=float)

    # 1. Forward vector (camera looks at the origin)
    forward = -location
    fnorm = np.linalg.norm(forward)
    
    # Handle case where camera is exactly at origin (undefined direction)
    if fnorm < 1e-6:
        return np.eye(4)
    forward /= fnorm

    # 2. Right vector
    # Attempt to compute right vector using the provided UP
    right = np.cross(forward, up)
    rnorm = np.linalg.norm(right)

    # --- GIMBAL LOCK FIX ---
    # If forward and up are parallel (dot product ~1 or -1), the cross product is zero.
    # We must choose an arbitrary different 'up' to calculate 'right'.
    print(forward, up, right)
    if rnorm < 1e-4:
        print("GIMBAL LOCK FIX")
        # If we are looking along Z, use X as the helper. Otherwise use Z.
        # (Any vector not parallel to forward works)
        helper = np.array([1, 0, 0]) if abs(forward[1]) < 0.9 else np.array([0, 0, 1])
        right = np.cross(forward, helper)
        rnorm = np.linalg.norm(right)
    # -----------------------

    right /= rnorm

    # 3. Recompute True Up vector (orthogonal to forward and right)
    real_up = np.cross(forward, right)
    real_up /= np.linalg.norm(real_up)

    # 4. Build Rotation Matrix
    # The columns are the basis vectors of the camera in world space
    rotation_matrix = np.column_stack((right, real_up, forward))

    # 5. Build 4x4 Matrix
    cam2world = np.eye(4)
    cam2world[:3, :3] = rotation_matrix
    cam2world[:3, 3] = location

    return cam2world

def show_cam(i, c2w, server, special=False):
    c2w = c2w
    R_wc = c2w[:3, :3]
    t_wc = c2w[:3, 3]
    # print(t_wc)

    # Convert rotation matrix → quaternion (wxyz for viser)
    quat_xyzw = R.from_matrix(R_wc).as_quat()  # (x,y,z,w)
    qx, qy, qz, qw = quat_xyzw
    quat_wxyz = (qw, qx, qy, qz)

    server.scene.add_camera_frustum(
        name=f"cam_{i}",
        fov=60,
        aspect=1,
        scale=0.1,
        line_width=2.0,
        color=(0.2, 0.2, 0.8) if special else (0.8, 0.2, 0.8),
        wxyz=quat_wxyz,
        position=t_wc,
        visible=True,
        variant="wireframe",
    )

# Helper functions for vector math
def normalize(v):
    norm = math.sqrt(sum(x*x for x in v))
    return [x/norm for x in v]

def cross(a, b):
    return [a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0]]

def main(): 
    server = viser.ViserServer() 

    pcd = o3d.io.read_point_cloud('examples/chair.ply') 
    points = np.array(pcd.points) 
    colors = (np.ones(points.shape) * 255).astype(np.uint8) 
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, 0.1) 
    mesh.compute_vertex_normals() 
    render = rendering.OffscreenRenderer(640, 480) 
    
    green = rendering.MaterialRecord() 
    green.base_color = [0.0, 0.5, 0.0, 1.0] 
    green.shader = "defaultLit" 

    render.scene.add_geometry("chair", mesh, green) 
    # render.scene.show_axes(True)

    dist = 1.5
    num_samples = 40

    center = [0, 0, 0]
    up = [0, 0, 1]
    z_axis = normalize(up)
    if abs(z_axis[2]) > 0.9:
        helper = [0, 1, 0]
    else:
        helper = [0, 0, 1]
    x_axis = normalize(cross(helper, z_axis))
    y_axis = cross(z_axis, x_axis)

    cameras = []
    golden_angle = math.pi * (3.0 - math.sqrt(5.0)) 
    for i in range(num_samples):
        y_local = 1 - (i / float(max(1, num_samples - 1))) * 2  
        
        # Calculate radius at this height (horizontal ring radius)
        radius_at_y = math.sqrt(1 - y_local * y_local)
        
        # Calculate the angle around the ring
        theta = golden_angle * i

        # Get x and z positions on the local unit sphere
        x_local = math.cos(theta) * radius_at_y
        z_local = math.sin(theta) * radius_at_y # This acts as the second horizontal axis

        # 3. Map local sphere to world space using your basis vectors
        # We treat y_local as the component along your "up" (z_axis)
        # We treat x_local and z_local as components along x_axis and y_axis
        
        px = center[0] + dist * (x_local * x_axis[0] + z_local * y_axis[0] + y_local * z_axis[0])
        py = center[1] + dist * (x_local * x_axis[1] + z_local * y_axis[1] + y_local * z_axis[1])
        pz = center[2] + dist * (x_local * x_axis[2] + z_local * y_axis[2] + y_local * z_axis[2])

        cameras.append([px, py, pz])

    # Optional: print first few points to verify
    for i, p in enumerate(cameras):
        render.setup_camera(60.0, [0, 0, 0], p, [0, 0, 1]) 
        img = render.render_to_image() 
        o3d.io.write_image(f"tmp_chair/imgs/{i}.png", img, 9) 
        
        c2w = camera_to_world(p, up)
        np.save(f"tmp_chair/c2w/{i}.npy", c2w)
        show_cam(i, c2w, server)    
        print(f"saved img {i}")
    
    server.scene.add_point_cloud( name="/pointcloud", points=points, colors=colors, point_size=0.05 ) 
    server.scene.set_up_direction([0.0, 0.0, 1.0]) 
    
    while True:
        time.sleep(10.0)

if __name__ == "__main__": 
    main()