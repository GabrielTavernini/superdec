import argparse
import glob
import os
import multiprocessing
import numpy as np
import trimesh
import open3d as o3d
from tqdm import tqdm

def process_file(args):
    input_file, output_root, n_points_surf, n_points_vol = args
    
    try:
        # Determine output directory
        file_name = os.path.basename(input_file)
        obj_id = os.path.splitext(file_name)[0]
        output_dir = os.path.join(output_root, obj_id)
        
        pc_path = os.path.join(output_dir, 'pointcloud.npz')
        points_path = os.path.join(output_dir, 'points.npz')
        path_4096 = os.path.join(output_dir, 'pointcloud_4096.npz')
        
        if os.path.exists(pc_path) and os.path.exists(points_path) and os.path.exists(path_4096):
            return # Skip if already exists
        
        # Load mesh
        # Handle Scene vs Mesh
        mesh_or_scene = trimesh.load(input_file, force='mesh')
        if isinstance(mesh_or_scene, trimesh.Scene):
            if len(mesh_or_scene.geometry) == 0:
                print(f"Empty scene {input_file}")
                return
            # Concatenate all geometries
            mesh = trimesh.util.concatenate(tuple(mesh_or_scene.geometry.values()))
        else:
            mesh = mesh_or_scene

        if not isinstance(mesh, trimesh.Trimesh):
            print(f"Not a mesh: {input_file} {type(mesh)}")
            return
        
        # Normalize
        # Center
        if mesh.bounds is None: return
        center = mesh.bounding_box.centroid
        mesh.apply_translation(-center)
        
        # Scale
        # Fit to unit cube [-0.5, 0.5] (size 1.0)
        max_extent = np.max(mesh.extents)
        scale = 1.0 / max_extent
        mesh.apply_scale(scale)

        # 1. Surface points
        points_surface, face_indices = trimesh.sample.sample_surface(mesh, n_points_surf)
        normals_surface = mesh.face_normals[face_indices]

        # 2. Volume points
        # Random uniform points in [-0.55, 0.55]
        points_vol = np.random.uniform(-0.55, 0.55, (n_points_vol, 3))
        
        # Check occupancy
        occupancies = mesh.contains(points_vol)
        packed_occ = np.packbits(occupancies)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_surface)
        pcd.normals = o3d.utility.Vector3dVector(normals_surface)
        downsampled_pcd = pcd.farthest_point_down_sample(4096)
        
        points_4096 = np.asarray(downsampled_pcd.points)
        normals_4096 = np.asarray(downsampled_pcd.normals)
        
        path_4096 = os.path.join(output_dir, 'pointcloud_4096.npz')
        
        # Save to files
        os.makedirs(output_dir, exist_ok=True)
        np.savez_compressed(pc_path, points=points_surface.astype(np.float32), normals=normals_surface.astype(np.float32), scale=np.float32(scale))
        np.savez_compressed(points_path, points=points_vol.astype(np.float32), occupancies=packed_occ, scale=np.float32(scale))
        np.savez_compressed(path_4096, points=points_4096.astype(np.float32), normals=normals_4096.astype(np.float32), scale=np.float32(scale))
        
    except Exception as e:
        print(f"Error processing {input_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Process ABO dataset.")
    parser.add_argument("--input_dir", type=str, default="data/ABO/raw-complete/3dmodels/original", help="Path to input GLB files")
    parser.add_argument("--output_dir", type=str, default="data/ABO/processed-complete", help="Path to output directory")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of worker processes")
    parser.add_argument("--n_surf", type=int, default=100000, help="Number of surface points")
    parser.add_argument("--n_vol", type=int, default=100000, help="Number of volume points")
    
    args = parser.parse_args()
    
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    
    print(f"Searching for glb files in {input_dir}...")
    files = glob.glob(os.path.join(input_dir, "**/*.glb"), recursive=True)
    if not files:
        files = glob.glob(os.path.join(input_dir, "*.glb"))
    files.sort()

    print(f"Found {len(files)} files.")
    
    if not files:
        return

    os.makedirs(output_dir, exist_ok=True)
    
    tasks = [(f, output_dir, args.n_surf, args.n_vol) for f in files]
    
    with multiprocessing.Pool(args.num_workers) as pool:
        list(tqdm(pool.imap_unordered(process_file, tasks), total=len(tasks)))

if __name__ == "__main__":
    main()
