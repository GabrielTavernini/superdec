import os
import torch
import numpy as np
from tqdm import tqdm
from superdec.utils.predictions_handler_extended import PredictionHandler
from superdec.utils.evaluation import get_outdict
from .batch_superq import SuperQ, BatchSuperQMulti
import viser
import random

def main():
    input_npz = "data/output_npz/shapenet_test.npz"
    output_npz = "data/output_npz/shapenet_test_tables_optimized.npz"

    # server = viser.ViserServer()
    # server.scene.set_up_direction([0.0, 1.0, 0.0])
    
    # Check if file exists
    if not os.path.exists(input_npz):
        print(f"Error: {input_npz} not found.")
        return

    print(f"Loading {input_npz}...")
    pred_handler = PredictionHandler.from_npz(input_npz)
    
    # Filter for category 04379243
    valid_indices = []
    category_path = "data/ShapeNet/04379243"
    for i, name in enumerate(pred_handler.names):
        if os.path.exists(os.path.join(category_path, name)):
            valid_indices.append(i)

    # valid_indices = valid_indices[:20] # Limit to 20 objects for testing
    
    print(f"Loaded {len(valid_indices)} objects from category 04379243 out of {pred_handler.scale.shape[0]}.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    truncation = 0.05
    num_epochs = 1000
    
    # Store aggregated metrics
    aggregated_metrics = {
        'chamfer-L1': 0.0,
        'chamfer-L2': 0.0,
        'num_primitives': 0.0,
        'count': 0
    }

    batch_size = 32
    
    for i in tqdm(range(0, len(valid_indices), batch_size), desc="Processing batches"):
        batch_indices = valid_indices[i : i + batch_size]
        # print(f"Processing batch {i//batch_size + 1}, indices: {batch_indices}")
        
        ply_paths = [f"data/ShapeNet/04379243/{pred_handler.names[idx]}/pointcloud.npz" for idx in batch_indices]
        
        superq = BatchSuperQMulti(
            pred_handler=pred_handler,
            indices=batch_indices,
            truncation=truncation,
            ply_paths=ply_paths,
            device=device
        )
        
        param_groups = superq.get_param_groups()
        optimizer = torch.optim.Adam(param_groups)
        
        # Center objects
        centers = []
        with torch.no_grad():
             for b in range(len(batch_indices)):
                  mask = superq.points_valid_mask[b]
                  if mask.any():
                      c = superq.points[b][mask].mean(dim=0)
                  else:
                      c = torch.zeros(3, device=device)
                  superq.points[b] -= c
                  if superq.outside_valid_mask[b].any():
                      superq.outside_points[b][superq.outside_valid_mask[b]] -= c
                  superq.translation.data[b] -= c
                  centers.append(c)
        centers = torch.stack(centers)
        
        weight_pos = 2.0
        weight_neg = 1.0
        
        # Optimization Loop
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            # forward: (B, Mp), (B, Mo), (B, N), (B, N)
            sdf_values, outside_values, counts_points, counts_outside = superq.forward()

            pos_part = torch.clamp(sdf_values, min=0)
            neg_part = torch.clamp(sdf_values, max=0)
            
            mask_pts = superq.points_valid_mask
            Lsdf_b = (weight_pos * pos_part + weight_neg * torch.abs(neg_part)) / (weight_pos + weight_neg)
            
            if mask_pts.any():
                Lsdf = Lsdf_b[mask_pts].mean()
            else:
                Lsdf = torch.tensor(0.0, device=device)
            
            outside_ratio = counts_outside / (counts_points + counts_outside + 1e-6)
            scale_weights = 1 + 10.0 * outside_ratio 
            norms = torch.norm(superq.scale(), p=1, dim=2) 
            
            mask_exist = superq.exist_mask
            Lreg_b = scale_weights * norms
            if mask_exist.any():
                Lreg = 0.005 * Lreg_b[mask_exist].mean()
            else:
                Lreg = torch.tensor(0.0, device=device)

            mask_out = superq.outside_valid_mask
            if mask_out.any():
                Lempty_val = torch.relu(-outside_values)
                Lempty = 0.5 * Lempty_val[mask_out].mean()
            else:
                Lempty = torch.tensor(0.0, device=device)
            
            loss = Lsdf + Lreg + Lempty
            
            if torch.isnan(loss):
                break
            
            loss.backward()
            optimizer.step()
        
        # Restore Center
        with torch.no_grad():
            superq.translation.data += centers.unsqueeze(1)

        superq.update_handler(compute_meshes=False)
        
        # Evaluate
        for idx in batch_indices:
             try:
                mesh = pred_handler.get_mesh(idx, resolution=100, colors=False)
             except Exception as e:
                print(f"Error generating mesh for object {idx}: {e}")
                continue

             if mesh is None:
                continue
                
             gt_pc = pred_handler.pc[idx] 
             
             try:
                 num_points = gt_pc.shape[0]
                 pc_pred, idx_face = mesh.sample(num_points, return_index=True)
                 normals_pred = mesh.face_normals[idx_face]
             except Exception as e:
                 continue
                 
             gt_normal = None
            
             out_dict_cur = get_outdict(gt_pc, gt_normal, pc_pred, normals_pred)
             num_prim = (pred_handler.exist[idx] > 0.5).sum()
            
             aggregated_metrics['chamfer-L1'] += out_dict_cur['chamfer-L1']
             aggregated_metrics['chamfer-L2'] += out_dict_cur['chamfer-L2']
             aggregated_metrics['num_primitives'] += num_prim
             aggregated_metrics['count'] += 1

    # Save results
    print(f"Saving optimized results to {output_npz}...")
    pred_handler.save_npz(output_npz)
    
    # Print Metrics
    count = aggregated_metrics['count']
    if count > 0:
        mean_chamfer_l1 = aggregated_metrics['chamfer-L1'] / count
        mean_chamfer_l2 = aggregated_metrics['chamfer-L2'] / count
        mean_num_primitives = aggregated_metrics['num_primitives'] / count
        
        print("\n----- Evaluation Results -----")
        print(f"{'mean_chamfer_l1':>25}: {mean_chamfer_l1:.6f}")
        print(f"{'mean_chamfer_l2':>25}: {mean_chamfer_l2:.6f}")
        print(f"{'avg_num_primitives':>25}: {mean_num_primitives:.6f}")
    else:
        print("No valid objects evaluated.")

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    main()
