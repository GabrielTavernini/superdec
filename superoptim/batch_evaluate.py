import os
import torch
import numpy as np
from tqdm import tqdm
from superdec.utils.predictions_handler_extended import PredictionHandler
import viser
import random
import argparse
import importlib
import gc

from .evaluation import get_outdict, eval_mesh

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default="empty", help="Type of superq module (empty, segmented, etc.)")
    parser.add_argument("--prefix", type=str, default="shapenet_test", help="Npz file prefix")
    args = parser.parse_args()

    try:
        # Import dynamically based on type
        module_name = f"superoptim.{args.type}.batch_superq"
        if args.type == "none":
            module_name = f"superoptim.iou.batch_superq"
        module = importlib.import_module(module_name)
        BatchSuperQMulti = module.BatchSuperQMulti
    except ImportError:
        print(f"Error importing BatchSuperQMulti for type '{args.type}': {e}")
        return

    print(f"Evaluating {args.type} on {args.prefix}...")
    input_npz = f"data/output_npz/{args.prefix}.npz"
    output_npz = f"data/output_npz/{args.prefix}_optimized_{args.type}.npz"

    # server = viser.ViserServer()
    # server.scene.set_up_direction([0.0, 1.0, 0.0])
    
    # Check if file exists
    if not os.path.exists(input_npz):
        print(f"Error: {input_npz} not found.")
        return

    print(f"Loading {input_npz}...")
    pred_handler = PredictionHandler.from_npz(input_npz)
    
    # Filter for category 04379243
    valid_objs = []
    data_root = "data/ShapeNet"
    for c in os.listdir(data_root):
        category_path = os.path.join(data_root, c)
        for i, name in enumerate(pred_handler.names):
            if os.path.exists(os.path.join(category_path, name)):
                valid_objs.append((i, category_path))

    # valid_objs = valid_objs[:32] # Limit to 32 objects for testing
    print(f"Loaded {len(valid_objs)} objects from all categories out of {pred_handler.scale.shape[0]}.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_epochs = 1_000
    if args.type == "none": num_epochs = 0
    
    # Store aggregated metrics
    aggregated_metrics = {
        'chamfer-L1': 0.0,
        'chamfer-L2': 0.0,
        'iou': 0.0,
        'f-score': 0.0,
        'f-score-15': 0.0,
        'f-score-20': 0.0,
        'num_primitives': 0.0,
        'count': 0
    }
    
    # Store per-object metrics for ranking
    object_metrics = [] # List of tuples: (index, name, chamfer_l1)
    batch_size = 128
    
    for i in tqdm(range(0, len(valid_objs), batch_size), desc="Processing batches"):
        batch_objs = valid_objs[i : i + batch_size]
        batch_indices = [x[0] for x in batch_objs]
        # print(f"Processing batch {i//batch_size + 1}, indices: {batch_indices}")
        
        ply_paths = [f"{category_path}/{pred_handler.names[idx]}/pointcloud.npz" for idx, category_path in batch_objs]
        
        superq = BatchSuperQMulti(
            pred_handler=pred_handler,
            indices=batch_indices,
            ply_paths=ply_paths,
            device=device
        )
        
        param_groups = superq.get_param_groups()
        optimizer = torch.optim.Adam(param_groups)
        

        best_losses = [float('inf')] * len(batch_indices)
        best_params = [None] * len(batch_indices)        
        
        # Optimization Loop
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            # forward: returns a dict output
            forward_out = superq.forward()

            # Compute per-batch losses using shared function
            loss, _ = superq.compute_losses(forward_out)
            batch_loss = loss.mean()

            if torch.isnan(batch_loss):
                print(f"nan loss at epoch {epoch}")
                break
            
            batch_loss.backward()
            optimizer.step()

            # Save best parameters (based on Total Loss per object)
            with torch.no_grad():
                for b in range(len(batch_indices)):
                    current_loss = loss[b].item()
                    if current_loss < best_losses[b]:
                        best_losses[b] = current_loss
                        best_params[b] = {
                            "raw_scale": superq.raw_scale[b].clone(),
                            "raw_exponents": superq.raw_exponents[b].clone(),
                            "raw_rotation": superq.raw_rotation[b].clone(),
                            "raw_tapering": superq.raw_tapering[b].clone(),
                            "translation": superq.translation[b].clone()
                        }
        
        # Restore best parameters
        with torch.no_grad():
             for b in range(len(batch_indices)):
                  if best_params[b] is not None:
                       superq.raw_scale[b].copy_(best_params[b]["raw_scale"])
                       superq.raw_exponents[b].copy_(best_params[b]["raw_exponents"])
                       superq.raw_rotation[b].copy_(best_params[b]["raw_rotation"])
                       superq.raw_tapering[b].copy_(best_params[b]["raw_tapering"])
                       superq.translation[b].copy_(best_params[b]["translation"])


        superq.update_handler(compute_meshes=False)
        
        # Cleanup to avoid OOM
        del superq
        del optimizer
        del param_groups
        del best_params
        del best_losses
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Evaluate
        for b_idx in range(len(batch_indices)):
            idx = batch_indices[b_idx]
            try:
                mesh = pred_handler.get_mesh(idx, resolution=100, colors=False)
            except Exception as e:
                print(f"Error generating mesh for object {idx}: {e}")
                continue

            if mesh is None:
                continue

            gt_pc = pred_handler.pc[idx] 
            gt_normal = None
            
            points_iou = None
            occ_tgt = None
            obj_name = pred_handler.names[idx]
            points_file = os.path.join(category_path, obj_name, "points.npz")

            if os.path.exists(points_file):
                try:
                    points_dict = np.load(points_file)
                    points_iou = points_dict['points']
                    occ_tgt = points_dict['occupancies']
                    if np.issubdtype(occ_tgt.dtype, np.uint8):
                        occ_tgt = np.unpackbits(occ_tgt)[:points_iou.shape[0]]
                except Exception as e:
                    print(f"Failed to load points.npz for {obj_name}: {e}")
            
            try:
                out_dict_cur = eval_mesh(mesh, gt_pc, gt_normal, points_iou, occ_tgt)
            except Exception as e:
                print(f"Eval mesh failed: {e}")
                continue
            
            num_prim = (pred_handler.exist[idx] > 0.5).sum()
            aggregated_metrics['chamfer-L1'] += out_dict_cur['chamfer-L1']
            aggregated_metrics['chamfer-L2'] += out_dict_cur['chamfer-L2']
            aggregated_metrics['iou'] += out_dict_cur.get('iou', 0.0)
            aggregated_metrics['f-score'] += out_dict_cur.get('f-score', 0.0)
            aggregated_metrics['f-score-15'] += out_dict_cur.get('f-score-15', 0.0)
            aggregated_metrics['f-score-20'] += out_dict_cur.get('f-score-20', 0.0)
            aggregated_metrics['num_primitives'] += num_prim
            aggregated_metrics['count'] += 1
            
            object_metrics.append({
                'index': idx,
                'name': pred_handler.names[idx],
                'chamfer-L1': out_dict_cur['chamfer-L1'],
                'chamfer-L2': out_dict_cur['chamfer-L2'],
                'iou': out_dict_cur.get('iou', 0.0),
                'f-score': out_dict_cur.get('f-score', 0.0),
                'f-score-15': out_dict_cur.get('f-score-15', 0.0),
                'f-score-20': out_dict_cur.get('f-score-20', 0.0),
                'num_primitives': int(num_prim)
            })

    # Save results
    print(f"Saving optimized results to {output_npz}...")
    pred_handler.save_npz(output_npz)
    
    # Save detailed metrics
    metrics_csv = output_npz.replace(".npz", "_metrics.csv")
    print(f"Saving per-object metrics to {metrics_csv}...")
    import csv
    with open(metrics_csv, 'w', newline='') as csvfile:
        fieldnames = ['index', 'name', 'chamfer-L1', 'chamfer-L2', 'iou', 'f-score', 'f-score-15', 'f-score-20', 'num_primitives']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for data in object_metrics:
            writer.writerow(data)

    # Print Metrics
    count = aggregated_metrics['count']
    if count > 0:
        mean_chamfer_l1 = aggregated_metrics['chamfer-L1'] / count
        mean_chamfer_l2 = aggregated_metrics['chamfer-L2'] / count
        mean_iou = aggregated_metrics['iou'] / count
        mean_fscore = aggregated_metrics['f-score'] / count
        mean_fscore_15 = aggregated_metrics['f-score-15'] / count
        mean_fscore_20 = aggregated_metrics['f-score-20'] / count
        mean_num_primitives = aggregated_metrics['num_primitives'] / count
        
        print("\n----- Evaluation Results -----")
        print(f"{'mean_chamfer_l1':>25}: {mean_chamfer_l1:.6f}")
        print(f"{'mean_chamfer_l2':>25}: {mean_chamfer_l2:.6f}")
        print(f"{'mean_iou':>25}: {mean_iou:.6f}")
        print(f"{'mean_f-score':>25}: {mean_fscore:.6f}")
        print(f"{'mean_f-score-15':>25}: {mean_fscore_15:.6f}")
        print(f"{'mean_f-score-20':>25}: {mean_fscore_20:.6f}")
        print(f"{'avg_num_primitives':>25}: {mean_num_primitives:.6f}")
        
        # Sort by Chamfer-L1 descending (worst first)
        object_metrics.sort(key=lambda x: x['chamfer-L1'], reverse=True)
        print("\n----- Top 10 Worst Objects (by Chamfer-L1) -----")
        print(f"{'Index':<10} {'Name':<40} {'Chamfer-L1':<15}")
        for item in object_metrics[:10]:
            print(f"{item['index']:<10} {item['name']:<40} {item['chamfer-L1']:.6f}")
    else:
        print("No valid objects evaluated.")

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    main()
