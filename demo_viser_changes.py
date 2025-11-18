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
def main():
    eval_steps = [1, 3_000, 7_000, 30_000]
    server = viser.ViserServer()
    for steps in eval_steps:
        path = f"test_{steps-1}.npz"
        if not os.path.exists(path): continue
        pred_handler = PredictionHandler.from_npz(path)
        mesh = pred_handler.get_meshes(resolution=30)[0]
        pcs = pred_handler.get_segmented_pcs()[0]

        server.scene.add_mesh_trimesh(f"superquadrics_{steps-1}", mesh=mesh, visible=True)

    server.scene.set_up_direction([0.0, 1.0, 0.0])
    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.position = (0.8, 0.8, 0.8)
        client.camera.look_at = (0., 0., 0.)
    while True:
        time.sleep(10.0)

if __name__ == "__main__":
    main()