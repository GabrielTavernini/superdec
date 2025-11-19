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
    paths = [
        "test.npz",
        "results/chair/superq/test_299.npz"
        "results/chair/superq/test_699.npz"
        "results/chair/superq/test_2999.npz"
    ]
    server = viser.ViserServer()
    for path in paths:
        if not os.path.exists(path):
            continue

        pred_handler = PredictionHandler.from_npz(path)

        # Parent group/directory for this step
        for i, m in enumerate(pred_handler.get_meshes(resolution=30)):
            server.scene.add_mesh_trimesh(
                f"{path}/mesh_{i}",
                mesh=m,
                visible=True,
            )

    server.scene.set_up_direction([0.0, 0.0, 1.0])
    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.position = (0.8, 0.8, 0.8)
        client.camera.look_at = (0., 0., 0.)
    while True:
        time.sleep(10.0)

if __name__ == "__main__":
    main()