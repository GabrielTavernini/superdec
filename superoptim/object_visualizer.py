import numpy as np
import os
import viser
import time
import argparse
from superdec.utils.predictions_handler_extended import PredictionHandler 
from superdec.utils.visualizations import generate_ncolors
import trimesh

RESOLUTION = 30

def main(modes: list, resolution: int = RESOLUTION) -> None:
  server = viser.ViserServer()

  modes_input = list(modes)
  paths = ["data/output_npz/shapenet_test.npz"]
  modes_used = ['None']
  for m in modes_input:
    p = os.path.join("data", "output_npz", f"shapenet_test_tables_optimized_{m}.npz")
    if os.path.exists(p):
      paths.append(p)
      modes_used.append(m)
    elif os.path.exists(m):
      paths.append(m)
      modes_used.append(m)
    else:
      print(f"Warning: file for mode '{m}' not found at {p} and not provided as path; skipping")

  if len(paths) == 0:
    raise ValueError(f"No valid files found for modes {modes_input}")

  print(f"Opening npz files: {paths}")
  handlers = [PredictionHandler.from_npz(p) for p in paths]

  # compute maximum number of models across the two files to populate index dropdown
  max_models = max(len(h.names) for h in handlers)

  server.scene.set_up_direction([0.0, 1.0, 0.0])

  file_labels = modes_used
  file_dropdown = server.gui.add_dropdown("Mode", file_labels, initial_value=file_labels[0])
  model_dropdown = server.gui.add_dropdown("Model index", [str(i) for i in range(max_models)], initial_value='0')

  def draw_current():
    # map selected file label to its index
    try:
      active_idx = file_labels.index(file_dropdown.value)
    except Exception:
      active_idx = 0
    handler = handlers[active_idx]

    try:
      model_idx = int(model_dropdown.value)
    except Exception:
      model_idx = 0

    # clamp index to available models in the active file
    if len(handler.names) == 0:
      print(f"No models in selected file {file_labels[active_idx]}")
      return
    if model_idx >= len(handler.names):
      model_idx = len(handler.names) - 1
      model_dropdown.value = str(model_idx)

    # generate mesh on demand for only the selected model
    print(f"Generating mesh for file {active_idx} model {model_idx} (resolution={resolution})")
    try:
      mesh = handler.get_mesh(model_idx, resolution=resolution, colors=True)
    except Exception as e:
      print(f"Failed to generate mesh: {e}")
      mesh = None

    if mesh is not None:
      server.scene.add_mesh_trimesh("superquadrics", mesh=mesh, visible=True)

    # segmented point cloud for the selected model
    try:
      pc = handler.get_segmented_pc(model_idx)
      points = np.asarray(pc.points)
      colors = np.asarray(pc.colors)
      server.scene.add_point_cloud(name="/segmented_pointcloud", points=points, colors=colors, point_size=0.005)
    except Exception as e:
      print(f"Failed to add point cloud: {e}")

  # attach callbacks
  file_dropdown.on_update(lambda _: draw_current())
  model_dropdown.on_update(lambda _: draw_current())

  draw_current()

  @server.on_client_connect
  def _(client: viser.ClientHandle) -> None:
    client.camera.position = (0.8, 0.8, 0.8)
    client.camera.look_at = (0., 0., 0.)

  while True:
      time.sleep(10.0)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Visualize multiple superdec NPZ files and toggle between them")
  parser.add_argument("files", nargs='+', help="Paths to one or more npz files")
  parser.add_argument("--resolution", type=int, default=RESOLUTION, help="Mesh resolution for superquadrics")
  args = parser.parse_args()
  main(args.files, resolution=args.resolution)