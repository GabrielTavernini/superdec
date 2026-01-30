import time
import numpy as np
import viser
from scipy.spatial.transform import Rotation as R
from superdec.utils.predictions_handler_extended import PredictionHandler

def main():
    resolution = 30 
    pred_path = "data/output_npz/objects/rect_table.npz"
    
    # 1. Load Data
    pred_handler = PredictionHandler.from_npz(pred_path)
    B_IDX = 0
    
    # Flag to prevent sliders from overwriting data during initialization
    is_loading = False

    # Get static point cloud context
    pcs = pred_handler.get_segmented_pcs()[B_IDX]

    # 2. Setup Viser
    server = viser.ViserServer()
    server.scene.set_up_direction([0.0, 1.0, 0.0])

    server.scene.add_point_cloud(
        name="/segmented_pointcloud",
        points=np.array(pcs.points),
        colors=np.array(pcs.colors),
        point_size=0.005,
    )

    # 3. GUI Setup
    existing_indices = np.where(pred_handler.exist[B_IDX] > 0.5)[0]
    
    with server.gui.add_folder("Selection"):
        selector = server.gui.add_dropdown(
            "Edit Primitive",
            options=[str(i) for i in existing_indices],
            initial_value=str(existing_indices[0])
        )

    gui_controls = {}
    with server.gui.add_folder("Primitive Parameters"):
        gui_controls['tx'] = server.gui.add_slider("Trans X", -1.0, 1.0, 0.005, 0.0)
        gui_controls['ty'] = server.gui.add_slider("Trans Y", -1.0, 1.0, 0.005, 0.0)
        gui_controls['tz'] = server.gui.add_slider("Trans Z", -1.0, 1.0, 0.005, 0.0)
        
        gui_controls['sx'] = server.gui.add_slider("Scale X", 0.001, 0.5, 0.001, 0.1)
        gui_controls['sy'] = server.gui.add_slider("Scale Y", 0.001, 0.5, 0.001, 0.1)
        gui_controls['sz'] = server.gui.add_slider("Scale Z", 0.001, 0.5, 0.001, 0.1)

        gui_controls['rx'] = server.gui.add_slider("Rot X (deg)", -180, 180, 1, 0)
        gui_controls['ry'] = server.gui.add_slider("Rot Y (deg)", -180, 180, 1, 0)
        gui_controls['rz'] = server.gui.add_slider("Rot Z (deg)", -180, 180, 1, 0)

        gui_controls['e1'] = server.gui.add_slider("Eps 1", 0.01, 2.0, 0.01, 1.0)
        gui_controls['e2'] = server.gui.add_slider("Eps 2", 0.01, 2.0, 0.01, 1.0)

        gui_controls['k1'] = server.gui.add_slider("K 1", -1, 1.0, 0.01, 0.0)
        gui_controls['k2'] = server.gui.add_slider("K 2", -1, 1.0, 0.01, 0.0)

    # 4. Functions
    def render_all():
        """Regenerates and displays the mesh."""
        full_mesh = pred_handler.get_meshes(resolution=resolution)[B_IDX]
        server.scene.add_mesh_trimesh(
            name="/superquadrics/batch_0",
            mesh=full_mesh,
        )
        
        # Add a visual gizmo at the selected primitive's center
        p_idx = int(selector.value)
        server.scene.add_frame(
            name="/selected_gizmo",
            axes_length=0.1,
            axes_radius=0.005,
            position=pred_handler.translation[B_IDX, p_idx]
        )

    def update_gui_from_handler():
        """Reads data from the handler and forces it into the sliders."""
        nonlocal is_loading
        is_loading = True  # Block the on_update callback
        
        p_idx = int(selector.value)
        t = pred_handler.translation[B_IDX, p_idx]
        s = pred_handler.scale[B_IDX, p_idx]
        e = pred_handler.exponents[B_IDX, p_idx]
        r = R.from_matrix(pred_handler.rotation[B_IDX, p_idx]).as_euler('xyz', degrees=True)
        k = pred_handler.tapering[B_IDX, p_idx]

        gui_controls['tx'].value = float(t[0])
        gui_controls['ty'].value = float(t[1])
        gui_controls['tz'].value = float(t[2])
        gui_controls['sx'].value = float(s[0])
        gui_controls['sy'].value = float(s[1])
        gui_controls['sz'].value = float(s[2])
        gui_controls['rx'].value = float(r[0])
        gui_controls['ry'].value = float(r[1])
        gui_controls['rz'].value = float(r[2])
        gui_controls['e1'].value = float(e[0])
        gui_controls['e2'].value = float(e[1])
        gui_controls['k1'].value = float(k[0])
        gui_controls['k2'].value = float(k[1])
        
        is_loading = False # Re-enable sync
        render_all()

    def sync_handler_and_render(_):
        """Writes slider values back to the handler and renders."""
        if is_loading:
            return
            
        p_idx = int(selector.value)
        
        # Update handler
        pred_handler.translation[B_IDX, p_idx] = [gui_controls['tx'].value, gui_controls['ty'].value, gui_controls['tz'].value]
        pred_handler.scale[B_IDX, p_idx] = [gui_controls['sx'].value, gui_controls['sy'].value, gui_controls['sz'].value]
        pred_handler.exponents[B_IDX, p_idx] = [gui_controls['e1'].value, gui_controls['e2'].value]
        
        rot_obj = R.from_euler('xyz', [gui_controls['rx'].value, gui_controls['ry'].value, gui_controls['rz'].value], degrees=True)
        pred_handler.rotation[B_IDX, p_idx] = rot_obj.as_matrix()

        pred_handler.tapering[B_IDX, p_idx] = [gui_controls['k1'].value, gui_controls['k2'].value]

        render_all()

    # 5. Bindings
    selector.on_update(lambda _: update_gui_from_handler())
    
    for ctrl in gui_controls.values():
        ctrl.on_update(sync_handler_and_render)

    # Initial Draw
    update_gui_from_handler()

    while True:
        time.sleep(1.0)

if __name__ == "__main__":
    main()