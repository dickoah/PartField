import gradio as gr
import os
import shutil
import tempfile
import atexit
import time
import trimesh
import subprocess
import logging
import math

# Import the new segmenter
from partfield.partfield_segmenter import PartFieldSegmenter

# Setup logging (use local file location)
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logger = logging.getLogger("partfield_app")
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(os.path.join(LOG_DIR, "app.log"))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
fh.setFormatter(formatter)
ch.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(fh)
    logger.addHandler(ch)

# Get the directory where this script is located (should be PartField root)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize segmenter globally to avoid re-checking resources on every run
segmenter = PartFieldSegmenter(script_dir=SCRIPT_DIR)

# Temporary base directory for generated runs - will be removed at process exit
TEMP_BASE_DIR = tempfile.mkdtemp(prefix="partfield_app_")

def _cleanup_temp():
    shutil.rmtree(TEMP_BASE_DIR)
atexit.register(_cleanup_temp)

def run_segmentation_and_prepare(
    mesh_file_path: str, 
    n_parts: int,
    use_agglo: bool = True,
    clustering_option: int = 0,
    with_knn: bool = False,
    is_pc: bool = False,
    single_output: bool = False,
    preprocess_model: bool = False
) -> tuple[str, str | None, list[str]]:
    """
    Run the full segmentation pipeline: inference + clustering + conversion.
    Returns: (status_message, output_glb_file, part_file_list)
    
    Args:
        mesh_file_path: Path to input mesh
        n_parts: Number of parts for clustering
        use_agglo: Use agglomerative clustering (True) or K-Means (False)
        clustering_option: 0=naive, 1=MST-based adjacency
        with_knn: Add KNN edges (for messy meshes)
        is_pc: Input is point cloud (PLY)
        single_output: Only generate one output with exactly N parts (faster)
        preprocess_model: Use preprocessed mesh for inference
    """
    print("==========================================================")
    print(f"Running segmentation on {mesh_file_path} with :\n"
          f"- n_parts={n_parts}\n"
          f"- use_agglo={use_agglo}\n"
          f"- clustering_option={clustering_option}\n"
          f"- with_knn={with_knn}\n"
          f"- is_pc={is_pc}\n"
          f"- single_output={single_output}\n"
          f"- preprocess_model={preprocess_model}")
    print("==========================================================")
    # Run process using the global segmenter instance
    result = segmenter.process(
        mesh_file_path, 
        n_parts=n_parts,
        use_agglo=use_agglo,
        clustering_option=clustering_option,
        with_knn=with_knn,
        is_pc=is_pc,
        single_output=single_output,
        enable_preprocessing=preprocess_model
    )
    
    if result['status'].startswith("Error") or not result['glb_files']:
        return result['status'], None, []
    
    # Copy files to app's temp dir to ensure persistence
    timestamp = int(time.time())
    output_dir = os.path.join(TEMP_BASE_DIR, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    saved_files = []
    for glb_file in result['glb_files']:
        filename = os.path.basename(glb_file)
        dst = os.path.join(output_dir, filename)
        shutil.copy2(glb_file, dst)
        saved_files.append(dst)
        
    return result['status'], saved_files[0] if saved_files else None, saved_files


# Build the UI
with gr.Blocks(title="PartField - simplified UI") as demo:
    gr.Markdown("## PartField Segmentation – simplified")
    with gr.Row():
        with gr.Column(scale=1):
            input_model = gr.Model3D(label="Input Mesh (.glb)")
            num_parts = gr.Slider(label="Number of parts (N)", minimum=1, maximum=100, value=10, step=1)
            
            gr.Markdown("### Clustering Settings")
            use_agglo = gr.Checkbox(label="Use Agglomerative Clustering", value=True, 
                                     info="Uncheck for K-Means clustering")
            clustering_option = gr.Radio(
                choices=[(f"Option {i}: {desc}", i) for i, desc in enumerate([
                    "Naive (simple chaining)",
                    "MST-based (better for fragmented)",
                ])],
                value=0,
                label="Adjacency Matrix Option",
                info="How to handle disconnected mesh components"
            )
            with_knn = gr.Checkbox(
                label="Use KNN edges", 
                value=False,
                info="Helps with messy mesh connectivity"
            )
            is_pc = gr.Checkbox(label="Input is Point Cloud (.ply)", value=False,
                                info="Check if input is point cloud instead of mesh")
            preprocess_model = gr.Checkbox(
                label="Preprocess Model", 
                value=False,
                info="Use preprocessed mesh for inference (may improve results for messy meshes)"
            )
            single_output = gr.Checkbox(
                label="Single output mode", 
                value=True,
                info="Only generate one segmentation with exactly N parts (faster). Uncheck to get all segmentations from 2 to N parts."
            )
            
            generate_btn = gr.Button("Generate", variant="primary")
            status_box = gr.Textbox(label="Status / Output", lines=10, interactive=False)
        with gr.Column(scale=1):
            output_model = gr.Model3D(label="Segmentation Output")
            parts_slider = gr.Slider(label="Select part to view/download", minimum=1, maximum=100, value=1, step=1)
            # Hidden state to hold the list of GLB files
            files_state = gr.State([])

    # Wire events
    generate_btn.click(
        fn=run_segmentation_and_prepare,
        inputs=[input_model, num_parts, use_agglo, clustering_option, with_knn, is_pc, single_output, preprocess_model],
        outputs=[status_box, output_model, files_state],
    )

    # When files_state changes, update slider max and show first part
    def update_slider_and_model(file_list):
        if not isinstance(file_list, (list, tuple)) or not file_list:
            # Use a large maximum to avoid validation errors if client sends an old value
            return None, gr.update(minimum=1, maximum=100, value=1)
        max_parts = len(file_list)
        return file_list[0], gr.update(minimum=1, maximum=max_parts, value=1)

    files_state.change(
        fn=update_slider_and_model,
        inputs=[files_state],
        outputs=[output_model, parts_slider]
    )

    # When slider changes, show the saelected part
    def slider_to_model(slider_val, file_list):
        if not isinstance(file_list, (list, tuple)) or not file_list:
            return None
        try:
            idx = int(slider_val) - 1
        except (TypeError, ValueError):
            return None
        if idx < 0 or idx >= len(file_list):
            return None
        return file_list[idx]

    parts_slider.change(
        fn=slider_to_model,
        inputs=[parts_slider, files_state],
        outputs=output_model
    )

    # Reset interface when input model changes
    def reset_on_change():
        return None, []

    input_model.change(
        fn=reset_on_change,
        inputs=[],
        outputs=[output_model, files_state]
    )


if __name__ == "__main__":
    demo.launch()
