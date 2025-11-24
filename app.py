import gradio as gr
import os
import shutil
import tempfile
import atexit
import time
import trimesh
from pathlib import Path
import subprocess
import logging
import math

# Import clustering and inference functions directly
from run_part_clustering import solve_clustering
from partfield_inference import predict
from partfield.config import default_argument_parser, setup

# Try to import torch and GPU utilities for memory analysis
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

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

# Temporary base directory for generated runs - will be removed at process exit
TEMP_BASE_DIR = tempfile.mkdtemp(prefix="partfield_app_")


def _cleanup_temp():
    shutil.rmtree(TEMP_BASE_DIR)

atexit.register(_cleanup_temp)


def log_gpu_memory():
    """Log GPU memory usage for debugging memory issues."""
    if not TORCH_AVAILABLE:
        logger.debug("PyTorch not available; skipping GPU memory logging")
        return
    
    if not torch.cuda.is_available():
        logger.info("CUDA not available")
        return
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        total = props.total_memory / 1e9
        free = total - (reserved)
        logger.info(
            "GPU %d (%s): Total=%.2f GB, Allocated=%.2f GB, Reserved=%.2f GB, Free=%.2f GB",
            i, props.name, total, allocated, reserved, free
        )


def clear_gpu_memory():
    """Release cached GPU memory when possible."""
    if not TORCH_AVAILABLE:
        logger.debug("PyTorch not available; skipping GPU memory cleanup")
        return

    if not torch.cuda.is_available():
        logger.debug("CUDA not available; skipping GPU memory cleanup")
        return

    torch.cuda.empty_cache()
    if hasattr(torch.cuda, "ipc_collect"):
        torch.cuda.ipc_collect()
    logger.info("Cleared CUDA cached memory")


def load_model(file):
    """Load a model file and copy it to data directory in a subfolder named after the model (without extension)"""
    if file is None or not hasattr(file, "name"):
        return None, "Please upload a valid model file."

    src_path = getattr(file, "name", "")
    if not src_path or not os.path.isfile(src_path):
        return None, f"Invalid file path: {src_path}"

    original_filename = os.path.basename(src_path)
    model_name = Path(original_filename).stem
    
    dest_dir = os.path.join(SCRIPT_DIR, "data", model_name)
    os.makedirs(dest_dir, exist_ok=True)
    
    dest_path = os.path.join(dest_dir, original_filename)
    shutil.copy2(src_path, dest_path)
    
    return model_name, f"Model '{original_filename}' loaded successfully!\nSaved to: {dest_path}"


def convert_ply_to_glb(ply_path, glb_dir):
    """Convert a PLY file to GLB format with caching"""
    os.makedirs(glb_dir, exist_ok=True)
    
    ply_filename = os.path.basename(ply_path)
    glb_filename = os.path.splitext(ply_filename)[0] + ".glb"
    glb_path = os.path.abspath(os.path.join(glb_dir, glb_filename))
    
    # Use cache if GLB exists and is newer than PLY
    if os.path.exists(glb_path):
        if os.path.getmtime(glb_path) >= os.path.getmtime(ply_path):
            return glb_path
    
    # Convert PLY to GLB
    mesh = trimesh.load(ply_path, process=False)
    mesh.export(glb_path)
    return glb_path


def convert_all_ply_to_glb(model_name):
    """Convert all PLY files to GLB and return list of GLB files"""
    if not model_name:
        return []
    
    ply_dir = os.path.join(SCRIPT_DIR, "exp_results", "clustering", model_name, "ply")
    glb_dir = os.path.join(SCRIPT_DIR, "exp_results", "clustering", model_name, "glb")
    
    if not os.path.isdir(ply_dir):
        return []

    filenames = [f for f in sorted(os.listdir(ply_dir)) if f.endswith(".ply")]
    return [
        (os.path.basename(glb_path), glb_path)
        for filename in filenames
        for ply_path in [os.path.abspath(os.path.join(ply_dir, filename))]
        if os.path.isfile(ply_path)
        for glb_path in [convert_ply_to_glb(ply_path, glb_dir)]
        if glb_path
    ]


def upload_partfield_model():
    """Ensure a local copy of the PartField checkpoint exists in `model/`.

    If no .ckpt files are present in `model/`, this will run a `git clone`
    of the Hugging Face repo directly into the `model/` directory with
    progress logging.
    """
    model_dir = os.path.join(SCRIPT_DIR, "model")
    os.makedirs(model_dir, exist_ok=True)

    # Look for any .ckpt files already present
    ckpt_files = [p for p in os.listdir(model_dir) if p.endswith(".ckpt")]
    if ckpt_files:
        logger.info("Found existing checkpoint files in %s: %s", model_dir, ckpt_files)
        return True

    # Clone the repo directly into model/
    clone_url = "https://huggingface.co/mikaelaangel/partfield-ckpt"
    logger.info("No local checkpoint found — starting git clone of %s into %s", clone_url, model_dir)
    
    cmd = ["git", "clone", clone_url, model_dir]
    logger.info("Executing command: %s", " ".join(cmd))
    
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=SCRIPT_DIR)
        
        # Stream clone output for progress logging
        for line in iter(proc.stdout.readline, ""):
            if line:
                logger.info("git clone progress: %s", line.rstrip())
        
        proc.wait()
        returncode = proc.returncode
        
        logger.info("git clone completed with returncode=%s", returncode)
        if returncode != 0:
            logger.error("Failed to clone checkpoint repository")
            return False
    except FileNotFoundError:
        logger.error("git is not installed or not found in PATH; cannot clone checkpoint repository")
        return False
    except Exception as e:
        logger.error("Error during git clone: %s", e)
        return False

    # Verify that .ckpt files now exist
    ckpt_files = [p for p in os.listdir(model_dir) if p.endswith(".ckpt")]
    if ckpt_files:
        logger.info("Successfully downloaded checkpoint files: %s", ckpt_files)
        return True
    else:
        logger.warning("Cloned repo but no .ckpt files found in %s", model_dir)
        return False


def combine_glbs_into_cumulative(glb_paths, out_dir, n):
    """
    Create n GLB files where file k contains the union of the first k parts.

    glb_paths: list of absolute file paths for individual part GLBs (ordered)
    out_dir: directory to write cumulative GLBs
    n: number of cumulative files to create (<= len(glb_paths))
    Returns list of generated file paths.
    """
    if not glb_paths:
        logger.warning("No GLB paths provided for cumulative export")
        return []

    os.makedirs(out_dir, exist_ok=True)
    generated = []
    total = min(max(n, 0), len(glb_paths))
    if total == 0:
        return []

    # Pre-load meshes to avoid repeated disk reads
    def _to_scene(path: str) -> trimesh.Scene:
        obj = trimesh.load(path, force='scene')
        return trimesh.Scene(obj) if isinstance(obj, trimesh.Trimesh) else obj

    loaded = [_to_scene(p) for p in glb_paths]

    for k in range(1, total + 1):
        scene = trimesh.Scene()
        for i in range(k):
            s = loaded[i]
            if s is None:
                continue
            # Add each geometry from the scene
            for name, geom in s.geometry.items():
                # Ensure unique names
                scene.add_geometry(geom.copy(), node_name=f"part_{i}_{name}")

        out_path = os.path.join(out_dir, f"parts_1_to_{k}.glb")
        # Export scene as glb
        scene.export(out_path)
        generated.append(out_path)

    return generated


def _get_mesh_path(mesh_file_path) -> str:
    """Extract actual file path from various input formats."""
    if mesh_file_path is None:
        raise ValueError("Mesh path not provided")

    if isinstance(mesh_file_path, str):
        candidate = mesh_file_path
    elif isinstance(mesh_file_path, dict) and "name" in mesh_file_path:
        candidate = mesh_file_path["name"]
    elif hasattr(mesh_file_path, "name"):
        candidate = mesh_file_path.name
    else:
        candidate = str(mesh_file_path)

    if not candidate:
        raise ValueError("Empty mesh path value")

    if not os.path.isfile(candidate):
        raise FileNotFoundError(f"Mesh file not found: {candidate}")

    return candidate


def _convert_ply_dir_to_glb(ply_dir: str, glb_dir: str) -> list[str]:
    """Convert all PLY files in a directory to GLB format."""
    if not os.path.isdir(ply_dir):
        logger.warning("PLY directory missing: %s", ply_dir)
        return []

    os.makedirs(glb_dir, exist_ok=True)
    logger.info("Converting PLY files from %s to GLB in %s", ply_dir, glb_dir)

    glb_files = [
        glb_path
        for filename in sorted(os.listdir(ply_dir))
        if filename.endswith(".ply")
        for ply_path in [os.path.abspath(os.path.join(ply_dir, filename))]
        if os.path.isfile(ply_path)
        for glb_path in [convert_ply_to_glb(ply_path, glb_dir)]
        if glb_path
    ]

    logger.info("Converted %d PLY files to GLB", len(glb_files))
    return glb_files


def _preprocess_mesh(mesh_path: str, output_dir: str) -> str:
    """
    Preprocess a mesh by cleaning up vertices and removing duplicates.
    
    Args:
        mesh_path: Path to input mesh (.glb, .obj, etc.)
        output_dir: Directory to save preprocessed mesh
        
    Returns:
        Path to preprocessed mesh
    """
    if not mesh_path or not os.path.isfile(mesh_path):
        raise FileNotFoundError(f"Cannot preprocess missing mesh: {mesh_path}")

    logger.info("=== Starting mesh preprocessing ===")
    logger.info("Input mesh: %s", mesh_path)
    
    threshold = 1e-6

    def _clean_trimesh_mesh(original: trimesh.Trimesh) -> trimesh.Trimesh:
        if original.vertices.size == 0:
            raise ValueError("Mesh has no vertices")
        if original.faces.size == 0:
            raise ValueError("Mesh has no faces")
        if threshold <= 0:
            raise ValueError(f"Threshold must be positive, got {threshold}")
        if threshold > 1.0:
            raise ValueError(f"Threshold too large ({threshold}); expected < 1.0")

        digits_vertex = max(1, int(-math.log10(threshold)))
        processed = original.copy()

        logger.debug("Cleaning mesh '%s': %d vertices, %d faces before", getattr(original, 'metadata', {}).get('name', 'mesh'), len(processed.vertices), len(processed.faces))

        processed.merge_vertices(
            merge_tex=True,
            merge_norm=True,
            digits_vertex=digits_vertex,
            digits_norm=max(1, digits_vertex - 1),
            digits_uv=max(1, digits_vertex - 1),
        )

        processed.update_faces(processed.nondegenerate_faces())
        processed.update_faces(processed.unique_faces())
        processed.remove_unreferenced_vertices()

        # Refresh caches so normals/UVs stay consistent
        processed._cache.clear()
        if hasattr(processed, "vertex_normals"):
            _ = processed.vertex_normals

        logger.debug("Cleaned mesh now has %d vertices, %d faces", len(processed.vertices), len(processed.faces))
        return processed

    loaded = trimesh.load(mesh_path, force='scene')
    scene = loaded if isinstance(loaded, trimesh.Scene) else trimesh.Scene(loaded)

    if not scene.geometry:
        raise ValueError("No valid geometries found in the input scene")

    scene_items = list(scene.geometry.items())
    trimesh_items = {name: geom for name, geom in scene_items if isinstance(geom, trimesh.Trimesh)}
    if not trimesh_items:
        raise ValueError("No Trimesh geometries available for preprocessing")

    total_vertices_before = sum(len(geom.vertices) for geom in trimesh_items.values())
    total_faces_before = sum(len(geom.faces) for geom in trimesh_items.values())
    logger.info("Before preprocessing: %d vertices, %d faces", total_vertices_before, total_faces_before)

    cleaned_geoms = {name: _clean_trimesh_mesh(geom) for name, geom in trimesh_items.items()}

    cleaned_meshes = list(cleaned_geoms.values())

    combined_mesh = cleaned_meshes[0] if len(cleaned_meshes) == 1 else trimesh.util.concatenate(tuple(cleaned_meshes))

    total_vertices_after = len(combined_mesh.vertices)
    total_faces_after = len(combined_mesh.faces)
    logger.info("After preprocessing: %d vertices, %d faces", total_vertices_after, total_faces_after)
    logger.info(
        "Removed %d vertices, %d faces",
        total_vertices_before - total_vertices_after,
        total_faces_before - total_faces_after,
    )

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "mesh_preprocessed.ply")
    combined_mesh.export(output_path)
    logger.info("Exported preprocessed mesh to %s", output_path)

    return output_path


def run_segmentation_and_prepare(
    mesh_file_path: str, 
    n_parts: int,
    use_agglo: bool = True,
    clustering_option: int = 0,
    with_knn: bool = False,
    is_pc: bool = False
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
    """
    logger.info("=== Starting segmentation pipeline ===")
    logger.info("Mesh file: %s, Parts: %d, Agglo: %s, Option: %d, KNN: %s", 
                mesh_file_path, n_parts, use_agglo, clustering_option, with_knn)
    log_gpu_memory()

    if n_parts < 1:
        msg = "Number of parts must be at least 1"
        logger.error(msg)
        return msg, None, []

    try:
        mesh_path = _get_mesh_path(mesh_file_path)
    except (ValueError, FileNotFoundError) as err:
        logger.error("Invalid mesh input: %s", err)
        return str(err), None, []

    try:
        # Determine config file
        mesh_lower = os.path.basename(mesh_path).lower()
        config_name = "correspondence_demo.yaml" if "correspond" in mesh_lower else "demo.yaml"
        config_path = os.path.join(SCRIPT_DIR, "configs", "final", config_name)
        if not os.path.exists(config_path):
            msg = f"Config not found: {config_path}"
            logger.error(msg)
            return msg, None, []

        # Set up unique result directory in temp folder
        timestamp = int(time.time())
        model_basename = os.path.basename(mesh_path).split('.')[0]
        result_name = f"results_{timestamp}_{model_basename}"
        
        # Use temp directory for all generated files
        temp_result_dir = os.path.join(TEMP_BASE_DIR, result_name)
        model_dir = os.path.join(temp_result_dir, "data")
        os.makedirs(model_dir, exist_ok=True)

        # Copy input mesh
        mesh_filename = os.path.basename(mesh_path)
        target_mesh_path = os.path.join(model_dir, mesh_filename)
        shutil.copy2(mesh_path, target_mesh_path)
        logger.info("Copied input mesh to %s", target_mesh_path)
        
        # Preprocess the mesh
        logger.info("=== Mesh Preprocessing ===")
        preprocess_dir = os.path.join(temp_result_dir, "preprocessing")
        os.makedirs(preprocess_dir, exist_ok=True)
        try:
            preprocessed_mesh_path = _preprocess_mesh(target_mesh_path, preprocess_dir)
            # Use preprocessed mesh for the pipeline
            target_mesh_path = preprocessed_mesh_path
            logger.info("Using preprocessed mesh for pipeline")
        except Exception as e:
            logger.warning("Mesh preprocessing failed: %s. Continuing with original mesh.", e)
            # Continue with original mesh if preprocessing fails

        # Verify checkpoint exists
        checkpoint_path = os.path.join(SCRIPT_DIR, "model", "model_objaverse.ckpt")
        if not os.path.exists(checkpoint_path):
            msg = f"Checkpoint not found: {checkpoint_path}"
            logger.error(msg)
            return msg, None, []

        # Run inference directly (not via subprocess)
        log_gpu_memory()
        
        # Create temp features dir
        features_dir = os.path.join(temp_result_dir, "features")
        os.makedirs(features_dir, exist_ok=True)
        
        logger.info("Running inference directly with config: %s", config_path)
        try:
            # Build args and config for inference
            parser = default_argument_parser()
            args = parser.parse_args([
                "-c", config_path,
                "--opts",
                "continue_ckpt", checkpoint_path,
                "result_name", result_name,
                "dataset.data_path", model_dir
            ])
            cfg = setup(args, freeze=False)
            
            # Run prediction directly
            predict(cfg)
            logger.info("Inference completed successfully")
            clear_gpu_memory()
            
        except Exception as e:
            msg = f"Inference failed: {str(e)}"
            logger.error(msg)
            logger.exception(e)
            return msg, None, []

        # Verify features were generated in exp_results
        actual_features_dir = os.path.join(SCRIPT_DIR, "exp_results", result_name)
        if not os.path.exists(actual_features_dir):
            msg = f"Inference output not found: {actual_features_dir}"
            logger.error(msg)
            return msg, None, []
        
        # Copy features to temp dir
        for fname in os.listdir(actual_features_dir):
            src = os.path.join(actual_features_dir, fname)
            dst = os.path.join(features_dir, fname)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
            else:
                shutil.copytree(src, dst, dirs_exist_ok=True)
        
        # Clean up the exp_results copy
        shutil.rmtree(actual_features_dir, ignore_errors=True)
        logger.info("Cleaned up exp_results/%s", result_name)

        # Create clustering output directory (in temp)
        clustering_output_dir = os.path.join(temp_result_dir, "clustering")
        os.makedirs(clustering_output_dir, exist_ok=True)
        
        # Create subdirectories that solve_clustering expects
        os.makedirs(os.path.join(clustering_output_dir, "ply"), exist_ok=True)
        os.makedirs(os.path.join(clustering_output_dir, "cluster_out"), exist_ok=True)

        # Run clustering directly (not as subprocess)
        log_gpu_memory()
        logger.info("Running clustering directly (use_agglo=%s, option=%d, with_knn=%s, is_pc=%s)", 
                    use_agglo, clustering_option, with_knn, is_pc)
        
        try:
            # Get list of input files to process
            input_files = [
                f for f in os.listdir(model_dir)
                if (f.endswith(".ply") and is_pc) or (f.endswith((".obj", ".glb")) and not is_pc)
            ]

            if not input_files:
                raise FileNotFoundError("No compatible input files found for clustering")

            for input_file in input_files:
                input_fname = os.path.join(model_dir, input_file)
                uid = input_file.split('.')[0]
                view_id = 0
                
                logger.info("Processing: %s (uid=%s)", input_file, uid)
                
                # Call solve_clustering directly
                solve_clustering(
                    input_fname=input_fname,
                    uid=uid,
                    view_id=view_id,
                    save_dir=features_dir,
                    out_render_fol=clustering_output_dir,
                    use_agglo=use_agglo,
                    max_num_clusters=n_parts,
                    is_pc=is_pc,
                    option=clustering_option,
                    with_knn=with_knn,
                    export_mesh=True
                )
                
            clear_gpu_memory()

        except Exception as e:
            msg = f"Clustering failed: {str(e)}"
            logger.error(msg)
            logger.exception(e)
            return msg, None, []

        # Convert PLY to GLB (all in temp)
        ply_dir = os.path.join(clustering_output_dir, "ply")
        if not os.path.isdir(ply_dir):
            msg = f"PLY output directory not found: {ply_dir}"
            logger.warning(msg)
            return msg, None, []

        glb_dir = os.path.join(clustering_output_dir, "glbs")
        glb_files = _convert_ply_dir_to_glb(ply_dir, glb_dir)

        logger.info("Temp directory: %s", temp_result_dir)

        if glb_files:
            msg = f"✓ Segmentation complete! {n_parts} part(s) segmented.\nSelect part from slider to view and download."
            first_model = glb_files[0] if glb_files else None
            return msg, first_model, glb_files
        else:
            return "No parts generated", None, []

    except Exception as e:
        logger.exception("Unexpected error during segmentation: %s", e)
        return f"Error: {e}", None, []


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
                value=1,
                label="Adjacency Matrix Option",
                info="How to handle disconnected mesh components"
            )
            with_knn = gr.Checkbox(
                label="Use KNN edges", 
                value=True,
                info="Helps with messy mesh connectivity"
            )
            is_pc = gr.Checkbox(label="Input is Point Cloud (.ply)", value=False,
                                info="Check if input is point cloud instead of mesh")
            
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
        inputs=[input_model, num_parts, use_agglo, clustering_option, with_knn, is_pc],
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

    # When slider changes, show the selected part
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
    ok = upload_partfield_model()
    if not ok:
        logger.warning("PartField model not available locally. The UI will still start but segmentation will fail until a checkpoint is provided.")

    demo.launch()
