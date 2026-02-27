"""
PartField Mesh Segmentation Module

A modular interface for semantic mesh segmentation using PartField features.
Can be used as a standalone module or integrated into other projects.

Example:
    from partfield_segmenter import PartFieldSegmenter
    
    segmenter = PartFieldSegmenter(
        n_parts=10,
        use_agglo=True,
        single_output=True
    )
    result = segmenter.process(mesh_path)
"""

import os
import shutil
import tempfile
import logging
import math
import time
import subprocess
from datetime import datetime
from typing import Union, Optional, Literal, List, Dict, Any
from pathlib import Path
from collections import Counter, defaultdict

import trimesh
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering

# Try to import torch and GPU utilities for memory analysis
try:
    import torch
    from lightning.pytorch import seed_everything, Trainer
    from lightning.pytorch.callbacks import ModelCheckpoint
    
    # Set optimal precision for GPUs with Tensor Cores
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')
    
    # Register safe globals for PyTorch 2.6+ weights_only loading
    try:
        import yacs.config
        torch.serialization.add_safe_globals([yacs.config.CfgNode])
    except ImportError:
        pass
        
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Import inference and clustering functions
from .partfield_clusterer import PartFieldClusterer
from .config.defaults import _C as base_cfg
from .model_trainer_pvcnn_only_demo import Model
from .utils import count_unique_colors, load_mesh_util

# Setup logging
logger = logging.getLogger("partfield_segmenter")


class PartFieldSegmenter:
    """
    PartField Mesh Segmentation Engine.
    
    Segments 3D meshes into semantic parts using learned PartField features.
    Includes inference, clustering, and mesh export.
    
    Attributes:
        script_dir: Root directory of the PartField package.
        checkpoint_path: Path to the model checkpoint file.
        config_path: Path to the YAML configuration file.
        temp_dirs: List of temporary directories created by ``process()``.
    """
    
    # ======== Main methods ========
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        config_path: Optional[str] = None,
        script_dir: Optional[str] = None
    ):
        """
        Initialize the PartField segmenter.
        
        Args:
            checkpoint_path: Path to model checkpoint (auto-detected if None)
            config_path: Path to config file (auto-detected if None)
            script_dir: Script directory (auto-detected if None)
        """

        # Paths
        self.script_dir = script_dir or os.path.dirname(os.path.abspath(__file__))
        self.checkpoint_path = checkpoint_path or os.path.join(
            self.script_dir, "model", "model_objaverse.ckpt"
        )
        self.config_path = config_path or os.path.join(
            self.script_dir, "configs", "final", "demo.yaml"
        )
        
        # Temp directories (one per process() call)
        self.temp_dirs: List[str] = []
        
        # Download model if needed, then validate all resources exist
        self._download_model()
        self._check_resources()
        
        logger.info("[PartFieldSegmenter] Initialized")

    def process(
        self,
        mesh: Union[trimesh.Trimesh, trimesh.Scene, str, Path],
        n_parts: int = 10,
        use_agglo: bool = True,
        clustering_option: int = 1,
        with_knn: bool = True,
        is_pc: bool = False,
        single_output: bool = True,
        enable_preprocessing: bool = True,
        preprocessing_threshold: float = 1e-6,
    ) -> Dict[str, Any]:
        """
        Segment a mesh into semantic parts.
        
        Args:
            mesh: Input mesh (Trimesh, Scene, or file path)
            n_parts: Number of parts for clustering (default: 10)
            use_agglo: Use agglomerative clustering (default: True) vs K-Means
            clustering_option: 0=naive, 1=MST-based (default), 2=CC-MST
            with_knn: Add KNN edges (default: True)
            is_pc: Input is point cloud (default: False)
            single_output: Only output final N-part segmentation (default: True)
            enable_preprocessing: Enable mesh preprocessing (default: True)
            preprocessing_threshold: Vertex dedup threshold (default: 1e-6)
        
        Returns:
            dict with keys:
                - 'status': Success message
                - 'glb_files': List of output GLB file paths
                - 'scene': Trimesh Scene object of the result
                - 'temp_dir': Temporary directory path (will be cleaned up on exit)
                - 'n_files': Number of output files generated
                - 'n_parts': Number of parts requested
        """
        if n_parts < 1:
            raise ValueError("Number of parts must be at least 1")

        logger.info("=== Starting segmentation pipeline ===")
        logger.info("Parts: %d, Agglo: %s, Option: %d, KNN: %s, Single: %s", 
                    n_parts, use_agglo, clustering_option, with_knn, single_output)

        # Create temp directory for this run
        temp_dir = tempfile.mkdtemp(prefix="partfield_seg_")
        self.temp_dirs.append(temp_dir)
        
        try:
            # Set up unique result directory in temp folder
            timestamp = int(time.time())
            
            # Determine basename
            if isinstance(mesh, (str, Path)):
                model_basename = os.path.basename(str(mesh)).split('.')[0]
            else:
                model_basename = "mesh_object"
                
            result_name = f"results_{timestamp}_{model_basename}"
            
            temp_result_dir = os.path.join(temp_dir, result_name)
            model_dir = os.path.join(temp_result_dir, "data")
            os.makedirs(model_dir, exist_ok=True)

            # Handle input mesh: copy or export to model_dir
            target_mesh_path = ""
            if isinstance(mesh, (str, Path)):
                mesh_path = str(mesh)
                if not os.path.isfile(mesh_path):
                    raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
                
                mesh_filename = os.path.basename(mesh_path)
                target_mesh_path = os.path.join(model_dir, mesh_filename)
                shutil.copy2(mesh_path, target_mesh_path)
                logger.info("Copied input mesh to %s", target_mesh_path)
            elif isinstance(mesh, (trimesh.Trimesh, trimesh.Scene)):
                target_mesh_path = os.path.join(model_dir, "input_mesh.glb")
                mesh.export(target_mesh_path)
                logger.info("Exported input mesh object to %s", target_mesh_path)
            else:
                raise ValueError(f"Unsupported input type: {type(mesh)}")
            
            # Preprocess the mesh if enabled
            if enable_preprocessing:
                logger.info("=== Mesh Preprocessing ===")
                preprocess_dir = os.path.join(temp_result_dir, "preprocessing")
                os.makedirs(preprocess_dir, exist_ok=True)
                try:
                    preprocessed_mesh_path = self._preprocess_mesh(target_mesh_path, preprocess_dir, preprocessing_threshold, is_pc=is_pc)
                    logger.info("Using preprocessed mesh for inference: %s", preprocessed_mesh_path)
                    model_dir = preprocess_dir
                except Exception as e:
                    logger.warning("Mesh preprocessing failed: %s. Continuing with original mesh.", e)
            else:
                logger.info("Preprocessing disabled. Using original mesh for inference.")

            # Run inference directly            
            # Create temp features dir
            features_dir = os.path.join(temp_result_dir, "features")
            os.makedirs(features_dir, exist_ok=True)
            
            logger.info("Running inference directly with config: %s", self.config_path)
            try:
                # Build config programmatically
                cfg = self._build_config(data_path=model_dir, result_name=result_name)
                
                # Run prediction directly
                self._run_inference(cfg)
                logger.info("Inference completed successfully")
                self._clear_gpu_memory()
                
            except Exception as e:
                msg = f"Inference failed: {str(e)}"
                logger.error(msg)
                logger.exception(e)
                return {"status": msg, "glb_files": [], "scene": trimesh.Scene(), "temp_dir": temp_dir, "n_files": 0, "n_parts": 0, "unique_colors": 0}

            # Verify features were generated in exp_results
            # Model saves to "exp_results/{result_name}" relative to CWD
            # But we also check script_dir just in case
            possible_dirs = [
                os.path.join(os.getcwd(), "exp_results", result_name),
                os.path.join(self.script_dir, "exp_results", result_name)
            ]
            
            actual_features_dir = None
            for d in possible_dirs:
                if os.path.exists(d):
                    actual_features_dir = d
                    break
            
            if not actual_features_dir:
                msg = f"Inference output not found in: {possible_dirs}"
                logger.error(msg)
                return {"status": msg, "glb_files": [], "scene": trimesh.Scene(), "temp_dir": temp_dir, "n_files": 0, "n_parts": 0, "unique_colors": 0}
            
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
            
            # Create ply subdirectory for PartFieldClusterer output
            ply_dir = os.path.join(clustering_output_dir, "ply")
            os.makedirs(ply_dir, exist_ok=True)

            # Run clustering directly
            logger.info("Running clustering directly")
            
            try:
                # Initialize clusterer
                clusterer = PartFieldClusterer()
                
                # Get list of input files to process
                input_files = [
                    f for f in os.listdir(model_dir)
                    if (f.endswith(".ply") and is_pc) or (f.endswith((".obj", ".glb")) and not is_pc)
                ]

                if not input_files:
                    raise FileNotFoundError("No compatible input files found for clustering")

                for input_file in input_files:
                    input_path = os.path.join(model_dir, input_file)
                    uid = input_file.split('.')[0]
                    
                    # Load features for this model
                    features_path = os.path.join(features_dir, f"part_feat_{uid}_0.npy")
                    if not os.path.exists(features_path):
                        features_path = os.path.join(features_dir, f"part_feat_{uid}_0_batch.npy")
                    
                    if not os.path.exists(features_path):
                        logger.warning("Features not found for %s, skipping", uid)
                        continue
                    
                    logger.info("Processing: %s (uid=%s)", input_file, uid)
                    
                    # Call PartFieldClusterer (outputs to ply_dir)
                    result = clusterer.process(
                        features=features_path,
                        mesh=input_path,
                        n_parts=n_parts,
                        use_agglo=use_agglo,
                        clustering_option=clustering_option,
                        with_knn=with_knn,
                        is_pc=is_pc,
                        single_output=single_output,
                        export_mesh=True,
                        output_dir=ply_dir,
                        verbose=True
                    )
                    
                self._clear_gpu_memory()

            except Exception as e:
                msg = f"Clustering failed: {str(e)}"
                logger.error(msg)
                logger.exception(e)
                return {"status": msg, "glb_files": [], "scene": trimesh.Scene(), "temp_dir": temp_dir, "n_files": 0, "n_parts": 0, "unique_colors": 0}

            # Convert PLY to GLB (all in temp)
            if not os.path.isdir(ply_dir):
                msg = f"PLY output directory not found: {ply_dir}"
                logger.warning(msg)
                return {"status": msg, "glb_files": [], "scene": trimesh.Scene(), "temp_dir": temp_dir, "n_files": 0, "n_parts": 0, "unique_colors": 0}

            glb_dir = os.path.join(clustering_output_dir, "glbs")
            glb_files = self._convert_outputs_to_glb(ply_dir, glb_dir)

            # Load the best result into a Trimesh Scene for immediate use
            output_scene = trimesh.Scene()
            unique_color_count = 0
            if glb_files:
                # If multiple files are generated (hierarchical mode), pick the one with the highest granularity (last in sorted list)
                # If single_output=True, there is only one file anyway.
                best_file = glb_files[-1]

                output_scene = load_mesh_util(best_file)
                
                # Consolidate colors if needed (only for single output mode where we expect n_parts colors)
                current_colors = count_unique_colors(output_scene)
                if single_output:
                    if n_parts != current_colors:
                        output_scene = self._consolidate_mesh_colors(
                            mesh_or_scene=output_scene, 
                            n_parts=n_parts, 
                            method="agglo"
                        )
                                        
                    # Export the processed scene to a new GLB file
                    processed_glb_path = best_file.replace(".glb", "_processed.glb")
                    output_scene.export(processed_glb_path)
                    
                    # Update glb_files to point to the processed file
                    glb_files = [processed_glb_path]                        

                # Count unique colors in the segmented result
                unique_color_count = count_unique_colors(output_scene)

            if glb_files:
                if single_output:
                    msg = f"✓ Segmentation complete! Single output with {n_parts} part(s). Unique colors: {unique_color_count}"
                else:
                    msg = f"✓ Segmentation complete! Generated {len(glb_files)} segmentation(s). Unique colors: {unique_color_count}"
                
                return {
                    "status": msg,
                    "glb_files": glb_files,
                    "scene": output_scene,
                    "temp_dir": temp_dir,
                    "n_files": len(glb_files),
                    "n_parts": n_parts,
                    "unique_colors": unique_color_count
                }
            else:
                return {
                    "status": "No parts generated", 
                    "glb_files": [], 
                    "scene": trimesh.Scene(),
                    "temp_dir": temp_dir, 
                    "n_files": 0,
                    "n_parts": 0,
                    "unique_colors": 0
                }

        except Exception as e:
            logger.exception("Unexpected error during segmentation: %s", e)
            return {
                "status": f"Error: {e}", 
                "glb_files": [], 
                "scene": trimesh.Scene(),
                "temp_dir": temp_dir, 
                "n_files": 0,
                "n_parts": 0,
                "unique_colors": 0
            }
    
    def __del__(self) -> None:
        """Cleanup on deletion."""
        self._cleanup()
    
    # ======== Helper methods ========
    def _build_config(self, data_path: str, result_name: str) -> Any:
        """
        Constructs the configuration object programmatically.
        
        Args:
            data_path: Path to the dataset directory
            result_name: Name of the result (used for output folder naming)
            
        Returns:
            Config object (yacs CfgNode)
        """
        # Clone the base configuration
        cfg = base_cfg.clone()
        
        # Merge from config file if provided
        if self.config_path:
            cfg.merge_from_file(self.config_path)
        
        # Manual overrides (replacing argparse opts)
        cfg.dataset.data_path = data_path
        cfg.result_name = result_name
        cfg.continue_ckpt = self.checkpoint_path
        
        # Replicate setup() logic for output_dir
        # Note: This logic comes from partfield/config/__init__.py
        dt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        cfg.output_dir = os.path.join(cfg.output_dir, dt + "_" + cfg.name)
        
        # Override remesh_demo settings before freezing
        if cfg.remesh_demo:
            cfg.n_point_per_face = 10
        
        # Freeze the config to prevent accidental modification
        cfg.freeze()
        return cfg

    def _run_inference(self, cfg: Any) -> None:
        """
        Run inference using PyTorch Lightning Trainer.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch and Lightning are required for inference.")

        seed_everything(cfg.seed)
        
        checkpoint_callbacks = [ModelCheckpoint(
            monitor="train/current_epoch",
            dirpath=cfg.output_dir,
            filename="{epoch:02d}",
            save_top_k=100,
            save_last=True,
            every_n_epochs=cfg.save_every_epoch,
            mode="max",
            verbose=True
        )]

        trainer = Trainer(devices=1,
                          accelerator="gpu",
                          precision="16-mixed",
                          strategy="auto",
                          max_epochs=cfg.training_epochs,
                          log_every_n_steps=1,
                          limit_train_batches=3500,
                          limit_val_batches=None,
                          callbacks=checkpoint_callbacks,
                          logger=False  # Disable default logger to avoid tensorboardX requirement
                         )

        model = Model(cfg)
        trainer.predict(model, ckpt_path=cfg.continue_ckpt)
    
    def _clear_gpu_memory(self) -> None:
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
    
    def _download_model(self) -> bool:
        """Ensure a local copy of the PartField checkpoint exists at checkpoint_path."""
        # First check if the configured checkpoint file already exists
        if os.path.isfile(self.checkpoint_path):
            logger.info("Checkpoint file already exists: %s", self.checkpoint_path)
            return True
        
        model_dir = os.path.dirname(self.checkpoint_path)
        os.makedirs(model_dir, exist_ok=True)

        # Look for any .ckpt files already present in the directory
        ckpt_files = [p for p in os.listdir(model_dir) if p.endswith(".ckpt")] if os.path.isdir(model_dir) else []
        if ckpt_files:
            logger.info("Found existing checkpoint files in %s: %s", model_dir, ckpt_files)
            return True

        # Clone the repo directly into model_dir
        clone_url = "https://huggingface.co/mikaelaangel/partfield-ckpt"
        logger.info("Checkpoint not found at %s — starting git clone of %s", self.checkpoint_path, clone_url)
        
        # Remove the directory if it exists but is incomplete (to allow clean clone)
        if os.path.isdir(model_dir) and os.listdir(model_dir):
            logger.warning("Model directory exists but is incomplete. It will be cleaned before cloning.")
            try:
                shutil.rmtree(model_dir)
            except Exception as e:
                logger.error("Failed to remove incomplete model directory: %s", e)
                return False
        
        cmd = ["git", "clone", clone_url, model_dir]
        logger.info("Executing command: %s", " ".join(cmd))
        
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=self.script_dir)
            
            # Stream clone output for progress logging
            for line in iter(proc.stdout.readline, ""):
                if line:
                    logger.info("git clone progress: %s", line.rstrip())
            
            proc.wait(timeout=600)
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

        # Verify that the checkpoint file now exists
        if os.path.isfile(self.checkpoint_path):
            logger.info("Successfully downloaded checkpoint: %s", self.checkpoint_path)
            return True
        else:
            logger.warning("Downloaded checkpoint directory but configured file not found: %s", self.checkpoint_path)
            return False

    def _check_resources(self) -> None:
        """Verify that required checkpoint and config files exist."""
        if not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {self.checkpoint_path}")
        if not os.path.isfile(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
       
    def _preprocess_mesh(self, mesh_input: Union[trimesh.Trimesh, trimesh.Scene, str, Path], output_dir: str, threshold: float = 1e-6, is_pc: bool = False) -> str:
        """
        Preprocess a mesh by cleaning up vertices and removing duplicates.
        """
        logger.info("=== Starting mesh preprocessing ===")
        
        # Handle input loading
        if isinstance(mesh_input, (str, Path)):
            mesh_path = str(mesh_input)
            if not os.path.isfile(mesh_path):
                 raise FileNotFoundError(f"Cannot preprocess missing mesh: {mesh_path}")
            logger.info("Input mesh path: %s", mesh_path)
            loaded = load_mesh_util(mesh_path)
        elif isinstance(mesh_input, (trimesh.Trimesh, trimesh.Scene)):
            logger.info("Input mesh object provided")
            loaded = mesh_input
        else:
             raise ValueError(f"Unsupported input type: {type(mesh_input)}")

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

        cleaned_geoms = {name: self._clean_mesh_geometry(geom, threshold) for name, geom in trimesh_items.items()}

        cleaned_meshes = list(cleaned_geoms.values())

        combined_mesh = cleaned_meshes[0] if len(cleaned_meshes) == 1 else trimesh.util.concatenate(tuple(cleaned_meshes))

        total_vertices_after = len(combined_mesh.vertices)
        total_faces_after = len(combined_mesh.faces)
        logger.info(
            "After preprocessing: %d vertices, %d faces", 
            total_vertices_after, 
            total_faces_after
        )
        logger.info(
            "Removed %d vertices, %d faces",
            total_vertices_before - total_vertices_after,
            total_faces_before - total_faces_after,
        )

        os.makedirs(output_dir, exist_ok=True)
        
        # Determine output format based on is_pc
        # If is_pc is False (mesh mode), we must use .glb or .obj because Demo_Dataset 
        # ignores .ply files in mesh mode.
        ext = ".ply" if is_pc else ".glb"
        output_path = os.path.join(output_dir, f"mesh_preprocessed{ext}")
        
        combined_mesh.export(output_path)
        logger.info("Exported preprocessed mesh to %s", output_path)

        return output_path

    def _clean_mesh_geometry(
        self, 
        original: trimesh.Trimesh, 
        threshold: float
    ) -> trimesh.Trimesh:
        """Clean a single trimesh geometry."""
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
        
        # Detect mesh properties to decide merge behaviour.
        def _inspect_visual(mesh):
            vis = getattr(mesh, "visual", None)
            if not isinstance(vis, trimesh.visual.TextureVisuals):
                return False, False, False
            uv = getattr(vis, "uv", None)
            has_uv = uv is not None and len(uv) > 0
            mat = getattr(vis, "material", None)
            has_mat = mat is not None and any(
                getattr(mat, a, None) is not None
                for a in ("image", "baseColorTexture", "baseColorFactor")
            )
            has_nmap = mat is not None and any(
                getattr(mat, a, None) is not None
                for a in ("normalTexture", "normals")
            )
            return has_uv, has_mat, has_nmap

        has_uv, has_material, has_normal_map = _inspect_visual(processed)

        merge_tex = not (has_uv and has_material)
        merge_norm = not has_normal_map

        logger.debug("has_uv=%s, has_material=%s, has_normal_map=%s", has_uv, has_material, has_normal_map)
        logger.debug("merge_tex=%s, merge_norm=%s", merge_tex, merge_norm)
        
        verts_before = len(processed.vertices)
        processed.merge_vertices(
            merge_tex=merge_tex,
            merge_norm=merge_norm,
            digits_vertex=digits_vertex,
            digits_norm=max(1, digits_vertex - 1),
            digits_uv=max(1, digits_vertex - 1),
        )
        verts_after_merge = len(processed.vertices)
        logger.debug("Merge-vertices removed %d vertices", verts_before - verts_after_merge)

        processed.update_faces(processed.nondegenerate_faces())
        processed.update_faces(processed.unique_faces())
        processed.remove_unreferenced_vertices()
        verts_after_cleanup = len(processed.vertices)
        logger.debug("Cleanup removed %d vertices", verts_after_merge - verts_after_cleanup)

        # Refresh caches so normals/UVs stay consistent
        processed._cache.clear()

        # Compute vertex normals if the mesh doesn't already have them
        try:
            normals = processed.vertex_normals
            if normals is None or len(normals) == 0:
                raise ValueError("empty normals")
            logger.debug("  Vertex normals present (%d)", len(normals))
        except Exception:
            logger.debug("  No vertex normals found — computing from faces")
            processed.fix_normals()
            _ = processed.vertex_normals

        verts_final = len(processed.vertices)
        logger.debug("Cleaned mesh now has %d vertices, %d faces (total removed: %d)", verts_final, len(processed.faces), verts_before - verts_final)
        return processed

    def _convert_ply_to_glb(self, ply_path: str, glb_dir: str) -> str:
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
        mesh = load_mesh_util(ply_path)
        mesh.export(glb_path)
        return glb_path

    def _convert_outputs_to_glb(self, ply_dir: str, glb_dir: str) -> List[str]:
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
            for glb_path in [self._convert_ply_to_glb(ply_path, glb_dir)]
            if glb_path
        ]

        logger.info("Converted %d PLY files to GLB", len(glb_files))
        return glb_files

    def _cleanup(self) -> None:
        """Clean up all temporary directories created by process() calls."""
        for temp_dir in list(self.temp_dirs):
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    logger.info("[PartFieldSegmenter] Cleaned up temp dir: %s", temp_dir)
                except Exception as e:
                    logger.warning("[PartFieldSegmenter] Failed to cleanup %s: %s", temp_dir, e)
        self.temp_dirs.clear()
    
    def _consolidate_mesh_colors(
        self, 
        mesh_or_scene: Union[trimesh.Trimesh, trimesh.Scene], 
        n_parts: int,
        n_smooth_iters: int = 50,
        method: Literal["kmeans", "agglo"] = "kmeans",
        smoothing_thresholds: Optional[List[float]] = None,
        cleanup_iters: int = 30,
        cleanup_dist_ratio: float = 0.25,
        aggressive_absorption: bool = False
    ) -> trimesh.Scene:
        """
        Consolidate vertex colors into n_parts with robust boundary refinement.
        
        Args:
            method: 'kmeans' (faster) or 'agglo' (enforces connectivity).
            smoothing_thresholds: Majority vote thresholds for progressive smoothing.
            cleanup_iters: Iterations for small component removal.
            cleanup_dist_ratio: Max distance ratio for satellite protection.
            aggressive_absorption: When True, use aggressive fragment absorption
                that merges more small components (may over-merge small parts
                like eyes, buttons, etc.). When False (default), use milder
                settings that better preserve small but meaningful parts.
        """
        if smoothing_thresholds is None:
            smoothing_thresholds = [0.50, 0.45, 0.40, 0.35]
        
        # ── Shared micro-helpers ──────────────────────────────────────────
        def _build_wadj(nf, adjacency, angles):
            """Adjacency lists + cos(dihedral/2) edge weights."""
            ew = np.cos(np.minimum(angles, np.pi) * 0.5)
            al, aw = [[] for _ in range(nf)], [dict() for _ in range(nf)]
            for (a, b), w in zip(adjacency, ew):
                al[a].append(b)
                aw[a][b] = w
                al[b].append(a)
                aw[b][a] = w
            return al, aw

        def _wbest(f, al, aw, lbls):
            """Weighted vote for one face → (best_label, best_w, total_w)."""
            v = defaultdict(float)
            for n in al[f]:
                v[lbls[n]] += aw[f].get(n, 1.0)
            bl = max(v, key=v.get)
            return bl, v[bl], sum(v.values())

        def _smooth(lbls, al, aw, thresholds, iters=30):
            """Multi-phase dihedral-weighted majority-vote smoothing (in-place)."""
            for t in thresholds:
                for _ in range(iters):
                    u = {f: r[0] for f in range(len(al)) if al[f]
                        and (r := _wbest(f, al, aw, lbls))[0] != lbls[f]
                        and r[1] > r[2] * t}
                    if not u: break
                    lbls[list(u.keys())] = list(u.values())

        def _absorb(lbls, adj_pairs, al, min_sz, passes=6):
            """Absorb connected components < min_sz into neighbours (in-place)."""
            for _ in range(passes):
                dirty = False
                for lbl in np.unique(lbls):
                    m = lbls == lbl
                    comps = sorted(trimesh.graph.connected_components(
                        adj_pairs[m[adj_pairs].all(1)], nodes=np.where(m)[0]),
                        key=len, reverse=True)
                    for c in comps[1:]:
                        if len(c) >= min_sz: continue
                        for f in c:
                            if (nb := [lbls[n] for n in al[f] if lbls[n] != lbl]):
                                lbls[f] = Counter(nb).most_common(1)[0][0]
                                dirty = True
                if not dirty: break

        # ── Boundary Refinement ──────────────────────────────────────────
        def _refine_boundaries(scene_in: trimesh.Scene) -> trimesh.Scene:
            if not scene_in.geometry: return scene_in
            m = trimesh.util.concatenate(list(scene_in.geometry.values()))
            if len(m.faces) < 10: return scene_in

            uniq_cols, lbls = np.unique(
                m.visual.vertex_colors[:, :3][m.faces[:, 0]], axis=0, return_inverse=True)
            al, aw = _build_wadj(len(m.faces), m.face_adjacency, m.face_adjacency_angles)
            edges = m.face_adjacency

            # Vertex-ring progressive smoothing
            for thresh in smoothing_thresholds:
                for _ in range(n_smooth_iters):
                    u = {f: best
                        for vf in m.vertex_faces
                        for valid in [vf[vf != -1]] if len(valid) >= 2
                        for curr in [lbls[valid]] if len(set(curr)) > 1
                        for best, cnt in [Counter(curr).most_common(1)[0]]
                        if cnt >= len(valid) * thresh
                        for f in valid[curr != best]}
                    if not u: break
                    lbls[list(u.keys())] = list(u.values())

            # Face-adjacency weighted smoothing
            _smooth(lbls, al, aw, smoothing_thresholds[1:])

            # Component cleanup (weighted)
            min_sz = max(5, len(m.faces) // 80) if aggressive_absorption else max(5, len(m.faces) // 200)
            ctrs = m.vertices[m.faces].mean(1)
            bd = np.linalg.norm(m.bounds[1] - m.bounds[0])
            for _ in range(cleanup_iters):
                comps_by = defaultdict(list)
                [comps_by[lbls[c[0]]].append(c) for c in
                 trimesh.graph.connected_components(
                     edges[lbls[edges[:, 0]] == lbls[edges[:, 1]]],
                     nodes=np.arange(len(lbls)))]
                protected = np.zeros(len(lbls), dtype=bool)
                for cl in comps_by.values():
                    cl.sort(key=len, reverse=True)
                    protected[cl[0]] = len(cl[0]) >= min_sz
                    mp = ctrs[cl[0]].mean(0)
                    [np.put(protected, s, True) for s in cl[1:]
                     if len(s) >= min_sz or np.linalg.norm(ctrs[s].mean(0) - mp) < bd * cleanup_dist_ratio]
                flips = {f: r[0] for f in np.where(~protected)[0] if al[f]
                         and (r := _wbest(f, al, aw, lbls))[0] != lbls[f]}
                if not flips: break
                lbls[list(flips.keys())] = list(flips.values())

            res = trimesh.Scene()
            for i, col in enumerate(uniq_cols):
                if not (lbls == i).any(): continue
                sub = m.submesh([lbls == i], append=True)
                sub.visual.vertex_colors = np.tile(
                    np.append(col, 255).astype(np.uint8), (len(sub.vertices), 1))
                res.add_geometry(sub, node_name=f"part_{i}")
            return res

        # ── Main pipeline ────────────────────────────────────────────────
        if not isinstance(mesh_or_scene, (trimesh.Trimesh, trimesh.Scene)):
            raise TypeError(f"Expected Trimesh/Scene, got {type(mesh_or_scene).__name__}")

        logger.info("=== Color consolidation (%s): %d parts ===", method, n_parts)

        mesh = mesh_or_scene if isinstance(mesh_or_scene, trimesh.Trimesh) else mesh_or_scene.dump(concatenate=True)
        if len(mesh.faces) < max(n_parts, 10): return trimesh.Scene(mesh)

        # 1. Cluster face colours
        f_colors = (mesh.visual.vertex_colors[:, :3].astype(np.float32) / 255.0)[mesh.faces].mean(1)
        adj = mesh.face_adjacency

        if method == "agglo":
            from scipy.sparse import coo_matrix
            conn = coo_matrix((np.ones(len(adj), dtype=bool), (adj[:, 0], adj[:, 1])),
                              shape=(len(mesh.faces),) * 2)
            labels = AgglomerativeClustering(
                n_clusters=n_parts, connectivity=conn + conn.T).fit_predict(f_colors)
        else:
            labels = KMeans(n_clusters=n_parts, n_init=10, random_state=42).fit_predict(f_colors)

        # 2. Dihedral-weighted adjacency
        adj_list, adj_weight = _build_wadj(len(mesh.faces), adj, mesh.face_adjacency_angles)

        # 3. Spatial filtering — absorb small disconnected fragments
        centers, diag = mesh.triangles_center, np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
        if aggressive_absorption:
            min_frag = max(5, len(mesh.faces) // 100)
            sat_ratio, spatial_passes = 0.10, 4
        else:
            min_frag = max(3, len(mesh.faces) // 500)
            sat_ratio, spatial_passes = 0.05, 2
        for _ in range(spatial_passes):
            changed = False
            for lbl in range(n_parts):
                mask = labels == lbl
                if not mask.any(): continue
                comps = sorted(trimesh.graph.connected_components(
                    adj[mask[adj].all(1)], nodes=np.where(mask)[0]), key=len, reverse=True)
                if len(comps) < 2: continue
                mc = centers[comps[0]].mean(0)
                sats = [f for c in comps[1:]
                        if len(c) < min_frag or
                        (np.linalg.norm(centers[c].mean(0) - mc) > diag * cleanup_dist_ratio
                         and len(c) < len(comps[0]) * sat_ratio)
                        for f in c]
                updates = {f: Counter([labels[n] for n in adj_list[f] if labels[n] != lbl]).most_common(1)[0][0]
                           for f in sats if any(labels[n] != lbl for n in adj_list[f])}
                if updates:
                    labels[list(updates.keys())] = list(updates.values())
                    changed = True
            if not changed: break

        # 4. Multi-phase geometry-aware smoothing
        _smooth(labels, adj_list, adj_weight, smoothing_thresholds)

        # 4b. Morphological cleanup
        absorb_min = max(5, len(mesh.faces) // 80) if aggressive_absorption else max(3, len(mesh.faces) // 200)
        _absorb(labels, adj, adj_list, absorb_min)

        # 5. Split & colour
        unique_lbls = np.unique(labels)
        lbl_colors = {l: f_colors[labels == l].mean(0) for l in unique_lbls}
        scene = trimesh.Scene()
        for sub, lbl in zip(mesh.submesh([labels == l for l in unique_lbls], append=False), unique_lbls):
            c = np.append(lbl_colors[lbl] * 255, 255).astype(np.uint8)
            sub.visual.vertex_colors = np.tile(c, (len(sub.vertices), 1))
            scene.add_geometry(sub, node_name=f"part_{lbl}")

        return _refine_boundaries(scene)
