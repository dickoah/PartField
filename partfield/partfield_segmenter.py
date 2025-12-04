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
from typing import Union, Optional, List, Dict, Any
from pathlib import Path

import trimesh
import random
import numpy as np

# Try to import torch and GPU utilities for memory analysis
try:
    import torch
    from lightning.pytorch import seed_everything, Trainer
    from lightning.pytorch.strategies import DDPStrategy
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

# Setup logging
logger = logging.getLogger("partfield_segmenter")


class PartFieldSegmenter:
    """
    PartField Mesh Segmentation Engine.
    
    Segments 3D meshes into semantic parts using learned PartField features.
    Includes inference, clustering, and mesh export.
    
    Attributes:
        n_parts: Number of parts for clustering
        use_agglo: Use agglomerative clustering vs K-Means
        clustering_option: Adjacency matrix option (0=naive, 1=MST-based, 2=CC-MST)
        with_knn: Add KNN edges for messy meshes
        is_pc: Input is point cloud (True) vs mesh (False)
        single_output: Generate only final N-part segmentation (faster)
        enable_preprocessing: Enable mesh vertex deduplication
        preprocessing_threshold: Vertex merge threshold
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
        
        # Temp directory
        self.temp_dir = None
        
        # Download model if needed, then validate all resources exist
        self._download_model()
        self._check_resources()
        
        logger.info(f"[PartFieldSegmenter] Initialized")

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
        preprocess_model: bool = False,
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
            preprocess_model: Use preprocessed mesh for inference
            preprocessing_threshold: Vertex dedup threshold (default: 1e-6)
        
        Returns:
            dict with keys:
                - 'status': Success message
                - 'glb_files': List of output GLB file paths
                - 'scene': Trimesh Scene object of the result
                - 'temp_dir': Temporary directory path (will be cleaned up on exit)
                - 'n_parts': Number of parts generated
        """
        if n_parts < 1:
            raise ValueError("Number of parts must be at least 1")

        logger.info("=== Starting segmentation pipeline ===")
        logger.info("Parts: %d, Agglo: %s, Option: %d, KNN: %s, Single: %s", 
                    n_parts, use_agglo, clustering_option, with_knn, single_output)

        # Create temp directory for this run
        self.temp_dir = tempfile.mkdtemp(prefix="partfield_seg_")
        
        try:
            # Set up unique result directory in temp folder
            timestamp = int(time.time())
            
            # Determine basename
            if isinstance(mesh, (str, Path)):
                model_basename = os.path.basename(str(mesh)).split('.')[0]
            else:
                model_basename = "mesh_object"
                
            result_name = f"results_{timestamp}_{model_basename}"
            
            temp_result_dir = os.path.join(self.temp_dir, result_name)
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
                target_mesh_path = os.path.join(model_dir, "input_mesh.glb" )
                mesh.export(target_mesh_path)
                logger.info("Exported input mesh object to %s", target_mesh_path)
            else:
                raise ValueError(f"Unsupported input type: {type(mesh)}")
            
            # Preprocess the mesh
            if enable_preprocessing:
                logger.info("=== Mesh Preprocessing ===")
                preprocess_dir = os.path.join(temp_result_dir, "preprocessing")
                os.makedirs(preprocess_dir, exist_ok=True)
                try:
                    # Preprocess returns path to PLY in preprocess_dir
                    preprocessed_mesh_path = self._preprocess_mesh(target_mesh_path, preprocess_dir, preprocessing_threshold, is_pc=is_pc)
                    
                    if preprocess_model:
                        logger.info("Using preprocessed mesh for inference: %s", preprocessed_mesh_path)
                        model_dir = preprocess_dir
                    else:
                        logger.info("Mesh preprocessing completed at %s (but using original mesh for inference to match legacy behavior)", preprocessed_mesh_path)
                except Exception as e:
                    logger.warning("Mesh preprocessing failed: %s. Continuing with original mesh.", e)
                    # Continue with original mesh if preprocessing fails

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
                return {"status": msg, "glb_files": [], "scene": trimesh.Scene(), "temp_dir": self.temp_dir, "n_parts": 0}

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
                return {"status": msg, "glb_files": [], "scene": trimesh.Scene(), "temp_dir": self.temp_dir, "n_parts": 0}
            
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
                if preprocess_model and enable_preprocessing:
                     # If using preprocessed model, look for the file we just generated
                     ext = ".ply" if is_pc else ".glb"
                     input_files = [f for f in os.listdir(model_dir) if f.endswith(ext)]
                else:
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
                return {"status": msg, "glb_files": [], "scene": trimesh.Scene(), "temp_dir": self.temp_dir, "n_parts": 0}

            # Convert PLY to GLB (all in temp)
            if not os.path.isdir(ply_dir):
                msg = f"PLY output directory not found: {ply_dir}"
                logger.warning(msg)
                return {"status": msg, "glb_files": [], "scene": trimesh.Scene(), "temp_dir": self.temp_dir, "n_parts": 0}

            glb_dir = os.path.join(clustering_output_dir, "glbs")
            glb_files = self._convert_outputs_to_glb(ply_dir, glb_dir)

            # Load the best result into a Trimesh Scene for immediate use
            output_scene = trimesh.Scene()
            if glb_files:
                # If multiple files are generated (hierarchical mode), pick the one with the highest granularity (last in sorted list)
                # If single_output=True, there is only one file anyway.
                best_file = glb_files[-1]
                try:
                    output_scene = trimesh.load(best_file, force='scene')
                except Exception as e:
                    logger.warning("Failed to load result into Trimesh Scene: %s", e)

            if glb_files:
                if single_output:
                    msg = f"✓ Segmentation complete! Single output with {n_parts} part(s)."
                else:
                    msg = f"✓ Segmentation complete! Generated {len(glb_files)} segmentation(s)."
                
                return {
                    "status": msg,
                    "glb_files": glb_files,
                    "scene": output_scene,
                    "temp_dir": self.temp_dir,
                    "n_parts": len(glb_files)
                }
            else:
                return {
                    "status": "No parts generated", 
                    "glb_files": [], 
                    "scene": trimesh.Scene(),
                    "temp_dir": self.temp_dir, 
                    "n_parts": 0
                }

        except Exception as e:
            logger.exception("Unexpected error during segmentation: %s", e)
            return {
                "status": f"Error: {e}", 
                "glb_files": [], 
                "scene": trimesh.Scene(),
                "temp_dir": self.temp_dir, 
                "n_parts": 0
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

        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
        
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

        trainer = Trainer(devices=1, # Use 1 GPU to avoid DDP complications in interactive mode
                          accelerator="gpu",
                          precision="16-mixed",
                          strategy=DDPStrategy(find_unused_parameters=True),
                          max_epochs=cfg.training_epochs,
                          log_every_n_steps=1,
                          limit_train_batches=3500,
                          limit_val_batches=None,
                          callbacks=checkpoint_callbacks,
                          logger=False  # Disable default logger to avoid tensorboardX requirement
                         )

        model = Model(cfg)        

        if cfg.remesh_demo:
            cfg.n_point_per_face = 10

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
            loaded = trimesh.load(mesh_path, force='scene')
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
        logger.info("After preprocessing: %d vertices, %d faces", total_vertices_after, total_faces_after)
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

    def _clean_mesh_geometry(self, original: trimesh.Trimesh, threshold: float) -> trimesh.Trimesh:
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
        mesh = trimesh.load(ply_path, process=False)
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
        """Clean up temporary files."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                logger.info(f"[PartFieldSegmenter] Cleaned up temp dir")
            except Exception as e:
                logger.warning(f"[PartFieldSegmenter] Failed to cleanup: {e}")