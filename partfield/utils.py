import trimesh
import numpy as np
from pathlib import Path
from typing import Union, Optional
import logging

logger = logging.getLogger(__name__)


def load_mesh_util(input_fname):
    """
    Utility function to load a mesh from file using trimesh.
    """
    mesh = trimesh.load(
        input_fname, 
        force='mesh', 
        process=False,
        merge_primitives=True, skip_materials=False, maintain_order=True)
    return mesh


def normalize_colors_to_rgba8(colors: np.ndarray) -> np.ndarray:
    """Normalize color array to RGBA uint8 format (0-255)."""
    colors = np.asarray(colors)
    # Convert uint8 to float [0,1]
    if colors.dtype == np.uint8:
        colors = colors.astype(np.float32) / 255.0
    else:
        colors = colors.astype(np.float32)
    # Flatten to 2D if needed
    if colors.ndim == 1:
        colors = colors.reshape(1, -1)    
    # Ensure 4 channels (add alpha if needed)
    if colors.shape[1] == 3:
        colors = np.hstack([colors, np.ones((colors.shape[0], 1), dtype=np.float32)])
    elif colors.shape[1] != 4:
        raise ValueError(f"Expected 3 or 4 color channels, got {colors.shape[1]}")    
    # Clamp to [0,1] and convert to uint8
    return np.clip(colors * 255, 0, 255).astype(np.uint8)


def count_unique_colors(
    mesh_or_scene: Union[trimesh.Trimesh, trimesh.Scene, Path, str]
) -> int:
    """
    Count the number of unique colors in a mesh or scene.
    """
    if mesh_or_scene is None:
        raise ValueError("Input mesh_or_scene cannot be None.")
    
    # Load from path if needed
    if isinstance(mesh_or_scene, (str, Path)):
        mesh_path = str(mesh_or_scene)
        try:
            mesh_or_scene = trimesh.load(mesh_path, force='mesh', process=False)
        except Exception as e:
            raise ValueError(f"Failed to load mesh from {mesh_path}: {e}")    
    # Convert Trimesh to Scene
    if isinstance(mesh_or_scene, trimesh.Trimesh):
        scene = trimesh.Scene(mesh_or_scene)
    elif isinstance(mesh_or_scene, trimesh.Scene):
        scene = mesh_or_scene
    else:
        raise TypeError(f"Expected trimesh.Trimesh, trimesh.Scene, or file path, got {type(mesh_or_scene)}")
    
    if not scene.geometry:
        logger.warning("[count_unique_colors] Input scene has no geometry.")
        return 0
    
    # Collect all unique colors from all geometries
    all_unique_colors = set()
    
    for _, geom in scene.geometry.items():
        if not isinstance(geom, trimesh.Trimesh):
            continue
        
        # Priority 1: Try vertex colors first
        vc = getattr(geom.visual, 'vertex_colors', None)
        if vc is not None and len(vc) > 0:
            try:
                vertex_colors = normalize_colors_to_rgba8(vc)
                unique_colors_in_geom = np.unique(vertex_colors, axis=0)
                for color in unique_colors_in_geom:
                    all_unique_colors.add(tuple(color))
                continue  # Found vertex colors, skip to next geometry
            except Exception as e:
                logger.debug(f"[count_unique_colors] Failed to process vertex colors: {e}")
        
        # Priority 2: Try face colors
        fc = getattr(geom.visual, 'face_colors', None)
        if fc is not None and len(fc) > 0:
            try:
                face_colors = normalize_colors_to_rgba8(fc)
                unique_colors_in_geom = np.unique(face_colors, axis=0)
                for color in unique_colors_in_geom:
                    all_unique_colors.add(tuple(color))
                continue  # Found face colors, skip to next geometry
            except Exception as e:
                logger.debug(f"[count_unique_colors] Failed to process face colors: {e}")
        
        # Priority 3: Try material colors
        material = getattr(geom.visual, 'material', None)
        if material is not None:
            # Try multiple material color attributes
            for attr in ('main_color', 'baseColorFactor', 'diffuse'):
                val = getattr(material, attr, None)
                if val is not None:
                    try:
                        mat_color = normalize_colors_to_rgba8(val)
                        if mat_color.size > 0:
                            all_unique_colors.add(tuple(mat_color[0]))
                            break  # Found a material color, move to next geometry
                    except Exception as e:
                        logger.debug(f"[count_unique_colors] Failed to process material color ({attr}): {e}")
    
    num_unique_colors = len(all_unique_colors)
    logger.info(f"[count_unique_colors] Found {num_unique_colors} unique colors in scene with {len(scene.geometry)} geometries")
    
    return num_unique_colors