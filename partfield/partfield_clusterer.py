"""
Part Clustering Module

A modular interface for clustering mesh or point cloud features into semantic parts.
Can be used as a standalone module or integrated into other projects.

Example:
    from partfield_clusterer import PartClusterer
    
    clusterer = PartClusterer()
    result = clusterer.process(
        features="path/to/features.npy",
        mesh="path/to/mesh.ply",
        n_parts=10,
        use_agglo=True
    )
"""

import os
import logging
import numpy as np
from typing import Union, Optional, Dict, Any
from pathlib import Path
from collections import defaultdict

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
import networkx as nx
from networkx.utils import UnionFind
import trimesh
import open3d as o3d
import matplotlib.pyplot as plt

from .utils import load_mesh_util

# Setup logging
logger = logging.getLogger("partfield_clusterer")


class PartFieldClusterer:
    """
    Part Clustering Engine.
    
    Clusters mesh or point cloud features into semantic parts using K-Means or
    Agglomerative Clustering. Supports multiple adjacency matrix strategies for
    mesh-based clustering.
    
    Attributes:
        use_agglo: Use agglomerative clustering vs K-Means
        clustering_option: Adjacency matrix option (0=naive, 1=MST-based, 2=CC-MST)
        with_knn: Add KNN edges for messy meshes
        is_pc: Input is point cloud (True) vs mesh (False)
        single_output: Generate only final N-part segmentation (faster)
        export_mesh: Export colored mesh/point cloud files
    """
    
    def __init__(self):
        """Initialize the Part Clusterer."""
        logger.info("[PartClusterer] Initialized")
    
    def process(
        self,
        features: Union[str, Path, np.ndarray],
        mesh: Optional[Union[trimesh.Trimesh, trimesh.Scene, str, Path]] = None,
        n_parts: int = 10,
        use_agglo: bool = True,
        clustering_option: int = 1,
        with_knn: bool = True,
        is_pc: bool = False,
        single_output: bool = True,
        export_mesh: bool = True,
        output_dir: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Cluster features into semantic parts.
        
        Args:
            features: Path to .npy file or numpy array of shape (N, D) with feature vectors
            mesh: Optional mesh for adjacency matrix construction. Can be:
                - trimesh.Trimesh object
                - trimesh.Scene object
                - Path to mesh file (.obj, .ply, .glb, etc.)
                (required if use_agglo=True)
            n_parts: Number of parts for clustering (default: 10)
            use_agglo: Use agglomerative clustering (default: True) vs K-Means
            clustering_option: 0=naive, 1=MST-based (default), 2=CC-MST
            with_knn: Add KNN edges (default: True) - for noisy meshes
            is_pc: Input is point cloud (default: False)
            single_output: Only output final N-part segmentation (default: True)
            export_mesh: Export colored mesh/point cloud files (default: True)
            output_dir: Directory for saving results (default: current directory)
            verbose: Print progress messages (default: True)
        
        Returns:
            Dictionary with:
                - 'labels': np.ndarray of shape (N,) with cluster labels
                - 'hierarchical_labels': Optional list of intermediate segmentations
                - 'output_files': List of generated files
        """
        
        # Load features
        if isinstance(features, (str, Path)):
            if not os.path.exists(features):
                raise FileNotFoundError(f"Features file not found: {features}")
            point_feat = np.load(features)
            if verbose:
                logger.info(f"Loaded features from {features}: shape {point_feat.shape}")
        else:
            point_feat = np.asarray(features)
        
        # Normalize features
        point_feat = point_feat / np.linalg.norm(point_feat, axis=-1, keepdims=True)
        
        # Load mesh if needed
        mesh_obj = None
        if mesh is not None:
            if isinstance(mesh, str) or isinstance(mesh, Path):
                mesh_obj = load_mesh_util(str(mesh))
                if verbose:
                    logger.info(f"Loaded mesh: {mesh_obj.vertices.shape[0]} vertices, {mesh_obj.faces.shape[0]} faces")
            elif isinstance(mesh, trimesh.Scene):
                # Convert Scene to Trimesh by merging all geometries
                mesh_obj = mesh.dump(concatenate=True)
                if verbose:
                    logger.info(f"Loaded scene: {mesh_obj.vertices.shape[0]} vertices, {mesh_obj.faces.shape[0]} faces")
            else:
                mesh_obj = mesh
        
        # Create output directory
        if output_dir is None:
            output_dir = os.getcwd()
        os.makedirs(output_dir, exist_ok=True)
        
        output_files = []
        
        # Perform clustering
        if not use_agglo:
            # K-Means clustering
            if verbose:
                logger.info(f"Running K-Means with {n_parts} clusters")
            
            cluster_range = [n_parts] if single_output else range(2, n_parts + 1)
            
            for num_cluster in cluster_range:
                clustering = KMeans(n_clusters=num_cluster, random_state=0).fit(point_feat)
                labels = clustering.labels_
                
                # Relabel to 0-indexed
                relabeled = np.zeros(len(labels))
                for i, label in enumerate(np.unique(labels)):
                    relabeled[labels == label] = i
                
                # Export if requested
                if export_mesh and mesh_obj is not None:
                    out_file = self._export_result(
                        mesh_obj, relabeled, is_pc, output_dir, num_cluster, verbose
                    )
                    output_files.extend(out_file)
            
            return {
                'labels': relabeled,
                'hierarchical_labels': None,
                'output_files': output_files,
            }
        
        else:
            # Agglomerative clustering
            if mesh_obj is None:
                raise ValueError(
                    "Agglomerative clustering requires a mesh. "
                    "Please provide mesh parameter or set use_agglo=False."
                )
            
            if is_pc:
                raise NotImplementedError(
                    "Agglomerative clustering only works with mesh inputs, not point clouds."
                )
            
            if verbose:
                logger.info(f"Constructing adjacency matrix (option={clustering_option})")
            
            # Build adjacency matrix
            adj_matrix = self._construct_adjacency_matrix(
                mesh_obj.faces,
                mesh_obj.vertices,
                clustering_option,
                with_knn,
                verbose
            )
            
            if single_output:
                # Direct clustering to exactly n_parts
                if verbose:
                    logger.info(f"Running agglomerative clustering to {n_parts} clusters")
                
                clustering = AgglomerativeClustering(
                    connectivity=adj_matrix,
                    n_clusters=n_parts,
                ).fit(point_feat)
                
                labels = clustering.labels_
                
                # Export
                if export_mesh:
                    self._export_result(
                        mesh_obj, labels, is_pc, output_dir, n_parts, verbose
                    )
                
                return {
                    'labels': labels,
                    'hierarchical_labels': None,
                    'output_files': output_files,
                }
            
            else:
                # Hierarchical mode: compute all intermediate segmentations
                if verbose:
                    logger.info("Running hierarchical agglomerative clustering")
                
                clustering = AgglomerativeClustering(
                    connectivity=adj_matrix,
                    n_clusters=1,
                ).fit(point_feat)
                
                # Extract hierarchical labels at each step
                hierarchical_labels = self._extract_hierarchical_labels(
                    clustering.children_, point_feat.shape[0], n_parts, verbose
                )
                
                # Export all intermediate results
                if export_mesh:
                    unique_labels = np.unique(np.concatenate(hierarchical_labels))
                    for i, labels in enumerate(hierarchical_labels):
                        num_clusters = n_parts - i
                        self._export_result(
                            mesh_obj, labels, is_pc, output_dir, num_clusters, verbose
                        )
                
                return {
                    'labels': hierarchical_labels[-1] if hierarchical_labels else None,
                    'hierarchical_labels': hierarchical_labels,
                    'output_files': output_files,
                }
    
    def _construct_adjacency_matrix(
        self,
        faces,
        vertices,
        option: int = 1,
        with_knn: bool = True,
        verbose: bool = True,
    ) -> csr_matrix:
        """
        Construct face-based adjacency matrix.
        
        Args:
            faces: Array of shape (M, 3) with vertex indices
            vertices: Array of shape (N, 3) with vertex positions
            option: 0=naive, 1=MST-based, 2=CC-MST
            with_knn: Add KNN edges
            verbose: Print progress
        
        Returns:
            Sparse adjacency matrix of shape (M, M)
        """
        if option == 0:
            if verbose:
                logger.info("Using naive adjacency matrix")
            return self._construct_face_adjacency_matrix_naive(faces)
        elif option == 1:
            if verbose:
                logger.info("Using face-MST adjacency matrix")
            return self._construct_face_adjacency_matrix_facemst(
                faces, vertices, with_knn=with_knn
            )
        else:
            if verbose:
                logger.info("Using component-MST adjacency matrix")
            return self._construct_face_adjacency_matrix_ccmst(
                faces, vertices, with_knn=with_knn
            )
    
    def _construct_face_adjacency_matrix_naive(self, face_list):
        """
        Construct adjacency matrix using shared edges only.
        Connects disconnected components with dummy edges.
        """
        num_faces = len(face_list)
        if num_faces == 0:
            return csr_matrix((0, 0))
        
        edge_to_faces = defaultdict(list)
        
        for f_idx, (v0, v1, v2) in enumerate(face_list):
            edges = [
                tuple(sorted((v0, v1))),
                tuple(sorted((v1, v2))),
                tuple(sorted((v2, v0)))
            ]
            for e in edges:
                edge_to_faces[e].append(f_idx)
        
        row = []
        col = []
        for e, faces_sharing_e in edge_to_faces.items():
            f_indices = list(set(faces_sharing_e))
            if len(f_indices) > 1:
                for i in range(len(f_indices)):
                    for j in range(i + 1, len(f_indices)):
                        f_i = f_indices[i]
                        f_j = f_indices[j]
                        row.extend([f_i, f_j])
                        col.extend([f_j, f_i])
        
        data = np.ones(len(row), dtype=np.int8)
        face_adjacency = coo_matrix(
            (data, (row, col)),
            shape=(num_faces, num_faces)
        ).tocsr()
        
        # Ensure single connected component
        n_components, labels = connected_components(face_adjacency, directed=False)
        
        if n_components > 1:
            component_representatives = []
            for comp_id in range(n_components):
                faces_in_comp = np.where(labels == comp_id)[0]
                if len(faces_in_comp) > 0:
                    component_representatives.append(faces_in_comp[0])
            
            dummy_row = []
            dummy_col = []
            for i in range(len(component_representatives) - 1):
                f_i = component_representatives[i]
                f_j = component_representatives[i + 1]
                dummy_row.extend([f_i, f_j])
                dummy_col.extend([f_j, f_i])
            
            if dummy_row:
                dummy_data = np.ones(len(dummy_row), dtype=np.int8)
                dummy_mat = coo_matrix(
                    (dummy_data, (dummy_row, dummy_col)),
                    shape=(num_faces, num_faces)
                ).tocsr()
                face_adjacency = face_adjacency + dummy_mat
        
        return face_adjacency
    
    def _construct_face_adjacency_matrix_facemst(
        self, face_list, vertices, k=10, with_knn=True
    ):
        """Construct adjacency matrix using face MST method."""
        num_faces = len(face_list)
        if num_faces == 0:
            return csr_matrix((0, 0))
        
        # Step 1: Build shared-edge adjacency
        edge_to_faces = defaultdict(list)
        uf = UnionFind(range(num_faces))
        
        for f_idx, (v0, v1, v2) in enumerate(face_list):
            edges = [
                tuple(sorted((v0, v1))),
                tuple(sorted((v1, v2))),
                tuple(sorted((v2, v0)))
            ]
            for e in edges:
                edge_to_faces[e].append(f_idx)
        
        row = []
        col = []
        for edge, face_indices in edge_to_faces.items():
            unique_faces = list(set(face_indices))
            if len(unique_faces) > 1:
                for i in range(len(unique_faces)):
                    for j in range(i + 1, len(unique_faces)):
                        fi = unique_faces[i]
                        fj = unique_faces[j]
                        row.extend([fi, fj])
                        col.extend([fj, fi])
                        uf.union(fi, fj)
        
        data = np.ones(len(row), dtype=np.int8)
        face_adjacency = coo_matrix(
            (data, (row, col)), shape=(num_faces, num_faces)
        ).tocsr()
        
        # Step 2: Check connectivity
        n_components = len(set(uf[i] for i in range(num_faces)))
        
        if n_components == 1:
            return face_adjacency
        
        # Step 3: Compute face centroids
        face_centroids = []
        for (v0, v1, v2) in face_list:
            centroid = (vertices[v0] + vertices[v1] + vertices[v2]) / 3.0
            face_centroids.append(centroid)
        face_centroids = np.array(face_centroids)
        
        # Step 4: Build KNN graph
        knn = NearestNeighbors(n_neighbors=min(k, num_faces), algorithm='auto')
        knn.fit(face_centroids)
        distances, indices = knn.kneighbors(face_centroids)
        
        # Step 5: Build weighted graph and MST
        G = nx.Graph()
        G.add_nodes_from(range(num_faces))
        
        for i in range(num_faces):
            for j, dist in zip(indices[i], distances[i]):
                if i != j:
                    G.add_edge(i, j, weight=dist)
        
        mst = nx.minimum_spanning_tree(G, weight='weight')
        mst_edges_sorted = sorted(
            mst.edges(data=True), key=lambda e: e[2]['weight']
        )
        
        # Step 6: Add MST edges to adjacency
        adjacency_lil = face_adjacency.tolil()
        uf_mst = UnionFind(range(num_faces))
        
        for (u, v, attr) in mst_edges_sorted:
            if uf_mst[u] != uf_mst[v]:
                uf_mst.union(u, v)
                adjacency_lil[u, v] = 1
                adjacency_lil[v, u] = 1
        
        face_adjacency = adjacency_lil.tocsr()
        
        # Step 7: Optionally add KNN edges
        if with_knn:
            dummy_row = []
            dummy_col = []
            for i in range(num_faces):
                for j in indices[i]:
                    dummy_row.extend([i, j])
                    dummy_col.extend([j, i])
            
            dummy_data = np.ones(len(dummy_row), dtype=np.int16)
            dummy_mat = coo_matrix(
                (dummy_data, (dummy_row, dummy_col)),
                shape=(num_faces, num_faces)
            ).tocsr()
            face_adjacency = face_adjacency + dummy_mat
        
        return face_adjacency
    
    def _construct_face_adjacency_matrix_ccmst(
        self, face_list, vertices, k=10, with_knn=True
    ):
        """
        Construct adjacency matrix using connected-component MST method.
        Better for meshes with multiple disconnected parts.
        """
        num_faces = len(face_list)
        if num_faces == 0:
            return csr_matrix((0, 0))
        
        # Step 1: Build shared-edge adjacency
        edge_to_faces = defaultdict(list)
        uf = UnionFind(range(num_faces))
        
        for f_idx, (v0, v1, v2) in enumerate(face_list):
            edges = [
                tuple(sorted((v0, v1))),
                tuple(sorted((v1, v2))),
                tuple(sorted((v2, v0)))
            ]
            for e in edges:
                edge_to_faces[e].append(f_idx)
        
        row = []
        col = []
        for edge, face_indices in edge_to_faces.items():
            unique_faces = list(set(face_indices))
            if len(unique_faces) > 1:
                for i in range(len(unique_faces)):
                    for j in range(i + 1, len(unique_faces)):
                        fi = unique_faces[i]
                        fj = unique_faces[j]
                        row.extend([fi, fj])
                        col.extend([fj, fi])
                        uf.union(fi, fj)
        
        data = np.ones(len(row), dtype=np.int8)
        face_adjacency = coo_matrix(
            (data, (row, col)), shape=(num_faces, num_faces)
        ).tocsr()
        
        # Step 2: Check connectivity
        n_components = len(set(uf[i] for i in range(num_faces)))
        
        if n_components == 1:
            return face_adjacency
        
        # Step 3: Compute face centroids and group by component
        face_centroids = []
        for (v0, v1, v2) in face_list:
            centroid = (vertices[v0] + vertices[v1] + vertices[v2]) / 3.0
            face_centroids.append(centroid)
        face_centroids = np.array(face_centroids)
        
        component_dict = {}
        for face_idx in range(num_faces):
            root = uf[face_idx]
            if root not in component_dict:
                component_dict[root] = set()
            component_dict[root].add(face_idx)
        
        connected_components_list = list(component_dict.values())
        
        # Step 4: Find representative face for each component
        component_centroid_face_idx = []
        connected_component_centroids = []
        
        for component in connected_components_list:
            curr_component_faces = list(component)
            curr_component_face_centroids = face_centroids[curr_component_faces]
            component_centroid = np.mean(curr_component_face_centroids, axis=0)
            
            face_idx = curr_component_faces[
                np.argmin(
                    np.linalg.norm(
                        curr_component_face_centroids - component_centroid, axis=-1
                    )
                )
            ]
            
            connected_component_centroids.append(component_centroid)
            component_centroid_face_idx.append(face_idx)
        
        component_centroid_face_idx = np.array(component_centroid_face_idx)
        connected_component_centroids = np.array(connected_component_centroids)
        
        # Step 5: Build KNN on component centroids
        knn_k = min(k, n_components)
        knn = NearestNeighbors(n_neighbors=knn_k, algorithm='auto')
        knn.fit(connected_component_centroids)
        distances, indices = knn.kneighbors(connected_component_centroids)
        
        # Step 6: Build MST on components
        G = nx.Graph()
        G.add_nodes_from(range(n_components))
        
        for idx1 in range(n_components):
            i = component_centroid_face_idx[idx1]
            for idx2, dist in zip(indices[idx1], distances[idx1]):
                j = component_centroid_face_idx[idx2]
                if i != j:
                    G.add_edge(i, j, weight=dist)
        
        mst = nx.minimum_spanning_tree(G, weight='weight')
        mst_edges_sorted = sorted(
            mst.edges(data=True), key=lambda e: e[2]['weight']
        )
        
        # Step 7: Add MST edges
        adjacency_lil = face_adjacency.tolil()
        uf_mst = UnionFind(range(num_faces))
        
        for (u, v, attr) in mst_edges_sorted:
            if uf_mst[u] != uf_mst[v]:
                uf_mst.union(u, v)
                adjacency_lil[u, v] = 1
                adjacency_lil[v, u] = 1
        
        face_adjacency = adjacency_lil.tocsr()
        
        # Step 8: Optionally add KNN edges
        if with_knn:
            dummy_row = []
            dummy_col = []
            for idx1 in range(n_components):
                i = component_centroid_face_idx[idx1]
                for idx2 in indices[idx1]:
                    j = component_centroid_face_idx[idx2]
                    dummy_row.extend([i, j])
                    dummy_col.extend([j, i])
            
            dummy_data = np.ones(len(dummy_row), dtype=np.int16)
            dummy_mat = coo_matrix(
                (dummy_data, (dummy_row, dummy_col)),
                shape=(num_faces, num_faces)
            ).tocsr()
            face_adjacency = face_adjacency + dummy_mat
        
        return face_adjacency
    
    def _extract_hierarchical_labels(
        self, children, n_samples, max_cluster, verbose=True
    ):
        """Extract cluster labels at each hierarchical merge step."""
        uf = UnionFind(range(2 * n_samples - 1))
        
        hierarchical_labels = []
        for i, (child1, child2) in enumerate(children):
            uf.union(child1, i + n_samples)
            uf.union(child2, i + n_samples)
            
            current_cluster_count = n_samples - (i + 1)
            if current_cluster_count <= max_cluster:
                labels = np.array([uf[j] for j in range(n_samples)])
                hierarchical_labels.append(labels)
        
        return hierarchical_labels
    
    def _export_result(
        self,
        mesh_obj,
        labels,
        is_pc: bool,
        output_dir: str,
        num_clusters: int,
        verbose: bool = True,
    ):
        """Export clustering result to file."""
        output_files = []
        
        # Relabel to 0-indexed
        relabeled = np.zeros(len(labels), dtype=np.int32)
        for i, label in enumerate(np.unique(labels)):
            relabeled[labels == label] = i
        
        # Export PLY file
        if not is_pc:
            V = mesh_obj.vertices
            F = mesh_obj.faces
            
            out_file = os.path.join(output_dir, f"cluster_{num_clusters:02d}.ply")
            self._export_colored_mesh_ply(V, F, relabeled, filename=out_file)
            output_files.append(out_file)
            
            if verbose:
                logger.info(f"Exported mesh to {out_file}")
        
        else:
            out_file = os.path.join(output_dir, f"cluster_{num_clusters:02d}.ply")
            self._export_pointcloud_with_labels_to_ply(
                mesh_obj.vertices, relabeled, filename=out_file
            )
            output_files.append(out_file)
            
            if verbose:
                logger.info(f"Exported point cloud to {out_file}")
        
        return output_files
    
    def _export_colored_mesh_ply(
        self, V, F, FL, filename='segmented_mesh.ply'
    ):
        """Export mesh with per-face color labels."""
        assert V.shape[1] == 3
        assert F.shape[1] == 3
        assert F.shape[0] == FL.shape[0]
        
        unique_labels = np.unique(FL)
        colormap = plt.cm.get_cmap("tab20", len(unique_labels))
        label_to_color = {
            label: (np.array(colormap(i)[:3]) * 255).astype(np.uint8)
            for i, label in enumerate(unique_labels)
        }
        
        mesh = trimesh.Trimesh(vertices=V, faces=F)
        FL = np.squeeze(FL)
        for i, face in enumerate(F):
            label = FL[i]
            color = label_to_color[label]
            color_with_alpha = np.append(color, 255)
            mesh.visual.face_colors[i] = color_with_alpha
        
        mesh.export(filename)
    
    def _export_pointcloud_with_labels_to_ply(
        self, V, VL, filename='colored_pointcloud.ply'
    ):
        """Export point cloud with vertex color labels."""
        assert V.shape[0] == VL.shape[0], "Number of vertices and labels must match"
        
        unique_labels = np.unique(VL)
        colormap = plt.cm.get_cmap("tab20", len(unique_labels))
        label_to_color = {
            label: colormap(i)[:3] for i, label in enumerate(unique_labels)
        }
        
        VL = np.squeeze(VL)
        colors = np.array([label_to_color[label] for label in VL])
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(V)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        o3d.io.write_point_cloud(filename, pcd)
