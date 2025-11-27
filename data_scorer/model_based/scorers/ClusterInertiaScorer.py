from .base_scorer import BaseScorer
from .utils import get_total_lines, get_distance_function
from typing import Dict
import numpy as np
import os


class ClusterInertiaScorer(BaseScorer):
    def _validate_config(self):
        """Validate configuration parameters"""
        # Validate embedding file
        if "embedding_path" not in self.config:
            raise ValueError("embedding_path is required in config.")
        
        embedding_path = self.config["embedding_path"]
        if not os.path.exists(embedding_path):
            raise FileNotFoundError(f"Embedding file not found: {embedding_path}")
        
        if not embedding_path.endswith('.npy'):
            print(f"Warning: Embedding file should be a .npy file, but got: {embedding_path}")
        
        # Validate cluster centroids file
        if "cluster_centroids_path" not in self.config:
            raise ValueError("cluster_centroids_path is required in config.")
        
        cluster_centroids_path = self.config["cluster_centroids_path"]
        if not os.path.exists(cluster_centroids_path):
            raise FileNotFoundError(f"Cluster centroids file not found: {cluster_centroids_path}")
        
        if not cluster_centroids_path.endswith('.npy'):
            print(f"Warning: Cluster centroids file should be a .npy file, but got: {cluster_centroids_path}")
        
        # Validate cluster labels file
        if "cluster_labels_path" not in self.config:
            raise ValueError("cluster_labels_path is required in config.")
        
        cluster_labels_path = self.config["cluster_labels_path"]
        if not os.path.exists(cluster_labels_path):
            raise FileNotFoundError(f"Cluster labels file not found: {cluster_labels_path}")
        
        if not cluster_labels_path.endswith('.npy'):
            print(f"Warning: Cluster labels file should be a .npy file, but got: {cluster_labels_path}")
        
        # Validate distance metric parameter (optional)
        if "distance_metric" not in self.config:
            self.config["distance_metric"] = "cosine"
            print("Warning: No distance_metric specified, using default 'cosine'.")
        else:
            valid_metrics = ["euclidean", "squared_euclidean", "manhattan", "cosine"]
            if self.config["distance_metric"] not in valid_metrics:
                print(
                    f"Warning: Invalid distance_metric '{self.config['distance_metric']}', "
                    f"using default 'cosine'. Available metrics: {', '.join(valid_metrics)}"
                )
                self.config["distance_metric"] = "cosine"
            else:
                print(f"Using specified distance_metric: {self.config['distance_metric']}")

    def _setup(self):
        """Load embedding file, cluster centroids, and cluster labels"""
        # Load embeddings
        embedding_path = self.config["embedding_path"]
        print(f"Loading embeddings from: {embedding_path}")
        self.embeddings = np.load(embedding_path)
        print(f"Embeddings loaded. Shape: {self.embeddings.shape}")
        
        num_data, embedding_size = self.embeddings.shape
        self.num_data = num_data
        self.embedding_size = embedding_size
        
        # Load cluster centroids
        cluster_centroids_path = self.config["cluster_centroids_path"]
        print(f"Loading cluster centroids from: {cluster_centroids_path}")
        self.cluster_centroids = np.load(cluster_centroids_path)
        print(f"Cluster centroids loaded. Shape: {self.cluster_centroids.shape}")
        
        num_clusters, centroid_dim = self.cluster_centroids.shape
        self.num_clusters = num_clusters
        
        # Validate that cluster centroids dimension matches embedding dimension
        if centroid_dim != embedding_size:
            raise ValueError(
                f"Dimension mismatch: embeddings have dimension {embedding_size}, "
                f"but cluster centroids have dimension {centroid_dim}"
            )
        
        # Load cluster labels
        cluster_labels_path = self.config["cluster_labels_path"]
        print(f"Loading cluster labels from: {cluster_labels_path}")
        self.cluster_labels = np.load(cluster_labels_path)
        print(f"Cluster labels loaded. Shape: {self.cluster_labels.shape}")
        
        # Validate that number of labels matches number of embeddings
        if len(self.cluster_labels) != num_data:
            raise ValueError(
                f"Number of labels ({len(self.cluster_labels)}) does not match "
                f"number of embeddings ({num_data})"
            )
        
        # Get distance metric and computation function
        distance_metric = self.config["distance_metric"]
        self.distance_metric = distance_metric
        self.distance_func = get_distance_function(distance_metric)
        print(f"Using distance metric: {distance_metric}")
        
        print("Setting up ClusterInertiaScorer successfully")

    def score_item(self, data_item: Dict) -> float:
        """ClusterInertiaScorer scores the entire dataset, not individual samples"""
        raise NotImplementedError(
            "ClusterInertiaScorer computes a single score for the entire dataset. "
            "Use evaluate() method instead."
        )

    def evaluate(self, dataset) -> Dict:
        """Evaluate the entire dataset and compute Cluster Inertia
        
        Cluster inertia is defined as: sum of distances from all samples to their assigned cluster centroids
        Inertia = sum(sum(distance(x, c_i) for x in cluster_i) for all clusters i)
        
        Lower inertia values indicate tighter clusters, while higher values indicate more dispersed data (higher diversity)
        
        Args:
            dataset: Path to dataset file (jsonl format)
        
        Returns:
            Dictionary containing Cluster Inertia scores
        """
        num_lines = get_total_lines(dataset)
        
        # Validate that dataset line count matches embedding count
        if num_lines != self.num_data:
            print(f"Warning: Dataset has {num_lines} lines but embeddings have {self.num_data} rows.")
            print("Will use min(num_lines, num_data) for processing.")
            num_to_use = min(num_lines, self.num_data)
            embeddings_to_use = self.embeddings[:num_to_use]
            labels_to_use = self.cluster_labels[:num_to_use]
        else:
            embeddings_to_use = self.embeddings
            labels_to_use = self.cluster_labels
        
        print(f"Computing Cluster Inertia for {embeddings_to_use.shape[0]} samples...")
        print(f"Number of clusters: {self.num_clusters}")
        print(f"Using distance metric: {self.distance_metric}")
        
        # Compute cluster inertia
        total_inertia = 0.0
        cluster_inertias = {}
        cluster_sizes = {}
        
        # Compute inertia for each cluster
        for cluster_id in range(self.num_clusters):
            # Get all points belonging to current cluster
            cluster_mask = labels_to_use == cluster_id
            cluster_points = embeddings_to_use[cluster_mask]
            cluster_size = len(cluster_points)
            
            if cluster_size == 0:
                cluster_inertias[cluster_id] = 0.0
                cluster_sizes[cluster_id] = 0
                continue
            
            # Get the centroid of current cluster
            centroid = self.cluster_centroids[cluster_id]
            
            # Compute inertia for this cluster (sum of distances from all points to centroid)
            cluster_inertia = 0.0
            for point in cluster_points:
                distance = self.distance_func(point, centroid)
                cluster_inertia += distance
            
            cluster_inertias[cluster_id] = cluster_inertia
            cluster_sizes[cluster_id] = cluster_size
            total_inertia += cluster_inertia
        
        # Compute average inertia
        avg_inertia = total_inertia / embeddings_to_use.shape[0]
        
        print(f"Total Cluster Inertia: {total_inertia}")
        print(f"Average Inertia per sample: {avg_inertia}")
        print(f"Cluster sizes: {cluster_sizes}")
        
        return {
            "total_inertia": float(total_inertia),
            "avg_inertia_per_sample": float(avg_inertia),
            "num_samples": embeddings_to_use.shape[0],
            "num_clusters": self.num_clusters,
            "distance_metric": self.distance_metric,
            "cluster_sizes": cluster_sizes,
            "cluster_inertias": {int(k): float(v) for k, v in cluster_inertias.items()}
        }
