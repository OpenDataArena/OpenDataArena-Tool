import json
from typing import Dict, List
import numpy as np
from tqdm import tqdm
from .utils import get_total_lines
from concurrent.futures import ProcessPoolExecutor
from sklearn.neighbors import NearestNeighbors
import os
from .base_scorer import BaseScorer


# Global variables for multiprocessing (shared across worker processes)
_global_embeddings = None
_global_knn_model = None
_global_adjusted_k = None
_global_num_data = None


def _init_worker(embeddings, knn_model, adjusted_k, num_data):
    """Initialize worker process with shared data
    
    This function is called once when each worker process is created.
    It sets up the global variables that will be used by all tasks in this worker.
    """
    global _global_embeddings, _global_knn_model, _global_adjusted_k, _global_num_data
    _global_embeddings = embeddings
    _global_knn_model = knn_model
    _global_adjusted_k = adjusted_k
    _global_num_data = num_data


def _process_single_line(args):
    """Helper function to process a single line (for multiprocessing)
    
    Args:
        args: Tuple of (line, embedding_idx)
    
    Returns:
        Dict containing id and KNN_Score
    """
    line, embedding_idx = args
    
    try:
        item = json.loads(line.strip())
        
        # Validate embedding index
        if embedding_idx < 0 or embedding_idx >= _global_num_data:
            return {
                "id": item.get("id", ""),
                "score": 0.0
            }
        
        # Get query embedding for current data point
        query_embedding = _global_embeddings[embedding_idx:embedding_idx+1]
        
        # Find k+1 nearest neighbors (including itself, so we need to exclude the first result)
        k = _global_adjusted_k
        distances, indices = _global_knn_model.kneighbors(query_embedding, n_neighbors=k + 1)
        
        # Exclude itself (first nearest neighbor is itself), get distances of k nearest neighbors
        k_neighbor_distances = distances[0][1:k+1]
        
        # Calculate average K-nearest neighbor distance
        if len(k_neighbor_distances) == 0:
            score = 0.0
        else:
            score = float(np.mean(k_neighbor_distances))
        
        return {
            "id": item.get("id", ""),
            "score": score
        }
    except Exception as e:
        # If processing fails, return a result with error marker
        return {
            "id": item.get("id", "unknown") if 'item' in locals() else "unknown",
            "score": 0.0,
            "error": str(e)
        }


class KNNScorer(BaseScorer):
    def _validate_config(self):
        """Validate configuration parameters"""
        if "k" not in self.config or not isinstance(self.config["k"], int) or self.config["k"] <= 0:
            self.config["k"] = 5
            print("Warning: No/invalid k specified, use default value of 5.")
        else:
            print(f"Using specified k: {self.config['k']}.")

        if "embedding_path" not in self.config:
            raise ValueError("embedding_path is required in config.")
        
        embedding_path = self.config["embedding_path"]
        if not os.path.exists(embedding_path):
            raise FileNotFoundError(f"Embedding file not found: {embedding_path}")
        
        if not embedding_path.endswith('.npy'):
            print(f"Warning: Embedding file should be a .npy file, but got: {embedding_path}")

        if "max_workers" not in self.config or not isinstance(self.config["max_workers"], int) or self.config["max_workers"] <= 0:
            # Default to CPU core count
            default_workers = max(1, os.cpu_count() or 1)
            self.config["max_workers"] = default_workers
            print(f"Warning: No/invalid max_workers, use default value of {default_workers} (CPU count).")
        else:
            print(f"Using specified max_workers: {self.config['max_workers']}.")

        if "distance_metric" not in self.config:
            self.config["distance_metric"] = "euclidean"
            print("Warning: No distance_metric specified, use default value of 'euclidean'.")
        else:
            valid_metrics = ["euclidean", "cosine", "manhattan"]
            if self.config["distance_metric"] not in valid_metrics:
                print(f"Warning: Invalid distance_metric '{self.config['distance_metric']}', use default value of 'euclidean'.")
                self.config["distance_metric"] = "euclidean"
            else:
                print(f"Using specified distance_metric: {self.config['distance_metric']}.")

    def _setup(self):
        """Load embedding file and build KNN model"""
        embedding_path = self.config["embedding_path"]
        print(f"Loading embeddings from: {embedding_path}")
        
        self.embeddings = np.load(embedding_path)
        print(f"Embeddings loaded. Shape: {self.embeddings.shape}")
        
        # Validate embedding dimensions
        if len(self.embeddings.shape) != 2:
            raise ValueError(f"Embeddings should be 2D array, but got shape: {self.embeddings.shape}")
        
        num_data, embedding_size = self.embeddings.shape
        
        if num_data == 0:
            raise ValueError("Embeddings array is empty (0 samples)")
        if embedding_size == 0:
            raise ValueError("Embeddings have 0 dimensions")
        
        self.num_data = num_data
        self.embedding_size = embedding_size
        
        # Build KNN model
        k = self.config["k"]
        # If k >= num_data, use num_data - 1 (excluding itself)
        if k >= num_data:
            k = num_data - 1
            print(f"Warning: k ({self.config['k']}) >= num_data ({num_data}), using k={k}")
        
        # Save adjusted k value to avoid repeated calculation in score_item
        self.adjusted_k = k
        
        metric = self.config["distance_metric"]
        
        print(f"Building KNN model with k={k}, metric={metric}...")
        self.knn_model = NearestNeighbors(n_neighbors=k + 1, metric=metric, n_jobs=-1)
        self.knn_model.fit(self.embeddings)
        
        print("Setting up KNNScorer successfully")

    def score_item(self, data_item: Dict, embedding_idx: int) -> float:
        """Calculate K-nearest neighbor distance for a single data item
        
        Args:
            data_item: Data item dictionary
            embedding_idx: Index of this data item in the embedding matrix (starting from 0)
        
        Returns:
            Average K-nearest neighbor distance
        """
        if embedding_idx < 0 or embedding_idx >= self.num_data:
            print(f"Warning: Invalid embedding_idx {embedding_idx}, returning 0.0")
            return 0.0
        
        # Get query embedding for current data point
        query_embedding = self.embeddings[embedding_idx:embedding_idx+1]
        
        # Use adjusted k saved in _setup to avoid repeated calculation
        k = self.adjusted_k
        
        distances, indices = self.knn_model.kneighbors(query_embedding, n_neighbors=k + 1)
        
        # Exclude itself (first nearest neighbor is itself), get distances of k nearest neighbors
        k_neighbor_distances = distances[0][1:k+1]
        
        # Calculate average K-nearest neighbor distance
        if len(k_neighbor_distances) == 0:
            return 0.0
        else:
            return float(np.mean(k_neighbor_distances))

    def evaluate(self, dataset) -> List[Dict]:
        """Evaluate the entire dataset"""
        num_lines = get_total_lines(dataset)
        
        # Validate that dataset line count matches embedding count
        if num_lines != self.num_data:
            raise ValueError(
                f"Dataset size mismatch: Dataset has {num_lines} lines but embeddings have {self.num_data} rows. "
                f"Please ensure the dataset and embeddings are aligned and have the same number of samples."
            )
        
        max_workers = self.config.get('max_workers', 1)
        
        print(f"Using {max_workers} worker(s) for parallel processing")
        
        # Read all lines and prepare tasks
        with open(dataset, 'r', encoding='utf-8') as f:
            lines = [line for line in f]
        
        # Prepare task parameters: (line, embedding_idx)
        tasks = [(line, idx) for idx, line in enumerate(lines)]
        
        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_worker,
            initargs=(self.embeddings, self.knn_model, self.adjusted_k, self.num_data)
        ) as executor:
            # Use tqdm to display progress bar
            results = list(tqdm(
                executor.map(_process_single_line, tasks),
                total=num_lines,
                desc=self.config.get('name', 'KNNScorer')
            ))
        
        return results
