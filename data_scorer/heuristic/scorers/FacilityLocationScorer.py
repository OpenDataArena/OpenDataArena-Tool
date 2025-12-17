from .base_scorer import BaseScorer
from .utils import get_total_lines, get_distance_function
from typing import Dict
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


# Helper function for multiprocessing (must be at module level for pickling)
def _compute_min_distance(args):
    """Calculate the minimum distance from a single point to all points in the subset
    
    Args:
        args: Tuple of (point, subset_embeddings, distance_metric)
    
    Returns:
        float: Minimum distance
    """
    point, subset_embeddings, distance_metric = args
    
    # Re-obtain the distance function inside the function (because function objects cannot be pickled)
    distance_func = get_distance_function(distance_metric)
    
    min_dist = float('inf')
    for subset_point in subset_embeddings:
        dist = distance_func(point, subset_point)
        if dist < min_dist:
            min_dist = dist
    return min_dist


class FacilityLocationScorer(BaseScorer):
    def _validate_config(self):
        """Validate configuration parameters
        
        Configuration description:
        - embedding_path: embeddings of the full dataset (as reference background)
        - subset_embeddings_path: embeddings of the subset to be evaluated (corresponding to input_path)
        - Scoring objective: evaluate the coverage of the subset over the full dataset
        """
        # Validate full dataset embedding file (reference background)
        if "embedding_path" not in self.config:
            raise ValueError("embedding_path is required in config.")
        
        embedding_path = self.config["embedding_path"]
        if not os.path.exists(embedding_path):
            raise FileNotFoundError(f"Embedding file not found: {embedding_path}")
        
        if not embedding_path.endswith('.npy'):
            print(f"Warning: Embedding file should be a .npy file, but got: {embedding_path}")
        
        # Validate subset embedding file (the subject to be evaluated, corresponding to input_path)
        if "subset_embeddings_path" not in self.config:
            raise ValueError("subset_embeddings_path is required in config.")
        
        subset_embeddings_path = self.config["subset_embeddings_path"]
        if not os.path.exists(subset_embeddings_path):
            raise FileNotFoundError(f"Subset embeddings file not found: {subset_embeddings_path}")
        
        if not subset_embeddings_path.endswith('.npy'):
            print(f"Warning: Subset embeddings file should be a .npy file, but got: {subset_embeddings_path}")
        
        # Validate distance metric parameter (optional)
        if "distance_metric" not in self.config:
            self.config["distance_metric"] = "euclidean"
            print("Warning: No distance_metric specified, using default 'euclidean'.")
        else:
            valid_metrics = ["euclidean", "squared_euclidean", "manhattan", "cosine"]
            if self.config["distance_metric"] not in valid_metrics:
                print(
                    f"Warning: Invalid distance_metric '{self.config['distance_metric']}', "
                    f"using default 'euclidean'. Available metrics: {', '.join(valid_metrics)}"
                )
                self.config["distance_metric"] = "euclidean"
            else:
                print(f"Using specified distance_metric: {self.config['distance_metric']}")
        
        # Validate multi-process worker count parameter (optional, used for parallel computation acceleration)
        if "max_workers" in self.config and isinstance(self.config["max_workers"], int) and self.config["max_workers"] > 0:
            print(f"Using specified max_workers: {self.config['max_workers']}.")
        else:
            # Default to using CPU core count
            default_workers = max(1, os.cpu_count() or 1)
            print(f"Warning: No/invalid max_workers, using default value of {default_workers} (CPU count).")
            self.config['max_workers'] = default_workers

    def _setup(self):
        """Load full dataset embeddings and subset embeddings to be evaluated
        
        Logic description:
        - embeddings_all (X_all): full dataset, as reference background
        - subset_embeddings (X): subset to be evaluated (corresponding to input_path)
        - Computation: minimum distance from each point in the full dataset to the subset, evaluating subset coverage quality
        """
        # Load full dataset embeddings (X_all) - reference background
        embedding_path = self.config["embedding_path"]
        print(f"Loading full dataset embeddings (background) from: {embedding_path}")
        self.embeddings_all = np.load(embedding_path)
        print(f"Full dataset embeddings loaded. Shape: {self.embeddings_all.shape}")
        
        num_data, embedding_size = self.embeddings_all.shape
        self.num_data = num_data
        self.embedding_size = embedding_size
        
        # Load subset embeddings to be evaluated (X) - corresponding to input_path
        subset_embeddings_path = self.config["subset_embeddings_path"]
        print(f"Loading subset embeddings (to be evaluated) from: {subset_embeddings_path}")
        self.subset_embeddings = np.load(subset_embeddings_path)
        print(f"Subset embeddings loaded. Shape: {self.subset_embeddings.shape}")
        
        subset_size, subset_dim = self.subset_embeddings.shape
        
        # Verify if the dimension of the subset matches the dimension of the full dataset
        if subset_dim != embedding_size:
            raise ValueError(
                f"Dimension mismatch: full dataset embeddings have dimension {embedding_size}, "
                f"but subset embeddings have dimension {subset_dim}"
            )
        
        self.num_subset = self.subset_embeddings.shape[0]
        
        # Get distance metric and computation function
        distance_metric = self.config["distance_metric"]
        self.distance_metric = distance_metric
        self.distance_func = get_distance_function(distance_metric)
        print(f"Using distance metric: {distance_metric}")
        
        # Get parallel worker count
        self.max_workers = self.config["max_workers"]
        print(f"Using max_workers: {self.max_workers}")
        
        print("Setting up FacilityLocationScorer successfully")

    def score_item(self, data_item: Dict) -> float:
        """FacilityLocationScorer scores the entire dataset, not individual samples"""
        raise NotImplementedError(
            "FacilityLocationScorer computes a single score for the entire dataset. "
            "Use evaluate() method instead."
        )

    def evaluate(self, dataset) -> Dict:
        """Evaluate the quality of subset data and compute the Facility Location score
        
        Facility Location is defined as: the sum of distances from each point in the full dataset to the nearest point in the selected subset
        M_FL(X) = sum(min(distance(x_j, x_i) for x_i in X) for x_j in X_all)
        
        A lower score indicates that the selected subset has better coverage of the full dataset (as "facilities" better serve all "customers")
        A higher score indicates that the subset coverage is inadequate, with data points far from the subset
        
        Uses ProcessPoolExecutor to parallelize computation of minimum distances for acceleration
        
        Args:
            dataset: Subset data file path (jsonl format), corresponding to subset_embeddings_path
        
        Returns:
            Dictionary containing Facility Location score
        """
        num_lines = get_total_lines(dataset)
        
        # Verify if the number of lines in subset dataset matches the number of subset embeddings
        if num_lines != self.num_subset:
            print(f"Warning: Dataset has {num_lines} lines but subset embeddings have {self.num_subset} rows.")
            print(f"Expected dataset to match subset_embeddings_path.")
        
        # Use full dataset embeddings for computation
        embeddings_to_use = self.embeddings_all
        
        print(f"Evaluating subset quality using Facility Location metric...")
        print(f"Full dataset size: {self.num_data} samples")
        print(f"Subset size (to be evaluated): {self.num_subset} samples")
        print(f"Distance metric: {self.distance_metric}")
        print(f"Using {self.max_workers} worker(s) for parallel computation...")
        print(f"Computing minimum distances for {embeddings_to_use.shape[0]} full dataset samples to subset...")
        
        # Prepare task parameter list (each task contains: point, subset_embeddings, distance_metric)
        tasks = [(point, self.subset_embeddings, self.distance_metric) 
                 for point in embeddings_to_use]
        
        # Use ProcessPoolExecutor to parallelize computation of minimum distance for each point
        min_distances = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Use tqdm to display progress bar
            results = list(tqdm(
                executor.map(_compute_min_distance, tasks),
                total=len(tasks),
                desc="Computing min distances"
            ))
            min_distances = results
        
        # Convert to numpy array for statistics
        min_distances = np.array(min_distances)
        
        # Calculate various statistics
        total_min_distance = np.sum(min_distances)
        avg_min_distance = np.mean(min_distances)
        max_min_distance = np.max(min_distances)
        median_min_distance = np.median(min_distances)
        std_min_distance = np.std(min_distances)
        
        print(f"Total Facility Location score: {total_min_distance}")
        print(f"Average minimum distance: {avg_min_distance}")
        print(f"Maximum minimum distance: {max_min_distance}")
        print(f"Median minimum distance: {median_min_distance}")
        print(f"Std of minimum distances: {std_min_distance}")
        
        return {
            "facility_location_score": float(total_min_distance),
            "avg_min_distance": float(avg_min_distance),
            "max_min_distance": float(max_min_distance),
            "median_min_distance": float(median_min_distance),
            "std_min_distance": float(std_min_distance),
            "num_samples": embeddings_to_use.shape[0],
            "num_subset_samples": self.num_subset,
            "distance_metric": self.distance_metric,
            "subset_ratio": float(self.num_subset / embeddings_to_use.shape[0])
        }

