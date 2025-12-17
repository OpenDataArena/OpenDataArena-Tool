from .base_scorer import BaseScorer
from .utils import get_total_lines
from typing import Dict
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import numpy as np
import os


# Helper function for multiprocessing (must be at module level for pickling)
def _compute_dimension_stds(args):
    """Helper function to compute standard deviations for a chunk of dimensions (for multiprocessing)
    
    Args:
        args: Tuple of (embeddings, start_dim, end_dim)
    
    Returns:
        Array of standard deviations for the specified dimension range
    """
    embeddings, start_dim, end_dim = args
    
    try:
        # Compute standard deviation for the specified dimension range
        dimension_stds = np.std(embeddings[:, start_dim:end_dim], axis=0)
        return dimension_stds
    except Exception as e:
        # If computation fails, return zeros
        return np.zeros(end_dim - start_dim)


class RadiusScorer(BaseScorer):
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
        
        # Multiprocessing worker count validation
        if "max_workers" in self.config and isinstance(self.config["max_workers"], int) and self.config["max_workers"] > 0:
            print(f"Using specified max_workers: {self.config['max_workers']}.")
        else:
            # Default to CPU core count
            default_workers = max(1, os.cpu_count() or 1)
            print(f"Warning: No/invalid max_workers, using default value of {default_workers} (CPU count).")
            self.config['max_workers'] = default_workers

    def _setup(self):
        """Load embedding file"""
        # Load embeddings
        embedding_path = self.config["embedding_path"]
        print(f"Loading embeddings from: {embedding_path}")
        self.embeddings = np.load(embedding_path)
        print(f"Embeddings loaded. Shape: {self.embeddings.shape}")
        
        num_data, embedding_size = self.embeddings.shape
        self.num_data = num_data
        self.embedding_size = embedding_size
        
        print("Setting up RadiusScorer successfully")

    def score_item(self, data_item: Dict) -> float:
        """RadiusScorer computes a single score for the entire dataset, not for individual samples"""
        raise NotImplementedError(
            "RadiusScorer computes a single score for the entire dataset. "
            "Use evaluate() method instead."
        )

    def evaluate(self, dataset) -> Dict:
        """Evaluate the entire dataset, computing Radius as a diversity measure
        
        Radius is defined as: compute the standard deviation of all points on each embedding dimension,
        then take the geometric mean of all dimension standard deviations
        Radius = (std_1 * std_2 * ... * std_n)^(1/n)
        
        Higher Radius values indicate data is more widely distributed in embedding space, with higher diversity
        Lower Radius values indicate data is more concentrated, with lower diversity
        
        Args:
            dataset: Dataset file path (jsonl format)
        
        Returns:
            Dictionary containing Radius score
        """
        num_lines = get_total_lines(dataset)
        
        # Verify that dataset line count matches embedding count
        if num_lines != self.num_data:
            print(f"Warning: Dataset has {num_lines} lines but embeddings have {self.num_data} rows.")
            print("Will use min(num_lines, num_data) for processing.")
            num_to_use = min(num_lines, self.num_data)
            embeddings_to_use = self.embeddings[:num_to_use]
        else:
            embeddings_to_use = self.embeddings
        
        print(f"Computing Radius for {embeddings_to_use.shape[0]} samples...")
        print(f"Embedding dimension: {self.embedding_size}")
        
        max_workers = self.config.get('max_workers', 1)
        
        # Use parallel processing if max_workers > 1 and dimension is large enough
        if max_workers > 1 and self.embedding_size > max_workers * 10:
            print(f"Using {max_workers} worker(s) for parallel processing")
            
            # Split dimensions into chunks for parallel processing
            chunk_size = max(1, self.embedding_size // max_workers)
            tasks = []
            for i in range(0, self.embedding_size, chunk_size):
                end_dim = min(i + chunk_size, self.embedding_size)
                tasks.append((embeddings_to_use, i, end_dim))
            
            # Use ProcessPoolExecutor for parallel processing
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Use tqdm to display progress bar
                results = list(tqdm(
                    executor.map(_compute_dimension_stds, tasks),
                    total=len(tasks),
                    desc=self.config.get('name', 'RadiusScorer')
                ))
            
            # Concatenate results from all chunks
            dimension_stds = np.concatenate(results)
        else:
            # Compute standard deviation for each dimension (without parallelization)
            dimension_stds = np.std(embeddings_to_use, axis=0)
        
        # Check for zero standard deviations (dimensions where all values are the same)
        zero_std_dims = np.sum(dimension_stds == 0)
        if zero_std_dims > 0:
            print(f"Warning: Found {zero_std_dims} dimensions with zero standard deviation.")
            # To avoid geometric mean being 0, replace zeros with a very small number
            dimension_stds = np.where(dimension_stds == 0, 1e-10, dimension_stds)
        
        # Compute geometric mean: use logarithm trick to avoid numerical overflow
        # geometric_mean = exp(mean(log(x_i)))
        log_stds = np.log(dimension_stds)
        log_mean = np.mean(log_stds)
        radius = np.exp(log_mean)
        
        # Also compute arithmetic mean for reference
        arithmetic_mean_std = np.mean(dimension_stds)
        
        # Statistics
        min_std = np.min(dimension_stds)
        max_std = np.max(dimension_stds)
        median_std = np.median(dimension_stds)
        
        print(f"Radius (geometric mean of stds): {radius}")
        print(f"Arithmetic mean of stds: {arithmetic_mean_std}")
        print(f"Min std: {min_std}, Max std: {max_std}, Median std: {median_std}")
        
        return {
            "radius": float(radius),  # Radius
            "geometric_mean_std": float(radius),  # Geometric mean of standard deviations
            "arithmetic_mean_std": float(arithmetic_mean_std),  # Arithmetic mean of standard deviations
            "min_std": float(min_std),  # Minimum standard deviation
            "max_std": float(max_std),  # Maximum standard deviation
            "median_std": float(median_std),  # Median standard deviation
            "num_samples": embeddings_to_use.shape[0],  # Number of samples
            "embedding_dimension": self.embedding_size,  # Embedding dimension
            "zero_std_dimensions": int(zero_std_dims)  # Number of zero standard deviation dimensions
        }

