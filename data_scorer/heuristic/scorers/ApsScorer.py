from .base_scorer import BaseScorer
from .utils import get_total_lines, get_similarity_function
from typing import Dict
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import random


# Helper function for multiprocessing (must be at module level for pickling)
def _compute_similarity_pair(args):
    """Compute similarity for a single pair of embeddings
    
    Args:
        args: Tuple of (emb1, emb2, similarity_metric)
    
    Returns:
        float: Similarity score
    """
    emb1, emb2, similarity_metric = args
    
    # Get similarity function
    similarity_func = get_similarity_function(similarity_metric)
    
    # Compute similarity
    return similarity_func(emb1, emb2)


class ApsScorer(BaseScorer):
    def _validate_config(self):
        """Validate configuration parameters"""
        if "embedding_path" not in self.config:
            raise ValueError("embedding_path is required in config.")
        
        embedding_path = self.config["embedding_path"]
        if not os.path.exists(embedding_path):
            raise FileNotFoundError(f"Embedding file not found: {embedding_path}")
        
        if not embedding_path.endswith('.npy'):
            print(f"Warning: Embedding file should be a .npy file, but got: {embedding_path}")
        
        # Validate similarity metric parameters
        if "similarity_metric" not in self.config:
            self.config["similarity_metric"] = "cosine"
            print("Warning: No similarity_metric specified, using default 'cosine'.")
        else:
            valid_metrics = ["cosine", "euclidean", "manhattan", "dot_product", "pearson"]
            if self.config["similarity_metric"] not in valid_metrics:
                print(
                    f"Warning: Invalid similarity_metric '{self.config['similarity_metric']}', "
                    f"using default 'cosine'. Available metrics: {', '.join(valid_metrics)}"
                )
                self.config["similarity_metric"] = "cosine"
            else:
                print(f"Using specified similarity_metric: {self.config['similarity_metric']}")
        
        # Multi-processing worker count validation
        if "max_workers" in self.config and isinstance(self.config["max_workers"], int) and self.config["max_workers"] > 0:
            print(f"Using specified max_workers: {self.config['max_workers']}.")
        else:
            # Default to CPU count
            default_workers = max(1, os.cpu_count() or 1)
            print(f"Warning: No/invalid max_workers, using default value of {default_workers} (CPU count).")
            self.config['max_workers'] = default_workers
        
        # Sample pairs validation (for large datasets)
        if "sample_pairs" in self.config:
            if self.config["sample_pairs"] is None:
                print("sample_pairs is None, will compute all pairs.")
            elif isinstance(self.config["sample_pairs"], int) and self.config["sample_pairs"] > 0:
                print(f"Using specified sample_pairs: {self.config['sample_pairs']} (will randomly sample this many pairs).")
            else:
                print(f"Warning: Invalid sample_pairs value '{self.config['sample_pairs']}', will compute all pairs.")
                self.config["sample_pairs"] = None
        else:
            self.config["sample_pairs"] = None
            print("No sample_pairs specified, will compute all pairs by default.")

    def _setup(self):
        """Load embedding file and similarity function"""
        embedding_path = self.config["embedding_path"]
        print(f"Loading embeddings from: {embedding_path}")
        
        self.embeddings = np.load(embedding_path)
        print(f"Embeddings loaded. Shape: {self.embeddings.shape}")
        
        num_data, embedding_size = self.embeddings.shape
        self.num_data = num_data
        self.embedding_size = embedding_size
        
        # Get similarity calculation function
        similarity_metric = self.config["similarity_metric"]
        self.similarity_func = get_similarity_function(similarity_metric)
        print(f"Using similarity metric: {similarity_metric}")
        
        print("Setting up ApsScorer successfully")

    def score_item(self, data_item: Dict) -> float:
        """ApsScorer computes scores for the entire dataset, not for individual samples"""
        raise NotImplementedError(
            "ApsScorer computes a single score for the entire dataset. "
            "Use evaluate() method instead."
        )

    def evaluate(self, dataset) -> Dict:
        """Evaluate the entire dataset and compute average pairwise similarity (APS)
        
        Args:
            dataset: Path to dataset file (jsonl format)
        
        Returns:
            Dictionary containing APS score
        """
        num_lines = get_total_lines(dataset)
        
        # Verify if dataset line count matches embedding count
        if num_lines != self.num_data:
            print(f"Warning: Dataset has {num_lines} lines but embeddings have {self.num_data} rows.")
            print("Will use min(num_lines, num_data) for processing.")
            num_to_use = min(num_lines, self.num_data)
            embeddings_to_use = self.embeddings[:num_to_use]
        else:
            embeddings_to_use = self.embeddings
        
        print(f"Computing APS for {embeddings_to_use.shape[0]} samples...")
        print(f"Using similarity metric: {self.config['similarity_metric']}")
        
        n = embeddings_to_use.shape[0]
        
        # Handle edge case: less than 2 samples
        if n < 2:
            print(f"Warning: Dataset has {n} samples. At least 2 samples are required to compute APS.")
            return {
                "score": 0.0 if n == 0 else None,  # Return 0.0 for 0 samples, None for 1 sample
                "num_samples": n,
                "num_pairs": 0,
                "similarity_metric": self.config["similarity_metric"],
                "warning": f"Insufficient samples (found {n}, need at least 2)"
            }
        
        # Compute pairwise similarities using parallel processing
        total_pairs = n * (n - 1) // 2
        max_workers = self.config.get('max_workers', 1)
        print(f"Using {max_workers} worker(s) for parallel processing")
        print(f"Total possible pairs: {total_pairs}")
        
        # Prepare all pair combinations with necessary parameters
        similarity_metric = self.config["similarity_metric"]
        sample_pairs = self.config.get("sample_pairs")
        
        # Generate all possible pair indices
        all_pair_indices = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        # Determine whether to sample
        if sample_pairs is not None and sample_pairs < total_pairs:
            print(f"Sampling {sample_pairs} pairs from {total_pairs} total pairs...")
            sampled_indices = random.sample(all_pair_indices, sample_pairs)
            pair_tasks = [(embeddings_to_use[i], embeddings_to_use[j], similarity_metric) 
                         for i, j in sampled_indices]
            actual_pairs = sample_pairs
            is_sampled = True
        else:
            if sample_pairs is not None and sample_pairs >= total_pairs:
                print(f"sample_pairs ({sample_pairs}) >= total_pairs ({total_pairs}), will compute all pairs.")
            print("Computing all pairs...")
            pair_tasks = [(embeddings_to_use[i], embeddings_to_use[j], similarity_metric) 
                         for i, j in all_pair_indices]
            actual_pairs = total_pairs
            is_sampled = False
        
        # Parallel computation using ProcessPoolExecutor
        pairwise_similarities = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Use tqdm to show progress
            desc = f"Computing {'sampled ' if is_sampled else ''}pairwise similarities"
            results = list(tqdm(
                executor.map(_compute_similarity_pair, pair_tasks),
                total=actual_pairs,
                desc=desc
            ))
            pairwise_similarities = results
        
        # Compute average
        aps_score = np.mean(pairwise_similarities).item()
        
        if is_sampled:
            print(f"APS Score (sampled): {aps_score}")
            print(f"Total pairs computed: {len(pairwise_similarities)} (sampled from {total_pairs} possible pairs)")
        else:
            print(f"APS Score: {aps_score}")
            print(f"Total pairs computed: {len(pairwise_similarities)}")
        
        result = {
            "score": aps_score,
            "num_samples": embeddings_to_use.shape[0],
            "num_pairs": len(pairwise_similarities),
            "total_possible_pairs": total_pairs,
            "is_sampled": is_sampled,
            "similarity_metric": self.config["similarity_metric"],
            "max_workers": max_workers
        }
        
        # If sampled, add sampling info
        if is_sampled:
            result["sample_pairs"] = sample_pairs
        
        return result
