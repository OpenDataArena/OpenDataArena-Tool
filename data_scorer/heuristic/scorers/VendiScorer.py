from .base_scorer import BaseScorer
from .Vendi_Score import vendi
from .utils import get_total_lines, get_similarity_function
from typing import Dict, List
import numpy as np
import os
import json
from concurrent.futures import ProcessPoolExecutor


class VendiScorer(BaseScorer):
    def _validate_config(self):
        """Validate configuration parameters"""
        if "embedding_path" not in self.config:
            raise ValueError("embedding_path is required in config.")
        
        embedding_path = self.config["embedding_path"]
        if not os.path.exists(embedding_path):
            raise FileNotFoundError(f"Embedding file not found: {embedding_path}")
        
        if not embedding_path.endswith('.npy'):
            print(f"Warning: Embedding file should be a .npy file, but got: {embedding_path}")
        
        # Validate similarity metric parameter
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
        
        # Check max_workers (process count)
        if "max_workers" not in self.config or not isinstance(self.config["max_workers"], int) or self.config["max_workers"] <= 0:
            # Default to CPU core count
            default_workers = max(1, os.cpu_count() or 1)
            print(f"Warning: No/invalid max_workers, using default value of {default_workers} (CPU count).")
            self.config['max_workers'] = default_workers
        else:
            print(f"Using specified max_workers: {self.config['max_workers']}.")

    def _setup(self):
        """Load embedding file and similarity function"""
        embedding_path = self.config["embedding_path"]
        print(f"Loading embeddings from: {embedding_path}")
        
        self.embeddings = np.load(embedding_path)
        print(f"Embeddings loaded. Shape: {self.embeddings.shape}")
        
        num_data, embedding_size = self.embeddings.shape
        self.num_data = num_data
        self.embedding_size = embedding_size
        
        # Get similarity computation function
        similarity_metric = self.config["similarity_metric"]
        self.similarity_func = get_similarity_function(similarity_metric)
        print(f"Using similarity metric: {similarity_metric}")
        
        print("Setting up VendiScorer successfully")

    def score_item(self, data_item: Dict) -> float:
        """VendiScorer computes a single score for the entire dataset, not for individual samples"""
        raise NotImplementedError(
            "VendiScorer computes a single score for the entire dataset. "
            "Use evaluate() method instead."
        )

    def convert_2dnp_to_list_of_lists(self, np_array):
        """Convert 2D numpy array to Python list of lists"""
        return [row.tolist() for row in np_array]

    def evaluate(self, dataset) -> Dict:
        """Evaluate the entire dataset and return a global Vendi score
        
        Args:
            dataset: Path to dataset file (jsonl format)
        
        Returns:
            Dictionary containing Vendi score
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
        
        print(f"Computing Vendi Score for {embeddings_to_use.shape[0]} samples...")
        print(f"Using similarity metric: {self.config['similarity_metric']}")
        
        # Convert embeddings to list format (Vendi Score may require this format)
        embeddings_list = self.convert_2dnp_to_list_of_lists(embeddings_to_use)
        
        # Calculate Vendi Score using configured similarity function
        vendi_score = vendi.score(embeddings_list, self.similarity_func)
        
        print(f"Vendi Score: {vendi_score}")
        
        return {
            "vendi_score": vendi_score,
            "num_samples": embeddings_to_use.shape[0],
            "similarity_metric": self.config["similarity_metric"]
        }
