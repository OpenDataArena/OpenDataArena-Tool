from .base_scorer import BaseScorer
from .utils import get_total_lines
from typing import Dict
import numpy as np
import json


class PartitionEntropyScorer(BaseScorer):
    def _validate_config(self):
        """Validate configuration parameters"""
        # Validate global cluster number
        if "num_clusters" not in self.config:
            raise ValueError("num_clusters (number of global clusters) is required in config.")
        
        if not isinstance(self.config["num_clusters"], int) or self.config["num_clusters"] <= 0:
            raise ValueError(f"num_clusters must be a positive integer, got: {self.config['num_clusters']}")
        
        print(f"Global number of clusters: {self.config['num_clusters']}")

    def _setup(self):
        """Initialize scorer"""
        # Get global cluster number
        self.num_clusters = self.config["num_clusters"]
        print(f"Number of global clusters: {self.num_clusters}")
        
        print("Setting up PartitionEntropyScorer successfully")

    def score_item(self, data_item: Dict) -> float:
        """PartitionEntropyScorer scores the entire dataset, not individual samples"""
        raise NotImplementedError(
            "PartitionEntropyScorer computes a single score for the entire dataset. "
            "Use evaluate() method instead."
        )

    def evaluate(self, dataset) -> Dict:
        """Evaluate the entire dataset and compute Partition Entropy
        
        This method calculates the distribution entropy of subset data across global clusters.
        
        Partition Entropy is defined as: H = -Σ(p_i * log(p_i))
        where p_i is the proportion of samples in the i-th cluster within the subset
        
        Higher entropy indicates more uniform distribution across global clusters, higher diversity
        Lower entropy indicates more concentrated distribution across global clusters, lower diversity
        
        Args:
            dataset: Dataset file path (jsonl format), each data item must contain cluster_id field
        
        Returns:
            Dictionary containing Partition Entropy score
        """
        num_lines = get_total_lines(dataset)
        print(f"Computing Partition Entropy for {num_lines} samples in subset...")
        print(f"Global number of clusters: {self.num_clusters}")
        
        # Read dataset and count samples in each cluster
        cluster_counts = {}
        total_samples = 0
        missing_cluster_id_count = 0
        
        with open(dataset, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                
                    cluster_id = int(item["cluster_id"])
                    
                    # Count cluster occurrences
                    cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
                    total_samples += 1
                    
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {line_num}: {e}")
                    continue
                except (KeyError, ValueError, TypeError) as e:
                    print(f"Warning: Failed to extract cluster_id from line {line_num}: {e}")
                    continue
        
        if missing_cluster_id_count > 0:
            print(f"Warning: {missing_cluster_id_count} samples missing cluster_id field")
        
        if total_samples == 0:
            print("Error: No valid samples found with cluster_id")
            return {
                "entropy": 0.0,
                "normalized_entropy": 0.0,
                "max_entropy": 0.0,
                "num_samples": 0,
                "num_clusters_global": self.num_clusters,
                "num_clusters_in_subset": 0,
                "cluster_counts": {},
                "cluster_probabilities": {}
            }
        
        print(f"Successfully loaded {total_samples} samples with cluster_id")
        print(f"Number of unique clusters in subset: {len(cluster_counts)}")
        
        # Calculate probability distribution for each cluster
        cluster_probabilities = {}
        for cluster_id, count in cluster_counts.items():
            cluster_probabilities[cluster_id] = count / total_samples
        
        # Calculate partition entropy
        # H = -Σ(p_i * log(p_i))
        # Note: Only calculate entropy for clusters that actually appear in the subset
        entropy = 0.0
        for cluster_id, prob in cluster_probabilities.items():
            if prob > 0:  # Avoid log(0)
                entropy -= prob * np.log(prob)
        
        # Calculate normalized entropy (divided by global maximum possible entropy log(num_clusters))
        # Use global cluster count to calculate maximum entropy, which reflects subset diversity relative to global clusters
        max_entropy = np.log(self.num_clusters) if self.num_clusters > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        print(f"Partition Entropy: {entropy:.4f}")
        print(f"Normalized Entropy: {normalized_entropy:.4f}")
        print(f"Max possible entropy (based on global clusters): {max_entropy:.4f}")
        print(f"Cluster distribution in subset: {cluster_counts}")
        
        return {
            "entropy": float(entropy),  # Partition entropy
            "normalized_entropy": float(normalized_entropy),  # Normalized partition entropy (relative to global cluster count)
            "max_entropy": float(max_entropy),  # Maximum possible partition entropy (based on global cluster count)
            "num_samples": total_samples,  # Number of samples in subset
            "num_clusters_global": self.num_clusters,  # Global cluster count
            "num_clusters_in_subset": len(cluster_counts),  # Number of clusters actually appearing in subset
            "cluster_counts": cluster_counts,  # Cluster counts
            "cluster_probabilities": {int(k): float(v) for k, v in cluster_probabilities.items()}  # Cluster probability distribution
        }

