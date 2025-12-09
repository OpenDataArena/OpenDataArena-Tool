from .base_scorer import BaseScorer
from .utils import get_total_lines
from typing import Dict, List
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import numpy as np
import os


# Helper function for multiprocessing (must be at module level for pickling)
def _compute_cosine_similarity_row(args):
    """Helper function to compute a single row of cosine similarity matrix
    
    Args:
        args: Tuple of (row_idx, embeddings, norms)
    
    Returns:
        Tuple of (row_idx, similarity_row)
    """
    row_idx, embeddings, norms = args
    num_samples = embeddings.shape[0]
    
    # Compute cosine similarity for this row with all other samples
    # cosine_sim(i, j) = dot(emb[i], emb[j]) / (norm[i] * norm[j])
    row_embedding = embeddings[row_idx]
    row_norm = norms[row_idx]
    
    # Vectorized computation for the entire row
    dot_products = embeddings @ row_embedding  # [num_samples]
    similarity_row = dot_products / (norms * row_norm)
    
    # Ensure diagonal is exactly 1.0
    similarity_row[row_idx] = 1.0
    
    return (row_idx, similarity_row)


class LogDetDistanceScorer(BaseScorer):
    """Log-Det Distance Scorer - Measures dataset diversity using log-determinant of cosine similarity matrix
    
    Log-Det Distance = log(det(S))
    
    where S is the cosine similarity matrix. This metric measures dataset diversity:
    - Higher log-det values indicate more diverse data
    - Lower log-det values indicate high similarity and low diversity
    
    Important: The similarity matrix should be positive semi-definite to ensure det(S) >= 0,
    making log(det) mathematically valid.
    """
    
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
        
        # Numerical stability: regularization term added to diagonal (prevents singular matrix)
        if "ridge_alpha" not in self.config:
            self.config["ridge_alpha"] = 1e-10
        else:
            # Ensure ridge_alpha is float (handles YAML parsing scientific notation as string)
            self.config["ridge_alpha"] = float(self.config["ridge_alpha"])
        
        print(f"Ridge regularization alpha: {self.config['ridge_alpha']}")

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
        
        print("Setting up LogDetDistanceScorer successfully")

    def score_item(self, data_item: Dict) -> float:
        """LogDetDistanceScorer computes a single score for the entire dataset, not individual samples"""
        raise NotImplementedError(
            "LogDetDistanceScorer computes a single score for the entire dataset. "
            "Use evaluate() method instead."
        )

    def compute_cosine_similarity_matrix_vectorized(self, embeddings):
        """Compute cosine similarity matrix using vectorized method (efficient)
        
        For N samples, this is 100-1000x faster than loop-based computation.
        
        Args:
            embeddings: numpy array of shape [num_samples, embedding_dim]
            
        Returns:
            similarity_matrix: numpy array of shape [num_samples, num_samples]
        """
        num_samples = embeddings.shape[0]
        
        print(f"Computing cosine similarity matrix using vectorized method ({num_samples}x{num_samples})...")
        
        # Compute norms for all embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)  # [N, 1]
        
        # Avoid division by zero
        norms = np.maximum(norms, 1e-10)
        
        # Normalize embeddings
        normalized_embeddings = embeddings / norms  # [N, D]
        
        # Compute cosine similarity matrix: normalized_embeddings @ normalized_embeddings.T
        similarity_matrix = normalized_embeddings @ normalized_embeddings.T  # [N, N]
        
        # Ensure diagonal is exactly 1.0 (may have floating point errors)
        np.fill_diagonal(similarity_matrix, 1.0)
        
        return similarity_matrix

    def compute_cosine_similarity_matrix_parallel(self, embeddings):
        """Compute cosine similarity matrix using parallel processing
        
        Args:
            embeddings: numpy array of shape [num_samples, embedding_dim]
            
        Returns:
            similarity_matrix: numpy array of shape [num_samples, num_samples]
        """
        num_samples = embeddings.shape[0]
        max_workers = self.config.get('max_workers', 1)
        
        print(f"Computing cosine similarity matrix using parallel method with {max_workers} worker(s) ({num_samples}x{num_samples})...")
        
        # Compute norms for all embeddings
        norms = np.linalg.norm(embeddings, axis=1)  # [N]
        
        # Avoid division by zero
        norms = np.maximum(norms, 1e-10)
        
        # Prepare tasks
        tasks = [(i, embeddings, norms) for i in range(num_samples)]
        
        # Initialize result matrix
        similarity_matrix = np.zeros((num_samples, num_samples))
        
        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Use tqdm to display progress bar
            results = list(tqdm(
                executor.map(_compute_cosine_similarity_row, tasks),
                total=num_samples,
                desc="Computing similarity matrix"
            ))
        
        # Assemble the matrix from results
        for row_idx, similarity_row in results:
            similarity_matrix[row_idx] = similarity_row
        
        return similarity_matrix

    def compute_cosine_similarity_matrix(self, embeddings):
        """Compute cosine similarity matrix (automatically choose method)
        
        For smaller datasets, vectorized method is faster.
        For larger datasets, parallel method may be more efficient.
        """
        num_samples = embeddings.shape[0]
        
        # Use vectorized method for smaller datasets (< 5000 samples)
        # Use parallel method for larger datasets
        if num_samples < 5000:
            return self.compute_cosine_similarity_matrix_vectorized(embeddings)
        else:
            return self.compute_cosine_similarity_matrix_parallel(embeddings)

    def evaluate(self, dataset) -> List[Dict]:
        """Evaluate the entire dataset and compute Log-Det as diversity metric
        
        Log-Det Distance = log(det(S))
        
        where S is the cosine similarity matrix.
        
        Args:
            dataset: Dataset file path (jsonl format)
        
        Returns:
            List containing a single dictionary with log-det score and related statistics
        """
        num_lines = get_total_lines(dataset)
        
        # Validate dataset line count matches embedding count
        if num_lines != self.num_data:
            print(f"Warning: Dataset has {num_lines} lines but embeddings have {self.num_data} rows.")
            print("Will use min(num_lines, num_data) for processing.")
            num_to_use = min(num_lines, self.num_data)
            embeddings_to_use = self.embeddings[:num_to_use]
        else:
            embeddings_to_use = self.embeddings
        
        print(f"\nComputing Log-Det Distance for {embeddings_to_use.shape[0]} samples...")
        print(f"Embedding dimension: {self.embedding_size}")
        print(f"Similarity metric: cosine similarity")
        
        # Compute cosine similarity matrix
        similarity_matrix = self.compute_cosine_similarity_matrix(embeddings_to_use)
        
        # Add regularization term for numerical stability (Ridge regularization)
        ridge_alpha = self.config["ridge_alpha"]
        if ridge_alpha > 0:
            similarity_matrix += ridge_alpha * np.eye(similarity_matrix.shape[0])
            print(f"Added ridge regularization: alpha={ridge_alpha}")
        
        # Compute statistics of similarity matrix
        matrix_min = np.min(similarity_matrix)
        matrix_max = np.max(similarity_matrix)
        matrix_mean = np.mean(similarity_matrix)
        matrix_std = np.std(similarity_matrix)
        
        # Diagonal mean (should be close to 1 for cosine similarity)
        diag_mean = np.mean(np.diag(similarity_matrix))
        
        print(f"\nCosine similarity matrix statistics:")
        print(f"  Min: {matrix_min:.6f}, Max: {matrix_max:.6f}")
        print(f"  Mean: {matrix_mean:.6f}, Std: {matrix_std:.6f}")
        print(f"  Diagonal mean: {diag_mean:.6f}")
        
        # Compute eigenvalues to check positive definiteness
        print("\nComputing eigenvalues to check positive definiteness...")
        eigenvalues = np.linalg.eigvalsh(similarity_matrix)  # eigvalsh for symmetric matrices, faster
        min_eigenvalue = np.min(eigenvalues)
        max_eigenvalue = np.max(eigenvalues)
        num_negative_eigenvalues = np.sum(eigenvalues < -1e-10)  # Tolerate small numerical errors
        
        print(f"Eigenvalue range: [{min_eigenvalue:.6e}, {max_eigenvalue:.6e}]")
        print(f"Number of negative eigenvalues: {num_negative_eigenvalues}")
        
        is_positive_definite = min_eigenvalue > 0
        is_positive_semidefinite = min_eigenvalue >= -1e-10
        
        if not is_positive_semidefinite:
            print(f"Warning: Similarity matrix is NOT positive semi-definite!")
            print(f"This may affect the validity of log-determinant computation.")
        
        # Use slogdet to compute log(det) - numerically stable method
        print("\nComputing log-determinant...")
        try:
            sign, logdet = np.linalg.slogdet(similarity_matrix)
            
            print(f"Sign of determinant: {sign}")
            print(f"Log(abs(det)): {logdet}")
            
            if sign > 0:
                log_det = logdet
                print(f"✓ Log-Det Distance: {log_det:.6f}")
                is_valid = True
                warning_message = None
            elif sign == 0:
                log_det = -np.inf
                is_valid = False
                warning_message = "Determinant is zero (singular matrix). Log-Det set to -inf."
                print(f"✗ Warning: {warning_message}")
            else:  # sign < 0
                log_det = logdet  # Still return log(abs(det))
                is_valid = False
                warning_message = (
                    f"Determinant is negative (sign={sign}). "
                    f"This should not happen with a valid similarity matrix. "
                    f"Returning log(abs(det))."
                )
                print(f"✗ Warning: {warning_message}")
                
        except np.linalg.LinAlgError as e:
            log_det = None
            sign = None
            logdet = None
            is_valid = False
            warning_message = f"Failed to compute determinant: {str(e)}"
            print(f"✗ Error: {warning_message}")
        
        # Build result
        result = {
            "log_det": float(log_det) if log_det is not None and not np.isinf(log_det) else None,
            "sign": int(sign) if sign is not None else None,
            "is_valid": is_valid,
            "is_positive_definite": bool(is_positive_definite),
            "is_positive_semidefinite": bool(is_positive_semidefinite),
            "num_samples": embeddings_to_use.shape[0],
            "embedding_dimension": self.embedding_size,
            "similarity_metric": "cosine",
            "eigenvalue_stats": {
                "min": float(min_eigenvalue),
                "max": float(max_eigenvalue),
                "num_negative": int(num_negative_eigenvalues)
            },
            "similarity_matrix_stats": {
                "min": float(matrix_min),
                "max": float(matrix_max),
                "mean": float(matrix_mean),
                "std": float(matrix_std),
                "diagonal_mean": float(diag_mean)
            }
        }
        
        if warning_message:
            result["warning"] = warning_message
        
        if log_det is not None and np.isinf(log_det):
            result["log_det_is_inf"] = True
            result["log_det"] = str(log_det)
        
        print("\n" + "="*60)
        print("Evaluation completed!")
        print("="*60)
        
        # Return a list containing a single result dictionary to match base class interface
        return [result]


