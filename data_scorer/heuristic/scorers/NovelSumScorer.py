from .base_scorer import BaseScorer
from .utils import get_total_lines
from typing import Dict, List
import numpy as np
import os
from tqdm import tqdm
import random
import pathlib
import faiss
from concurrent.futures import ProcessPoolExecutor


def set_seed(seed: int) -> None:
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

class FaissIndex:
    def __init__(self, data: np.ndarray):
        self.data = data.astype(np.float32).reshape(-1, data.shape[-1])
        self.data = np.unique(self.data, axis=0)
        
        print(f"Creating Faiss index with {self.data.shape[0]} vectors, dimension {self.data.shape[1]}")
        print(f"Index data size: {self.data.nbytes / 1024 / 1024:.2f} MB")
        
        self.index = faiss.IndexFlatL2(self.data.shape[1])
        self.index.add(self.data)
        print("Faiss CPU index created successfully")

    def search(self, query: np.ndarray, n_neighbors: int):
        query = query.astype(np.float32)
        distances, indices = self.index.search(query, n_neighbors + 1)
        return distances[:, 1:], indices[:, 1:]

    def local_density(self, query_data: np.ndarray, n_neighbors=10, regularization=1e-9, power=1):
        distances, _ = self.search(query_data, n_neighbors)
        avg_distances = np.mean(distances, axis=-1)
        densities = 1 / np.power((avg_distances + regularization), power)
        return densities

def load_npy_data(file: str) -> np.ndarray:
    return np.load(file)

def load_dir_npy_data(npy_dir: str, desc: str = "Loading data") -> np.ndarray:
    print(f"Searching for .npy files in: {npy_dir}")
    files = sorted(pathlib.Path(npy_dir).glob("*.npy"))
    
    if not files:
        print(f"Warning: No .npy files found in {npy_dir}")
        return np.array([])
    
    print(f"Found {len(files)} .npy files")
    data_list = []
    for file in tqdm(files, desc=desc):
        print(f"  Loading: {file.name}")
        data = load_npy_data(str(file))
        print(f"    Shape: {data.shape}, Size: {data.nbytes / 1024 / 1024:.2f} MB")
        data_list.append(data)
    
    print("Concatenating all arrays...")
    result = np.concatenate(data_list, axis=0)
    print(f"Concatenation complete. Final shape: {result.shape}")
    return result

def _compute_cosine_distance_chunk(args):
    """Helper function to compute a chunk of cosine distance matrix (for multiprocessing)
    
    Args:
        args: Tuple of (data, start_idx, end_idx)
    
    Returns:
        Tuple of (start_idx, distance_chunk)
    """
    data, start_idx, end_idx = args
    
    # Normalize the data
    data_norm = data / (np.linalg.norm(data, axis=1, keepdims=True) + 1e-10)
    chunk_norm = data_norm[start_idx:end_idx]
    
    # Compute cosine similarity for the chunk
    cosine_similarity = np.dot(chunk_norm, data_norm.T)
    
    # Convert to distance
    cosine_distance = 1 - cosine_similarity
    
    return (start_idx, cosine_distance)


def compute_cos_distance(data: np.ndarray, max_workers: int = None) -> np.ndarray:
    """Compute cosine distance matrix using multiprocessing
    
    Args:
        data: Input data array of shape (n_samples, n_features)
        max_workers: Number of worker processes (default: CPU count)
    
    Returns:
        Cosine distance matrix of shape (n_samples, n_samples)
    """
    num_vectors = data.shape[0]
    
    # Determine chunk size based on number of workers
    if max_workers is None:
        max_workers = max(1, os.cpu_count() or 1)
    
    chunk_size = max(1, num_vectors // max_workers)
    
    # Prepare tasks
    tasks = []
    for start_idx in range(0, num_vectors, chunk_size):
        end_idx = min(start_idx + chunk_size, num_vectors)
        tasks.append((data, start_idx, end_idx))
    
    print(f"Computing cosine distance matrix using {len(tasks)} chunks with {max_workers} workers...")
    
    # Use ProcessPoolExecutor for parallel processing
    distance_matrix = np.zeros((num_vectors, num_vectors), dtype=np.float32)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(_compute_cosine_distance_chunk, tasks),
            total=len(tasks),
            desc="Computing distance"
        ))
    
    # Assemble results
    for start_idx, distance_chunk in results:
        end_idx = start_idx + distance_chunk.shape[0]
        distance_matrix[start_idx:end_idx, :] = distance_chunk
    
    return distance_matrix


def weighted_average(row: np.ndarray, power: float = 1.0) -> float:
    sorted_indices = np.argsort(row)
    weights = 1 / np.power(np.arange(1, len(row) + 1), power)
    sorted_row = row[sorted_indices]
    return np.average(sorted_row, weights=weights)


def novelsum(distance_matrix: np.ndarray, densities: np.ndarray, power: float = 1.0) -> float:
    weighted_matrix = distance_matrix * densities[:, np.newaxis]
    weighted_averages = np.apply_along_axis(weighted_average, 1, weighted_matrix, power=power)
    return np.mean(weighted_averages)


class NovelSumScorer(BaseScorer):
    def _validate_config(self):
        if "embedding_path" not in self.config:
            raise ValueError("embedding_path is required in config.")
        
        embedding_path = self.config["embedding_path"]
        if not os.path.exists(embedding_path):
            raise FileNotFoundError(f"Embedding file not found: {embedding_path}")
        
        if not embedding_path.endswith('.npy'):
            raise ValueError(f"Embedding file must be a .npy file, but got: {embedding_path}")
        
        if "dense_ref_path" not in self.config:
            self.config["dense_ref_path"] = os.path.dirname(embedding_path)
            print(f"Info: No dense_ref_path specified, using embedding directory: {self.config['dense_ref_path']}")
        else:
            dense_ref_path = self.config["dense_ref_path"]
            if not os.path.exists(dense_ref_path):
                raise FileNotFoundError(f"Dense reference path not found: {dense_ref_path}")
        
        if "max_workers" in self.config and isinstance(self.config["max_workers"], int) and self.config["max_workers"] > 0:
            print(f"Using specified max_workers: {self.config['max_workers']}.")
        else:
            default_workers = max(1, os.cpu_count() or 1)
            print(f"Info: No/invalid max_workers, using default value of {default_workers} (CPU count).")
            self.config['max_workers'] = default_workers
        
        if "density_powers" not in self.config:
            self.config["density_powers"] = [0, 0.25, 0.5]
            print(f"Info: No density_powers specified, using default {self.config['density_powers']}.")
        
        if "neighbors" not in self.config:
            self.config["neighbors"] = [5, 10]
            print(f"Info: No neighbors specified, using default {self.config['neighbors']}.")
        
        if "distance_powers" not in self.config:
            self.config["distance_powers"] = [0, 1, 2]
            print(f"Info: No distance_powers specified, using default {self.config['distance_powers']}.")

    def _setup(self):
        embedding_path = self.config["embedding_path"]
        print(f"Loading embeddings from: {embedding_path}")
        
        self.embeddings = np.load(embedding_path)
        print(f"Embeddings loaded. Shape: {self.embeddings.shape}")
        
        num_data, embedding_size = self.embeddings.shape
        self.num_data = num_data
        self.embedding_size = embedding_size
        
        dense_ref_path = self.config["dense_ref_path"]
        print(f"Loading dense reference data from: {dense_ref_path}")
        print(f"Path exists: {os.path.exists(dense_ref_path)}")
        print(f"Is directory: {os.path.isdir(dense_ref_path)}")
        
        if os.path.isdir(dense_ref_path):
            print("Loading from directory...")
            self.dense_ref_data = load_dir_npy_data(dense_ref_path, desc="Loading reference data")
        else:
            print("Loading from single file...")
            self.dense_ref_data = load_npy_data(dense_ref_path)
        
        print(f"Dense reference data loaded. Shape: {self.dense_ref_data.shape}")
        
        self.faiss_index = FaissIndex(self.dense_ref_data)
        print("Faiss index initialized successfully")
        
        self.density_powers = self.config["density_powers"]
        self.neighbors = self.config["neighbors"]
        self.distance_powers = self.config["distance_powers"]
        
        print("Setting up NovelSumScorer successfully")

    def score_item(self, data_item: Dict) -> float:
        raise NotImplementedError(
            "NovelSumScorer computes scores for the entire dataset. "
            "Use evaluate() method instead."
        )

    def evaluate(self, dataset) -> Dict:
        num_lines = get_total_lines(dataset)
        max_workers = self.config.get('max_workers', 1)
        
        if num_lines != self.num_data:
            print(f"Warning: Dataset has {num_lines} lines but embeddings have {self.num_data} rows.")
            print("Will use min(num_lines, num_data) for processing.")
            num_to_use = min(num_lines, self.num_data)
            embeddings_to_use = self.embeddings[:num_to_use]
        else:
            embeddings_to_use = self.embeddings
        
        print(f"Computing NovelSum for {embeddings_to_use.shape[0]} samples...")
        print(f"Using {max_workers} worker(s) for parallel processing")
        
        print("Computing cosine distance matrix...")
        cosine_distances = compute_cos_distance(embeddings_to_use, max_workers=max_workers)
        
        results = {
            'num_samples': embeddings_to_use.shape[0],
            'cos_distance': float(np.mean(cosine_distances))
        }
        
        total_combinations = len(self.density_powers) * len(self.neighbors) * len(self.distance_powers)
        print(f"Computing NovelSum scores for {total_combinations} parameter combinations...")
        
        pbar = tqdm(total=total_combinations, desc="Computing metrics")
        
        for dp in self.density_powers:
            for nb in self.neighbors:
                current_densities = self.faiss_index.local_density(
                    embeddings_to_use, 
                    n_neighbors=nb, 
                    power=dp, 
                    regularization=1e-9
                )
                
                for distp in self.distance_powers:
                    novelsum_score = novelsum(cosine_distances, current_densities, power=distp)
                    key = f'neighbor_{nb}_density_{dp}_distance_{distp}'
                    results[key] = float(novelsum_score)
                    pbar.update(1)
        
        pbar.close()
        
        print(f"NovelSum computation completed.")
        print(f"Average cosine distance: {results['cos_distance']:.6f}")
        
        return results

