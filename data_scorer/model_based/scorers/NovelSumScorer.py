from .base_scorer import BaseScorer
from .utils import get_total_lines
from typing import Dict, List
import numpy as np
import os
import torch
from tqdm import tqdm
import random
import pathlib
import faiss


def set_device(gpu_id: int):
    if gpu_id is not None and torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        return torch.device(f'cuda:{gpu_id}')
    else:
        return torch.device('cpu')
    
def set_seed(seed: int) -> None:
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

class FaissIndex:
    def __init__(self, data: np.ndarray, gpu_id: int, use_gpu: bool = True):
        self.gpu_id = gpu_id
        self.device = set_device(gpu_id)
        self.use_gpu = use_gpu
        self.data = data.astype(np.float32).reshape(-1, data.shape[-1])
        self.data = np.unique(self.data, axis=0)  # Remove duplicates
        
        print(f"Creating Faiss index with {self.data.shape[0]} vectors, dimension {self.data.shape[1]}")
        print(f"Index data size: {self.data.nbytes / 1024 / 1024:.2f} MB")
        
        if use_gpu and torch.cuda.is_available():
            try:
                self.res = faiss.StandardGpuResources()
                # Limit temporary memory usage
                self.res.setTempMemory(256 * 1024 * 1024)  # 256MB
                self.index = faiss.IndexFlatL2(self.data.shape[1])
                self.gpu_index = faiss.index_cpu_to_gpu(self.res, gpu_id, self.index)
                self.gpu_index.add(self.data)
                print(f"Faiss GPU index created successfully on GPU {gpu_id}")
            except Exception as e:
                print(f"Warning: Failed to create GPU index: {e}")
                print("Falling back to CPU index...")
                self.use_gpu = False
                self.gpu_index = faiss.IndexFlatL2(self.data.shape[1])
                self.gpu_index.add(self.data)
                print("Faiss CPU index created successfully")
        else:
            self.use_gpu = False
            self.gpu_index = faiss.IndexFlatL2(self.data.shape[1])
            self.gpu_index.add(self.data)
            print("Faiss CPU index created successfully")

    def search(self, query: np.ndarray, n_neighbors: int):
        query = query.astype(np.float32)
        distances, indices = self.gpu_index.search(query, n_neighbors + 1)  # +1 to exclude self
        return distances[:, 1:], indices[:, 1:]  # Exclude the first neighbor (self)

    def local_density(self, query_data: np.ndarray, n_neighbors=10, regularization=1e-9, power=1):
        distances, _ = self.search(query_data, n_neighbors)
        avg_distances = np.mean(distances, axis=-1)
        densities = 1 / np.power((avg_distances + regularization), power)
        return densities

def load_npy_data(file: str) -> np.ndarray:
    """Load a single .npy file"""
    return np.load(file)

def load_dir_npy_data(npy_dir: str, desc: str = "Loading data") -> np.ndarray:
    """Load and concatenate all .npy files from directory"""
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

def compute_cos_distance(data: np.ndarray, device: torch.device) -> np.ndarray:
    """Compute cosine distance matrix"""
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    list_a_tensor = torch.tensor(data, device=device).float()
    num_vectors = list_a_tensor.shape[0]
    block_size = 500  # Adjust according to GPU memory
    cosine_distance_matrix = torch.zeros((num_vectors, num_vectors), device=device)

    for i in range(0, num_vectors, block_size):
        for j in range(0, num_vectors, block_size):
            end_i = min(i + block_size, num_vectors)
            end_j = min(j + block_size, num_vectors)
            block_i = list_a_tensor[i:end_i]
            block_j = list_a_tensor[j:end_j]

            cosine_similarity_block = torch.nn.functional.cosine_similarity(
                block_i.unsqueeze(1), block_j.unsqueeze(0), dim=2
            )
            cosine_distance_block = 1 - cosine_similarity_block
            cosine_distance_matrix[i:end_i, j:end_j] = cosine_distance_block
    
    cosine_distance = cosine_distance_matrix.cpu().numpy()
    return cosine_distance


def weighted_average(row: np.ndarray, power: float = 1.0) -> float:
    """Compute weighted average"""
    sorted_indices = np.argsort(row)
    weights = 1 / np.power(np.arange(1, len(row) + 1), power)
    sorted_row = row[sorted_indices]
    return np.average(sorted_row, weights=weights)


def novelsum(distance_matrix: np.ndarray, densities: np.ndarray, power: float = 1.0) -> float:
    """Compute NovelSum score"""
    weighted_matrix = distance_matrix * densities[:, np.newaxis]
    weighted_averages = np.apply_along_axis(weighted_average, 1, weighted_matrix, power=power)
    return np.mean(weighted_averages)


class NovelSumScorer(BaseScorer):
    def _validate_config(self):
        """Validate configuration parameters"""
        # Validate embedding_path
        if "embedding_path" not in self.config:
            raise ValueError("embedding_path is required in config.")
        
        embedding_path = self.config["embedding_path"]
        if not os.path.exists(embedding_path):
            raise FileNotFoundError(f"Embedding file not found: {embedding_path}")
        
        if not embedding_path.endswith('.npy'):
            raise ValueError(f"Embedding file must be a .npy file, but got: {embedding_path}")
        
        # Set dense_ref_path default value: if not provided, use the directory of embedding_path
        if "dense_ref_path" not in self.config:
            self.config["dense_ref_path"] = os.path.dirname(embedding_path)
            print(f"Info: No dense_ref_path specified, using embedding directory: {self.config['dense_ref_path']}")
        else:
            dense_ref_path = self.config["dense_ref_path"]
            if not os.path.exists(dense_ref_path):
                raise FileNotFoundError(f"Dense reference path not found: {dense_ref_path}")
        
        # Set default parameters (gpu_id is fixed to 0, no need for user to specify)
        self.config["gpu_id"] = 0
        
        # Set whether to use GPU Faiss index (default to CPU, more stable)
        if "use_gpu_faiss" not in self.config:
            self.config["use_gpu_faiss"] = False
            print(f"Info: No use_gpu_faiss specified, using CPU Faiss index (more stable).")
        
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
        """Load embedding file and initialize Faiss index"""
        # Load embeddings
        embedding_path = self.config["embedding_path"]
        print(f"Loading embeddings from: {embedding_path}")
        
        self.embeddings = np.load(embedding_path)
        print(f"Embeddings loaded. Shape: {self.embeddings.shape}")
        
        num_data, embedding_size = self.embeddings.shape
        self.num_data = num_data
        self.embedding_size = embedding_size
        
        # Load reference density data (load all .npy files from directory)
        dense_ref_path = self.config["dense_ref_path"]
        print(f"Loading dense reference data from: {dense_ref_path}")
        print(f"Path exists: {os.path.exists(dense_ref_path)}")
        print(f"Is directory: {os.path.isdir(dense_ref_path)}")
        
        # Load data based on path type
        if os.path.isdir(dense_ref_path):
            print("Loading from directory...")
            self.dense_ref_data = load_dir_npy_data(dense_ref_path, desc="Loading reference data")
        else:
            print("Loading from single file...")
            self.dense_ref_data = load_npy_data(dense_ref_path)
        
        print(f"Dense reference data loaded. Shape: {self.dense_ref_data.shape}")
        
        # Initialize device and Faiss index
        gpu_id = self.config["gpu_id"]
        use_gpu_faiss = self.config["use_gpu_faiss"]
        self.device = set_device(gpu_id)
        print(f"Using device: {self.device}")
        
        self.faiss_index = FaissIndex(self.dense_ref_data, gpu_id, use_gpu=use_gpu_faiss)
        print("Faiss index initialized successfully")
        
        # Get hyperparameters
        self.density_powers = self.config["density_powers"]
        self.neighbors = self.config["neighbors"]
        self.distance_powers = self.config["distance_powers"]
        
        print("Setting up NovelSumScorer successfully")

    def score_item(self, data_item: Dict) -> float:
        """NovelSumScorer scores the entire dataset, not individual samples"""
        raise NotImplementedError(
            "NovelSumScorer computes scores for the entire dataset. "
            "Use evaluate() method instead."
        )

    def evaluate(self, dataset) -> Dict:
        """Evaluate the entire dataset and compute NovelSum score
        
        Args:
            dataset: dataset file path (jsonl format)
        
        Returns:
            Dictionary containing NovelSum scores under various configurations
        """
        num_lines = get_total_lines(dataset)
        
        # Verify if the number of dataset rows matches the number of embeddings
        if num_lines != self.num_data:
            print(f"Warning: Dataset has {num_lines} lines but embeddings have {self.num_data} rows.")
            print("Will use min(num_lines, num_data) for processing.")
            num_to_use = min(num_lines, self.num_data)
            embeddings_to_use = self.embeddings[:num_to_use]
        else:
            embeddings_to_use = self.embeddings
        
        print(f"Computing NovelSum for {embeddings_to_use.shape[0]} samples...")
        
        # Compute cosine distance matrix
        print("Computing cosine distance matrix...")
        cosine_distances = compute_cos_distance(embeddings_to_use, self.device)
        
        # Store results
        results = {
            'num_samples': embeddings_to_use.shape[0],
            'cos_distance': float(np.mean(cosine_distances))
        }
        
        # Compute NovelSum scores under different hyperparameter combinations
        total_combinations = len(self.density_powers) * len(self.neighbors) * len(self.distance_powers)
        print(f"Computing NovelSum scores for {total_combinations} parameter combinations...")
        
        pbar = tqdm(total=total_combinations, desc="Computing metrics")
        
        for dp in self.density_powers:
            for nb in self.neighbors:
                # Compute local density under current configuration
                current_densities = self.faiss_index.local_density(
                    embeddings_to_use, 
                    n_neighbors=nb, 
                    power=dp, 
                    regularization=1e-9
                )
                
                for distp in self.distance_powers:
                    # Compute NovelSum score
                    novelsum_score = novelsum(cosine_distances, current_densities, power=distp)
                    key = f'neighbor_{nb}_density_{dp}_distance_{distp}'
                    results[key] = float(novelsum_score)
                    pbar.update(1)
        
        pbar.close()
        
        print(f"NovelSum computation completed.")
        print(f"Average cosine distance: {results['cos_distance']:.6f}")
        
        return results

