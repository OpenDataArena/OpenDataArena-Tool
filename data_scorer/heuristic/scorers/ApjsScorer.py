from .base_scorer import BaseScorer
from .utils import get_total_lines
from typing import Dict, List, Set, Tuple
import json
import tiktoken
import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from tqdm import tqdm
from datasketch import MinHash
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import os
import random


# Helper function for multiprocessing (must be at module level for pickling)
def _compute_jaccard_pair(args):
    """Compute Jaccard similarity for a single pair of n-gram sets
    
    Args:
        args: Tuple of (set1, set2, similarity_method, num_perm)
    
    Returns:
        float: Jaccard similarity score
    """
    set1, set2, similarity_method, num_perm = args
    
    # If either set is empty, return 0.0 (cannot compare)
    if len(set1) == 0 or len(set2) == 0:
        return 0.0
    
    if similarity_method == "direct":
        # Direct computation
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    else:  # minhash
        # MinHash approximation
        m1 = MinHash(num_perm=num_perm)
        m2 = MinHash(num_perm=num_perm)
        
        # Update MinHash
        for item in set1:
            m1.update(str(item).encode('utf-8'))
        
        for item in set2:
            m2.update(str(item).encode('utf-8'))
        
        return m1.jaccard(m2)


class ApjsScorer(BaseScorer):
    def _validate_config(self):
        """Validate configuration parameters"""
        # Check tokenization method
        if "tokenization_method" not in self.config:
            self.config["tokenization_method"] = "gram"
            print("Warning: No tokenization_method specified, using default 'gram'.")
        else:
            valid_methods = ["gram", "token"]
            if self.config["tokenization_method"] not in valid_methods:
                print(
                    f"Warning: Invalid tokenization_method '{self.config['tokenization_method']}', "
                    f"using default 'gram'. Available methods: {', '.join(valid_methods)}"
                )
                self.config["tokenization_method"] = "gram"
            else:
                print(f"Using specified tokenization_method: {self.config['tokenization_method']}")
        
        # Check n parameter
        if "n" not in self.config or not isinstance(self.config["n"], int) or self.config["n"] <= 0:
            self.config["n"] = 1
            print("Warning: No/invalid n specified, use default value of 1.")
        else:
            print(f"Using specified n: {self.config['n']}.")
        
        # Check similarity method
        if "similarity_method" not in self.config:
            self.config["similarity_method"] = "direct"
            print("Warning: No similarity_method specified, using default 'direct'.")
        else:
            valid_sim_methods = ["direct", "minhash"]
            if self.config["similarity_method"] not in valid_sim_methods:
                print(
                    f"Warning: Invalid similarity_method '{self.config['similarity_method']}', "
                    f"using default 'direct'. Available methods: {', '.join(valid_sim_methods)}"
                )
                self.config["similarity_method"] = "direct"
            else:
                print(f"Using specified similarity_method: {self.config['similarity_method']}")
        
        # If using token tokenization, check encoder parameter
        if self.config["tokenization_method"] == "token":
            if "encoder" not in self.config:
                self.config["encoder"] = "o200k_base"
                print("Warning: No encoder specified, using default 'o200k_base'.")
            else:
                print(f"Using specified encoder: {self.config['encoder']}.")
        
        # If using MinHash, check num_perm parameter
        if self.config["similarity_method"] == "minhash":
            if "num_perm" not in self.config or not isinstance(self.config["num_perm"], int) or self.config["num_perm"] <= 0:
                self.config["num_perm"] = 128
                print("Warning: No/invalid num_perm specified for MinHash, use default value of 128.")
            else:
                print(f"Using specified num_perm for MinHash: {self.config['num_perm']}.")
        
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
        """Initialize necessary components"""
        # If using gram tokenization, need to download nltk data
        if self.config["tokenization_method"] == "gram":
            try:
                nltk.data.find('tokenizers/punkt_tab')
            except LookupError:
                print("Downloading NLTK punkt_tab tokenizer...")
                nltk.download('punkt_tab', quiet=True)
                print("NLTK punkt_tab tokenizer downloaded successfully.")
        
        # If using token tokenization, need to initialize tiktoken encoder
        if self.config["tokenization_method"] == "token":
            try:
                self.encoder = tiktoken.get_encoding(self.config["encoder"])
                print(f"Encoder '{self.config['encoder']}' loaded successfully.")
            except Exception as e:
                print(f"Error loading encoder: {e}. Falling back to 'o200k_base'.")
                self.encoder = tiktoken.get_encoding("o200k_base")
                self.config["encoder"] = "o200k_base"
        
        print("Setting up ApjsScorer successfully")

    def _extract_text(self, data_item: Dict) -> str:
        """Extract text from data item"""
        instruction = data_item["instruction"]
        input_text = data_item.get("input", "")
        response = data_item["output"]
        
        if input_text:
            text = instruction + '\n' + input_text + '\n' + response
        else:
            text = instruction + '\n' + response
        
        return text

    def _tokenize_gram(self, text: str) -> Set[Tuple]:
        """Tokenize using gram method"""
        text = text.lower()
        tokens = word_tokenize(text)
        
        if len(tokens) < self.config['n']:
            return set()
        
        n_grams = list(ngrams(tokens, self.config['n']))
        return set(n_grams)

    def _tokenize_token(self, text: str) -> Set[Tuple]:
        """Tokenize using token method"""
        try:
            tokens = self.encoder.encode(text, disallowed_special=())
        except Exception as e:
            print(f"Encoding error: {e}")
            return set()
        
        if len(tokens) < self.config['n']:
            return set()
        
        n_grams = [tuple(tokens[i:i+self.config['n']]) for i in range(len(tokens) - self.config['n'] + 1)]
        return set(n_grams)

    def _get_ngrams(self, data_item: Dict) -> Set[Tuple]:
        """Get n-grams based on configured tokenization method"""
        text = self._extract_text(data_item)
        
        if self.config["tokenization_method"] == "gram":
            return self._tokenize_gram(text)
        else:  # token
            return self._tokenize_token(text)

    def _jaccard_similarity_direct(self, set1: Set, set2: Set) -> float:
        """Compute Jaccard similarity directly"""
        # If either set is empty, return 0.0 (cannot compare)
        if len(set1) == 0 or len(set2) == 0:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        if union == 0:
            return 0.0
        
        return intersection / union

    def _jaccard_similarity_minhash(self, set1: Set, set2: Set) -> float:
        """Approximate Jaccard similarity using MinHash"""
        # If either set is empty, return 0.0 (cannot compare)
        if len(set1) == 0 or len(set2) == 0:
            return 0.0
        
        # Create MinHash objects
        m1 = MinHash(num_perm=self.config["num_perm"])
        m2 = MinHash(num_perm=self.config["num_perm"])
        
        # Update MinHash
        for item in set1:
            # Convert tuple to string for encoding
            m1.update(str(item).encode('utf-8'))
        
        for item in set2:
            m2.update(str(item).encode('utf-8'))
        
        # Return estimated Jaccard similarity
        return m1.jaccard(m2)

    def _compute_jaccard(self, set1: Set, set2: Set) -> float:
        """Compute Jaccard similarity based on configured method"""
        if self.config["similarity_method"] == "direct":
            return self._jaccard_similarity_direct(set1, set2)
        else:  # minhash
            return self._jaccard_similarity_minhash(set1, set2)

    def score_item(self, data_item: Dict) -> float:
        """A p j sScorer scores the entire dataset, not individual samples"""
        raise NotImplementedError(
            "ApjsScorer computes a single score for the entire dataset. "
            "Use evaluate() method instead."
        )

    def evaluate(self, dataset) -> Dict:
        """Evaluate the entire dataset and compute Average Pairwise Jaccard Similarity (Apjs)
        
        Args:
            dataset: Path to dataset file (jsonl format)
        
        Returns:
            Dictionary containing Apjs score
        """
        print(f"Loading dataset from: {dataset}")
        num_lines = get_total_lines(dataset)
        
        # Read all data and extract n-grams
        print("Extracting n-grams from all samples...")
        all_ngrams = []
        
        with open(dataset, 'r', encoding='utf-8') as f:
            for line in tqdm(f, total=num_lines, desc="Loading and tokenizing"):
                item = json.loads(line.strip())
                ngrams_set = self._get_ngrams(item)
                all_ngrams.append(ngrams_set)
        
        n = len(all_ngrams)
        print(f"Computing Apjs for {n} samples...")
        
        # Handle edge case: less than 2 samples in dataset
        if n < 2:
            print(f"Warning: Dataset has {n} samples. At least 2 samples are required to compute Apjs.")
            result = {
                "score": 0.0 if n == 0 else None,  # Return 0.0 for 0 samples, None for 1 sample indicating cannot compute
                "num_samples": n,
                "num_pairs": 0,
                "tokenization_method": self.config["tokenization_method"],
                "n": self.config["n"],
                "similarity_method": self.config["similarity_method"],
                "warning": f"Insufficient samples (found {n}, need at least 2)"
            }
            if self.config["tokenization_method"] == "token":
                result["encoder"] = self.config["encoder"]
            if self.config["similarity_method"] == "minhash":
                result["num_perm"] = self.config["num_perm"]
            return result
        
        print(f"Using tokenization method: {self.config['tokenization_method']}")
        print(f"Using n-gram size: {self.config['n']}")
        print(f"Using similarity method: {self.config['similarity_method']}")
        
        # Compute Jaccard similarity for all pairs (or sampled pairs)
        total_pairs = n * (n - 1) // 2
        max_workers = self.config.get('max_workers', 1)
        print(f"Using {max_workers} worker(s) for parallel processing")
        print(f"Total possible pairs: {total_pairs}")
        
        # Prepare all pair combinations with necessary parameters
        similarity_method = self.config["similarity_method"]
        num_perm = self.config.get("num_perm", 128)
        sample_pairs = self.config.get("sample_pairs")
        
        # Generate all possible pair indices
        all_pair_indices = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        # Determine whether to sample
        if sample_pairs is not None and sample_pairs < total_pairs:
            print(f"Sampling {sample_pairs} pairs from {total_pairs} total pairs...")
            sampled_indices = random.sample(all_pair_indices, sample_pairs)
            pair_tasks = [(all_ngrams[i], all_ngrams[j], similarity_method, num_perm) 
                         for i, j in sampled_indices]
            actual_pairs = sample_pairs
            is_sampled = True
        else:
            if sample_pairs is not None and sample_pairs >= total_pairs:
                print(f"sample_pairs ({sample_pairs}) >= total_pairs ({total_pairs}), will compute all pairs.")
            print("Computing all pairs...")
            pair_tasks = [(all_ngrams[i], all_ngrams[j], similarity_method, num_perm) 
                         for i, j in all_pair_indices]
            actual_pairs = total_pairs
            is_sampled = False
        
        # Parallel computation using ProcessPoolExecutor
        pairwise_similarities = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Use tqdm to show progress
            desc = f"Computing {'sampled ' if is_sampled else ''}pairwise Jaccard similarities"
            results = list(tqdm(
                executor.map(_compute_jaccard_pair, pair_tasks),
                total=actual_pairs,
                desc=desc
            ))
            pairwise_similarities = results
        
        # Compute mean
        apjs_score = np.mean(pairwise_similarities).item()
        
        if is_sampled:
            print(f"Apjs Score (sampled): {apjs_score}")
            print(f"Total pairs computed: {len(pairwise_similarities)} (sampled from {total_pairs} possible pairs)")
        else:
            print(f"Apjs Score: {apjs_score}")
            print(f"Total pairs computed: {len(pairwise_similarities)}")
        
        result = {
            "score": apjs_score,
            "num_samples": n,
            "num_pairs": len(pairwise_similarities),
            "total_possible_pairs": total_pairs,
            "is_sampled": is_sampled,
            "tokenization_method": self.config["tokenization_method"],
            "n": self.config["n"],
            "similarity_method": self.config["similarity_method"],
            "max_workers": max_workers
        }
        
        # If sampled, add sampling info
        if is_sampled:
            result["sample_pairs"] = sample_pairs
        
        # If using token method, add encoder info
        if self.config["tokenization_method"] == "token":
            result["encoder"] = self.config["encoder"]
        
        # If using MinHash, add num_perm info
        if self.config["similarity_method"] == "minhash":
            result["num_perm"] = self.config["num_perm"]
        
        return result

