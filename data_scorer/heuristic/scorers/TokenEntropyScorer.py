from .base_scorer import BaseScorer
import json
from typing import Dict, List
import tiktoken
from tqdm import tqdm
from .utils import get_total_lines
from concurrent.futures import ProcessPoolExecutor
import math
from collections import Counter


# Helper function for multiprocessing (must be at module level for pickling)
def _process_single_line(args):
    """Helper function to process a single line (for multiprocessing)
    
    Args:
        args: Tuple of (line, encoder_name)
    
    Returns:
        Dict containing id and Token_Entropy_Score
    """
    line, encoder_name = args
    
    try:
        item = json.loads(line.strip())
        
        # Get text content from data item
        instruction = item["instruction"]
        input_text = item.get("input", "")
        response = item["output"]
        
        if input_text:
            text = instruction + '\n' + input_text + '\n' + response
        else:
            text = instruction + '\n' + response
        
        # Initialize encoder for this process
        try:
            encoder = tiktoken.get_encoding(encoder_name)
        except Exception:
            encoder = tiktoken.get_encoding("o200k_base")
        
        # Tokenize using tiktoken to get token ID list
        try:
            tokens = encoder.encode(text, disallowed_special=())
        except Exception as e:
            return {
                "id": item.get("id", ""),
                "score": 0.0,
                "error": f"Encoding error: {str(e)}"
            }
        
        if len(tokens) == 0:
            # If no tokens, return 0
            return {
                "id": item.get("id", ""),
                "score": 0.0
            }
        
        # Count token frequencies
        token_counts = Counter(tokens)
        total_count = len(tokens)
        
        # Calculate entropy: H(X) = -Σ p(x) * log2(p(x))
        entropy = 0.0
        for count in token_counts.values():
            probability = count / total_count
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        return {
            "id": item.get("id", ""),
            "sore": entropy
        }
        
    except Exception as e:
        # If processing fails, return a result with error marker
        return {
            "id": item.get("id", "unknown") if 'item' in locals() else "unknown",
            "score": 0.0,
            "error": str(e)
        }


class TokenEntropyScorer(BaseScorer):
    def _validate_config(self):
        # Check if encoder is specified in config, if not, use default o200k_base
        if "encoder" not in self.config:
            print("Warning: No encoder specified in config. Using default 'o200k_base' encoder.")
            self.config['encoder'] = 'o200k_base'
        else:
            print(f"Using specified encoder: {self.config['encoder']}.")
        
        # Check max_workers (process count)
        if "max_workers" not in self.config or not isinstance(self.config["max_workers"], int) or self.config["max_workers"] <= 0:
            import os
            # Default to CPU core count
            default_workers = max(1, os.cpu_count() or 1)
            print(f"Warning: No/invalid max_workers, using default value of {default_workers} (CPU count).")
            self.config['max_workers'] = default_workers
        else:
            print(f"Using specified max_workers: {self.config['max_workers']}.")

    def _setup(self):
        """Initialize tiktoken encoder"""
        try:
            self.encoder = tiktoken.get_encoding(self.config.get("encoder", "o200k_base"))
            print("Setting up TokenEntropyScorer successfully")
        except Exception as e:
            print(f"Error loading encoder: {e}. Falling back to 'o200k_base'.")
            self.encoder = tiktoken.get_encoding("o200k_base")

    def score_item(self, data_item: Dict) -> float:
        """Calculate token entropy score for a single data item"""
        instruction = data_item["instruction"]
        input_text = data_item.get("input", "")
        response = data_item["output"]
        
        if input_text:
            text = instruction + '\n' + input_text + '\n' + response
        else:
            text = instruction + '\n' + response
        
        # Tokenize using tiktoken to get token ID list
        try:
            tokens = self.encoder.encode(text, disallowed_special=())
        except Exception as e:
            print(f"[score_item] Encoding error: {e}")
            return 0.0
        
        if len(tokens) == 0:
            # If no tokens, return 0
            return 0.0
        
        # Count token frequencies
        token_counts = Counter(tokens)
        total_count = len(tokens)
        
        # Calculate entropy: H(X) = -Σ p(x) * log2(p(x))
        entropy = 0.0
        for count in token_counts.values():
            probability = count / total_count
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        return entropy

    def evaluate(self, dataset) -> List[Dict]:
        """Evaluate the entire dataset"""
        num_lines = get_total_lines(dataset)
        max_workers = self.config.get('max_workers', 1)
        encoder_name = self.config.get('encoder', 'o200k_base')
        
        print(f"Using {max_workers} worker(s) for parallel processing")
        
        # Read all lines and prepare tasks
        with open(dataset, 'r', encoding='utf-8') as f:
            lines = [line for line in f]
        
        # Prepare task parameters
        tasks = [(line, encoder_name) for line in lines]
        
        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Use tqdm to display progress bar
            results = list(tqdm(
                executor.map(_process_single_line, tasks),
                total=num_lines,
                desc=self.config.get('name', 'TokenEntropyScorer')
            ))
        
        return results
