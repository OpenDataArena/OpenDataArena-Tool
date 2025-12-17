import json
from typing import Dict, List
import tiktoken
from tqdm import tqdm
from .base_scorer import BaseScorer
from .utils import get_total_lines
from concurrent.futures import ProcessPoolExecutor


# Helper function for multiprocessing (must be at module level for pickling)
def _process_single_line(args):
    """Helper function to process a single line (for multiprocessing)
    
    Args:
        args: Tuple of (line, encoder_name, n)
    
    Returns:
        Dict containing id and Unique_Ntoken_Score
    """
    line, encoder_name, n = args
    
    try:
        item = json.loads(line.strip())
        
        # Get encoder
        encoder = tiktoken.get_encoding(encoder_name)
        
        # Extract fields from data item
        instruction = item["instruction"]
        input_text = item.get("input", "")
        response = item["output"]
        
        if input_text:
            text = instruction + '\n' + input_text + '\n' + response
        else:
            text = instruction + '\n' + response
        
        # Tokenize using tiktoken to get token ID list
        tokens = encoder.encode(text, disallowed_special=())
        
        if len(tokens) < n:
            # If token count is less than n, return 0
            score = 0.0
        else:
            # Calculate n-grams
            n_grams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
            unique_ngrams = set(n_grams)
            score = len(unique_ngrams) / len(n_grams)
        
        return {
            "id": item.get("id", ""),
            "score": score
        }
    except Exception as e:
        # If processing fails, return a result with error marker
        return {
            "id": item.get("id", "unknown") if 'item' in locals() else "unknown",
            "score": 0.0,
            "error": str(e)
        }


class UniqueNtokenScorer(BaseScorer):
    def _validate_config(self):
        """Validate configuration parameters"""
        # Check if encoder is specified in config, use default 'o200k_base' if not
        if "encoder" not in self.config:
            print(
                "Warning: No encoder specified in config. Using default 'o200k_base' encoder.")
            self.config['encoder'] = 'o200k_base'
        else:
            print(f"Using specified encoder: {self.config['encoder']}.")

        # Check n parameter
        if "n" not in self.config or not isinstance(self.config["n"], int) or self.config["n"] <= 0:
            self.config["n"] = 2
            print("Warning: No/invalid n specified, use default value of 2.")
        else:
            print(f"Using specified n: {self.config['n']}.")

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
            self.encoder = tiktoken.get_encoding(
                self.config.get("encoder", "o200k_base"))
            print("Setting up UniqueNtokenScorer successfully")
        except Exception as e:
            print(f"Error loading encoder: {e}. Falling back to 'o200k_base'.")
            self.encoder = tiktoken.get_encoding("o200k_base")

    def score_item(self, data_item: Dict) -> float:
        """Calculate unique token n-gram ratio for a single data item"""
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
        
        if len(tokens) < self.config['n']:
            # If token count is less than n, return 0
            return 0.0
        
        # Calculate n-grams
        n_grams = [tuple(tokens[i:i+self.config['n']]) for i in range(len(tokens) - self.config['n'] + 1)]
        unique_ngrams = set(n_grams)
        
        return len(unique_ngrams) / len(n_grams)

    def evaluate(self, dataset) -> List[Dict]:
        """Evaluate the entire dataset"""
        num_lines = get_total_lines(dataset)
        max_workers = self.config.get('max_workers', 1)
        encoder_name = self.config.get('encoder', 'o200k_base')
        n = self.config.get('n', 2)
        
        print(f"Using {max_workers} worker(s) for parallel processing")
        
        # Read all lines and prepare tasks
        with open(dataset, 'r', encoding='utf-8') as f:
            lines = [line for line in f]
        
        # Prepare task parameters
        tasks = [(line, encoder_name, n) for line in lines]
        
        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Use tqdm to display progress bar
            results = list(tqdm(
                executor.map(_process_single_line, tasks),
                total=num_lines,
                desc=self.config.get('name', 'UniqueNtokenScorer')
            ))
        
        return results

