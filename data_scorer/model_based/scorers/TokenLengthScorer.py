import tiktoken
import json
from typing import Dict, List
from .base_scorer import BaseScorer
from .utils import get_total_lines
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


# Helper function for multiprocessing (must be at module level for pickling)
def _process_single_line(args):
    """Helper function to process a single line (for multiprocessing)
    
    Args:
        args: Tuple of (line, fields, encoder_name)
    
    Returns:
        Dict containing id and Token_Length
    """
    line, fields, encoder_name = args
    
    try:
        item = json.loads(line.strip())
        
        # Initialize encoder in the worker process
        try:
            encoder = tiktoken.get_encoding(encoder_name)
        except Exception as e:
            print(f"Error loading encoder: {e}. Falling back to 'o200k_base'.")
            encoder = tiktoken.get_encoding("o200k_base")
        
        # Extract specified fields from data item
        parts = []
        for field in fields:
            if field in item and item[field]:
                parts.append(str(item[field]))
        
        # Join all fields with newline separator
        text = "\n".join(parts) if parts else ""
        
        # Calculate token length
        token_length = len(encoder.encode(text, disallowed_special=()))
        
        return {
            "id": item.get("id", ""),
            "score": token_length
        }
    except Exception as e:
        # If processing fails, return a result with error marker
        return {
            "id": item.get("id", "unknown") if 'item' in locals() else "unknown",
            "score": 0,
            "error": str(e)
        }


class TokenLengthScorer(BaseScorer):
    def _validate_config(self):
        """Validate configuration parameters"""
        # Check if encoder is specified, default is 'o200k_base'
        if "encoder" not in self.config:
            print(
                "Warning: No encoder specified in config. Using default 'o200k_base' encoder.")
            self.config['encoder'] = 'o200k_base'
        else:
            print(f"Using specified encoder: {self.config['encoder']}.")

        # Check if fields are specified, default is ["instruction", "input", "output"]
        if "fields" not in self.config or not isinstance(self.config["fields"], list) or len(self.config["fields"]) == 0:
            print(
                "Warning: No fields specified in config. Using default fields: ['instruction', 'input', 'output'].")
            self.config['fields'] = ['instruction', 'input', 'output']
        else:
            print(f"Using specified fields: {self.config['fields']}.")

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
            print("Setting up TokenLengthScorer successfully")
        except Exception as e:
            print(f"Error loading encoder: {e}. Falling back to 'o200k_base'.")
            self.encoder = tiktoken.get_encoding("o200k_base")

    def score_item(self, data_item: Dict) -> int:
        """Calculate token length of a single data item"""
        fields = self.config['fields']
        
        # Extract specified fields from data item
        parts = []
        for field in fields:
            if field in data_item and data_item[field]:
                parts.append(str(data_item[field]))

        # Join all fields with newline separator
        text = "\n".join(parts) if parts else ""

        # Calculate token length
        try:
            token_length = len(self.encoder.encode(text, disallowed_special=()))
            return token_length
        except Exception as e:
            print(
                f"[score_item] Encoding error for item: {text[:100]}... \nError: {e}")
            return 0  # Return 0 when error occurs

    def evaluate(self, dataset) -> List[Dict]:
        """Evaluate the entire dataset"""
        num_lines = get_total_lines(dataset)
        max_workers = self.config.get('max_workers', 1)
        fields = self.config.get('fields', ['instruction', 'input', 'output'])
        encoder_name = self.config.get('encoder', 'o200k_base')
        
        print(f"Using {max_workers} worker(s) for parallel processing")
        
        # Read all lines and prepare tasks
        with open(dataset, 'r', encoding='utf-8') as f:
            lines = [line for line in f]
        
        # Prepare task parameters
        tasks = [(line, fields, encoder_name) for line in lines]
        
        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Use tqdm to display progress bar
            results = list(tqdm(
                executor.map(_process_single_line, tasks),
                total=num_lines,
                desc=self.config.get('name', 'TokenLengthScorer')
            ))
        
        return results
