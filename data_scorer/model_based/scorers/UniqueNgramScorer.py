import json
from typing import Dict, List
import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from tqdm import tqdm
from .base_scorer import BaseScorer
from .utils import get_total_lines
from concurrent.futures import ProcessPoolExecutor


# Helper function for multiprocessing (must be at module level for pickling)
def _process_single_line(args):
    """Helper function to process a single line (for multiprocessing)
    
    Args:
        args: Tuple of (line, n)
    
    Returns:
        Dict containing id and Unique_Ngram_Score
    """
    line, n = args
    
    try:
        item = json.loads(line.strip())
        
        # Extract instruction, input, and output from data item
        instruction = item["instruction"]
        input_text = item.get("input", "")
        response = item["output"]
        
        # Combine text fields
        if input_text:
            text = instruction + '\n' + input_text + '\n' + response
        else:
            text = instruction + '\n' + response
        
        # Tokenize and calculate unique n-gram ratio
        text = text.lower()
        tokens = word_tokenize(text)
        
        if len(tokens) < n:
            # If token count is less than n, return 0
            score = 0.0
        else:
            n_grams = list(ngrams(tokens, n))
            unique_ngrams = set(n_grams)
            
            if len(n_grams) == 0:
                score = 0.0
            else:
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


class UniqueNgramScorer(BaseScorer):
    def _validate_config(self):
        """Validate configuration parameters"""
        if "n" not in self.config or not isinstance(self.config["n"], int) or self.config["n"] <= 0:
            self.config["n"] = 2
            print("Warning: No/invalid n specified, use default value of 2.")
        else:
            print(f"Using specified n: {self.config['n']}.")

        if "max_workers" not in self.config or not isinstance(self.config["max_workers"], int) or self.config["max_workers"] <= 0:
            import os
            # Default to CPU core count
            default_workers = max(1, os.cpu_count() or 1)
            print(f"Warning: No/invalid max_workers, using default value of {default_workers} (CPU count).")
            self.config['max_workers'] = default_workers
        else:
            print(f"Using specified max_workers: {self.config['max_workers']}.")

    def _setup(self):
        """Initialize scorer and download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            print("Downloading NLTK punkt_tab tokenizer...")
            nltk.download('punkt_tab', quiet=True)
            print("NLTK punkt_tab tokenizer downloaded successfully.")
        
        print("Setting up UniqueNgramScorer successfully")

    def score_item(self, data_item: Dict) -> float:
        """Calculate unique n-gram ratio for a single data item"""
        instruction = data_item["instruction"]
        input_text = data_item.get("input", "")
        response = data_item["output"]
        
        if input_text:
            text = instruction + '\n' + input_text + '\n' + response
        else:
            text = instruction + '\n' + response
        
        text = text.lower()
        tokens = word_tokenize(text)
        
        if len(tokens) < self.config['n']:
            # If token count is less than n, return 0
            return 0.0
        
        n_grams = list(ngrams(tokens, self.config['n']))
        unique_ngrams = set(n_grams)
        
        if len(n_grams) == 0:
            return 0.0
        else:
            return len(unique_ngrams) / len(n_grams)

    def evaluate(self, dataset) -> List[Dict]:
        """Evaluate the entire dataset"""
        num_lines = get_total_lines(dataset)
        max_workers = self.config.get('max_workers', 1)
        n = self.config.get('n', 2)
        
        print(f"Using {max_workers} worker(s) for parallel processing")
        
        # Read all lines and prepare tasks
        with open(dataset, 'r', encoding='utf-8') as f:
            lines = [line for line in f]
        
        # Prepare task parameters
        tasks = [(line, n) for line in lines]
        
        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Use tqdm to display progress bar
            results = list(tqdm(
                executor.map(_process_single_line, tasks),
                total=num_lines,
                desc=self.config.get('name', 'UniqueNgramScorer')
            ))
        
        return results
