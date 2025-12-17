import json
import string
from typing import Dict, List
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from .base_scorer import BaseScorer
from .utils import get_total_lines


# Helper function for multiprocessing (must be at module level for pickling)
def _process_single_line(args):
    """Helper function to process a single line (for multiprocessing)
    
    Args:
        args: Tuple of (line, ttr_threshold)
    
    Returns:
        Dict containing id and MTLD_Score
    """
    line, ttr_threshold = args
    
    try:
        item = json.loads(line.strip())
        
        # Extract text
        instruction = item["instruction"]
        input_text = item.get("input", "")
        response = item["output"]
        
        # Concatenate text
        if input_text:
            text = instruction + '\n' + input_text + '\n' + response
        else:
            text = instruction + '\n' + response
        
        # Calculate MTLD score
        score = _compute_mtld(text.split(), ttr_threshold)
        
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


def _mtld_calc(word_array, ttr_threshold, remove_punctuation):
    """Internal method to calculate MTLD"""
    current_ttr = 1.0
    token_count = 0
    type_count = 0
    types = set()
    factors = 0.0

    for token in word_array:
        # trim punctuation, make lowercase
        token = token.translate(remove_punctuation).lower()
        token_count += 1
        if token not in types:
            type_count += 1
            types.add(token)
        current_ttr = type_count / token_count
        if current_ttr <= ttr_threshold:
            factors += 1
            token_count = 0
            type_count = 0
            types = set()
            current_ttr = 1.0

    excess = 1.0 - current_ttr
    excess_val = 1.0 - ttr_threshold
    factors += excess / excess_val
    if factors != 0:
        return len(word_array) / factors
    return -1


def _compute_mtld(word_array, ttr_threshold=0.72):
    """
    Calculate MTLD (Measure of Textual Lexical Diversity) score
    Used to measure lexical diversity of text
    """
    if isinstance(word_array, str):
        raise ValueError(
            "Input should be a list of strings, rather than a string. Try using string.split()")
    if len(word_array) < 1:
        return 0.0  # Return 0 instead of raising exception
    
    # Set up translation table for removing punctuation
    remove_punctuation = str.maketrans('', '', string.punctuation)
    
    return (_mtld_calc(word_array, ttr_threshold, remove_punctuation) + 
            _mtld_calc(word_array[::-1], ttr_threshold, remove_punctuation)) / 2


class MtldScorer(BaseScorer):
    def _validate_config(self):
        """Validate configuration parameters"""
        # TTR threshold parameter validation
        if "ttr_threshold" in self.config and isinstance(self.config["ttr_threshold"], (int, float)) and 0 < self.config["ttr_threshold"] < 1:
            print(f"Using specified ttr_threshold: {self.config['ttr_threshold']}.")
        elif "ttr_threshold" in self.config and isinstance(self.config["ttr_threshold"], (int, float)) and not (0 < self.config["ttr_threshold"] < 1):
            print("Warning: ttr_threshold should be between 0 and 1, using default value of 0.72.")
            self.config['ttr_threshold'] = 0.72
        else:
            print("Warning: No specific ttr_threshold, using default value of 0.72.")
            self.config['ttr_threshold'] = 0.72
        
        # Multiprocessing worker count validation
        if "max_workers" in self.config and isinstance(self.config["max_workers"], int) and self.config["max_workers"] > 0:
            print(f"Using specified max_workers: {self.config['max_workers']}.")
        else:
            import os
            # Default to CPU core count
            default_workers = max(1, os.cpu_count() or 1)
            print(f"Warning: No/invalid max_workers, using default value of {default_workers} (CPU count).")
            self.config['max_workers'] = default_workers

    def _setup(self):
        """Initialize MtldScorer"""
        print("Setting up MtldScorer successfully")

    def score_item(self, data_item: Dict) -> float:
        """Score a single data item"""
        instruction = data_item["instruction"]
        input_text = data_item.get("input", "")
        response = data_item["output"]
        
        # Concatenate text
        if input_text:
            text = instruction + '\n' + input_text + '\n' + response
        else:
            text = instruction + '\n' + response
        
        # Use ttr_threshold from configuration
        ttr_threshold = self.config.get('ttr_threshold', 0.72)
        return _compute_mtld(text.split(), ttr_threshold=ttr_threshold)

    def evaluate(self, dataset) -> List[Dict]:
        """Evaluate the entire dataset"""
        num_lines = get_total_lines(dataset)
        max_workers = self.config.get('max_workers', 1)
        ttr_threshold = self.config.get('ttr_threshold', 0.72)
        
        print(f"Using {max_workers} worker(s) for parallel processing")
        
        # Read all lines and prepare tasks
        with open(dataset, 'r', encoding='utf-8') as f:
            lines = [line for line in f]
        
        # Prepare task parameters
        tasks = [(line, ttr_threshold) for line in lines]
        
        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Use tqdm to display progress bar
            results = list(tqdm(
                executor.map(_process_single_line, tasks),
                total=num_lines,
                desc=self.config.get('name', 'MtldScorer')
            ))
        
        return results