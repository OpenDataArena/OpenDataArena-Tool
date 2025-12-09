import json
import math
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
        args: Tuple of (line, sample_size)
    
    Returns:
        Dict containing id and HDD_Score
    """
    line, sample_size = args
    
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
        
        # Calculate HDD score
        score = _compute_hdd(text.split(), sample_size)
        
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


def _compute_hdd(word_array, sample_size=42.0):
    """
    Calculate HD-D (Hypergeometric Distribution D) score
    Used to measure lexical diversity of text
    
    Reference: McCarthy & Jarvis (2010)
    """
    if isinstance(word_array, str):
        raise ValueError(
            "Input should be a list of strings, rather than a string. Try using string.split()")
    
    if len(word_array) < 1:
        return 0.0  # Return 0 instead of raising exception
    
    # If sample_size is greater than text length, use text length
    sample_size = min(int(sample_size), len(word_array))
    
    # Set up translation table for removing punctuation
    remove_punctuation = str.maketrans('', '', string.punctuation)
    
    # Create a dictionary of counts for each type
    type_counts = {}
    for token in word_array:
        # Trim punctuation, make lowercase
        token = token.translate(remove_punctuation).lower()
        if token and len(token) > 0:  # Ignore empty strings
            if token in type_counts:
                type_counts[token] += 1.0
            else:
                type_counts[token] = 1.0
    
    if len(type_counts) == 0:
        return 0.0
    
    # Sum the contribution of each token
    hdd_value = 0.0
    for token_type in type_counts.keys():
        try:
            contribution = (1.0 - _hypergeometric(
                len(word_array),
                type_counts[token_type],
                sample_size,
                0.0
            )) / sample_size
            hdd_value += contribution
        except (ValueError, ZeroDivisionError, OverflowError):
            # If calculation fails, skip this token
            continue
    
    return hdd_value


def _hypergeometric(population, population_successes, sample, sample_successes):
    """Calculate hypergeometric distribution probability"""
    # Convert to integers
    population = int(population)
    population_successes = int(population_successes)
    sample = int(sample)
    sample_successes = int(sample_successes)
    
    # Parameter validation: ensure hypergeometric distribution parameters are mathematically valid
    if population <= 0 or sample <= 0:
        return 0.0
    
    if population_successes < 0 or sample_successes < 0:
        return 0.0
    
    if sample > population:
        return 0.0
    
    if population_successes > population:
        return 0.0
    
    if sample_successes > sample:
        return 0.0
    
    if sample_successes > population_successes:
        return 0.0
    
    # Denominator
    denominator = _combination(population, sample)
    if denominator == 0:
        return 0.0
    
    # Numerator
    numerator = (_combination(population_successes, sample_successes) *
                 _combination(population - population_successes, sample - sample_successes))
    
    return numerator / denominator


def _combination(n, r):
    """Calculate combination: n choose r = n(n-1)(n-2)...(n-r+1)/(r!)"""
    # Convert to integers to avoid floating point precision issues
    n = int(n)
    r = int(r)
    
    # Use math.comb (Python 3.8+) or manual calculation
    try:
        return float(math.comb(n, r))
    except (ValueError, OverflowError):
        # If parameters are invalid or result is too large, return 0
        return 0.0


class HddScorer(BaseScorer):
    def _validate_config(self):
        """Validate configuration parameters"""
        # HDD-D sample_size parameter validation
        if "sample_size" in self.config and isinstance(self.config["sample_size"], (int, float)) and self.config["sample_size"] > 0:
            print(f"Using specified sample_size: {self.config['sample_size']}.")
        elif "sample_size" in self.config and isinstance(self.config["sample_size"], (int, float)) and self.config["sample_size"] <= 0:
            print("Warning: sample_size should be > 0, using default value of 42.0.")
            self.config['sample_size'] = 42.0
        else:
            print("Warning: No specific sample_size, using default value of 42.0.")
            self.config['sample_size'] = 42.0

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
        """Initialize HddScorer"""
        print("Setting up HddScorer successfully")

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
        
        # Use sample_size from configuration
        sample_size = self.config.get('sample_size', 42.0)
        return _compute_hdd(text.split(), sample_size=sample_size)

    def evaluate(self, dataset) -> List[Dict]:
        """Evaluate the entire dataset"""
        num_lines = get_total_lines(dataset)
        max_workers = self.config.get('max_workers', 1)
        sample_size = self.config.get('sample_size', 42.0)
        
        print(f"Using {max_workers} worker(s) for parallel processing")
        
        # Read all lines and prepare tasks
        with open(dataset, 'r', encoding='utf-8') as f:
            lines = [line for line in f]
        
        # Prepare task parameters
        tasks = [(line, sample_size) for line in lines]
        
        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Use tqdm to display progress bar
            results = list(tqdm(
                executor.map(_process_single_line, tasks),
                total=num_lines,
                desc=self.config.get('name', 'scorer')
            ))
        
        return results