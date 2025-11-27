import json
import re
from typing import Dict, List
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from .base_scorer import BaseScorer
from .utils import get_total_lines


# Compile regex patterns for think tags (module level for multiprocessing)
THINK_PATTERNS = [
    re.compile(r'<think\s*>', re.IGNORECASE),                      # think opening tag (with optional spaces)
    re.compile(r'</think\s*>', re.IGNORECASE),                     # think closing tag (with optional spaces)
    re.compile(r'<redacted_reasoning\s*>', re.IGNORECASE),         # redacted_reasoning opening tag
    re.compile(r'</redacted_reasoning\s*>', re.IGNORECASE),        # redacted_reasoning closing tag
]


def _contains_think_tag(text: str) -> bool:
    """
    Check if text contains think tags
    
    Args:
        text: Text to check
        
    Returns:
        bool: True if contains think tags, False otherwise
    """
    if not text or not isinstance(text, str):
        return False
    
    # Check all patterns
    for pattern in THINK_PATTERNS:
        if pattern.search(text):
            return True
    
    return False


# Helper function for multiprocessing (must be at module level for pickling)
def _process_single_line(args):
    """Helper function to process a single line (for multiprocessing)
    
    Args:
        args: Tuple of (line, field_name)
    
    Returns:
        Dict containing id and ThinkOrNot_Score
    """
    line, field_name = args
    
    try:
        item = json.loads(line.strip())
        
        # Extract text from specified field
        text = item.get(field_name, "")
        
        # Check if contains think tags
        if _contains_think_tag(text):
            score = 1.0
        else:
            score = 0.0
        
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


class ThinkOrNotScorer(BaseScorer):
    def _validate_config(self):
        """Validate configuration parameters"""
        # Check the field to examine
        if "field" not in self.config or not isinstance(self.config["field"], str):
            self.config["field"] = "output"
            print("Warning: No/invalid field specified, use default value of 'output'.")
        else:
            print(f"Using specified field: {self.config['field']}.")

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
        """Initialize ThinkOrNotScorer"""
        print("Setting up ThinkOrNotScorer successfully")

    def score_item(self, data_item: Dict) -> float:
        """
        Score a single data item
        Check if data contains LLM's thinking part
        
        Args:
            data_item: Data item dictionary
            
        Returns:
            float: 1.0 if contains think tags, 0.0 otherwise
        """
        # Extract text from specified field
        field_name = self.config["field"]
        text = data_item.get(field_name, "")
        
        # If field doesn't exist or is empty, return 0.0
        if not text or not isinstance(text, str):
            return 0.0
        
        # Check if contains think tags
        if _contains_think_tag(text):
            return 1.0
        else:
            return 0.0

    def evaluate(self, dataset) -> List[Dict]:
        """Evaluate the entire dataset"""
        num_lines = get_total_lines(dataset)
        max_workers = self.config.get('max_workers', 1)
        field_name = self.config.get('field', 'output')
        
        print(f"Using {max_workers} worker(s) for parallel processing")
        
        # Read all lines and prepare tasks
        with open(dataset, 'r', encoding='utf-8') as f:
            lines = [line for line in f]
        
        # Prepare task parameters
        tasks = [(line, field_name) for line in lines]
        
        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Use tqdm to display progress bar
            results = list(tqdm(
                executor.map(_process_single_line, tasks),
                total=num_lines,
                desc=self.config.get('name', 'ThinkOrNotScorer')
            ))
        
        return results
