import json
import re
from typing import Dict, List
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from .base_scorer import BaseScorer
from .utils import get_total_lines


# Helper functions for multiprocessing (must be at module level for pickling)
def _contains_think_tag(text: str, compiled_patterns: List) -> bool:
    """
    Check if text contains think tags
    
    Args:
        text: Text to check
        compiled_patterns: List of compiled regex patterns
        
    Returns:
        bool: True if contains think tags, False otherwise
    """
    if not text or not isinstance(text, str):
        return False
    
    # Check all patterns
    for pattern in compiled_patterns:
        if pattern.search(text):
            return True
    
    return False


def _extract_think_content(text: str) -> str:
    """
    Extract content within think tags
    
    Args:
        text: Text to extract from
        
    Returns:
        str: Content within think tags, empty string if no tags found
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Try to match <think>...</think>
    pattern1 = r'<think\s*>(.*?)</think\s*>'
    match1 = re.search(pattern1, text, re.IGNORECASE | re.DOTALL)
    if match1:
        return match1.group(1).strip()
    
    # Try to match <redacted_reasoning>...</redacted_reasoning>
    pattern2 = r'<redacted_reasoning\s*>(.*?)</redacted_reasoning\s*>'
    match2 = re.search(pattern2, text, re.IGNORECASE | re.DOTALL)
    if match2:
        return match2.group(1).strip()
    
    return ""


def _remove_think_content(text: str) -> str:
    """
    Remove think tags and their content, return the remaining text
    
    Args:
        text: Text to process
        
    Returns:
        str: Text with think sections removed
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Remove <think>...</think> tags and their content
    pattern1 = r'<think\s*>.*?</think\s*>'
    result = re.sub(pattern1, '', text, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove <redacted_reasoning>...</redacted_reasoning> tags and their content
    pattern2 = r'<redacted_reasoning\s*>.*?</redacted_reasoning\s*>'
    result = re.sub(pattern2, '', result, flags=re.IGNORECASE | re.DOTALL)
    
    return result.strip()


def _has_code_block(text: str) -> bool:
    """
    Check if text contains code blocks (markdown format)
    
    Args:
        text: Text to check
        
    Returns:
        bool: True if code blocks exist, False otherwise
    """
    if not text or not isinstance(text, str):
        return False
    
    # Match markdown code block pattern: ```language ... ``` or ``` ... ```
    pattern = r'```[a-zA-Z0-9+\-#.]*\s*\n?.*?```'
    
    # Use re.DOTALL to make . match newlines
    match = re.search(pattern, text, re.DOTALL)
    
    return match is not None


def _score_item(data_item: Dict, field_name: str, compiled_patterns: List) -> float:
    """
    Score a single data item
    
    Args:
        data_item: Data item dictionary
        field_name: Field name to extract text from
        compiled_patterns: List of compiled regex patterns
        
    Returns:
        float: 
            - If no think section exists, return -2
            - If think section exists, after removing think section, check if code blocks can be parsed:
                - If not, return -1
            - If think section exists, after removing think section, if code blocks can be parsed:
                - Further check if code blocks also exist in think section, return 0 if yes, otherwise return 1
    """
    # Extract text from specified field
    text = data_item.get(field_name, "")
    
    if not text or not isinstance(text, str):
        # If field doesn't exist or is empty, return -2 (treated as no think section)
        return -2.0
    
    # Case 1: Check if think section exists
    if not _contains_think_tag(text, compiled_patterns):
        return -2.0
    
    # Case 2 and 3: Think section exists
    # Extract think section content
    think_content = _extract_think_content(text)
    
    # Remove think section, get remaining text
    remaining_text = _remove_think_content(text)
    
    # Case 2: Check if code blocks can be parsed after removing think section
    if not _has_code_block(remaining_text):
        return -1.0
    
    # Case 3: After removing think section, code blocks can be parsed
    # Further check if code blocks also exist in think section
    if _has_code_block(think_content):
        return 0.0
    else:
        return 1.0


def _process_single_line(args):
    """Helper function to process a single line (for multiprocessing)
    
    Args:
        args: Tuple of (line, field_name, compiled_patterns)
    
    Returns:
        Dict containing id and PureThink_Score
    """
    line, field_name, compiled_patterns = args
    
    try:
        item = json.loads(line.strip())
        score = _score_item(item, field_name, compiled_patterns)
        
        return {
            "id": item.get("id", ""),
            "score": score
        }
    except Exception as e:
        # If processing fails, return a result with error marker
        return {
            "id": item.get("id", "unknown") if 'item' in locals() else "unknown",
            "score": -2.0,
            "error": str(e)
        }


class PureThinkScorer(BaseScorer):
    def _validate_config(self):
        """Validate configuration parameters"""
        # Check specified field to examine
        if "field" not in self.config or not isinstance(self.config["field"], str):
            self.config["field"] = "output"
            print("Warning: No/invalid field specified, use default value of 'output'.")
        else:
            print(f"Using specified field: {self.config['field']}.")
        
        # Check max_workers
        if "max_workers" not in self.config or not isinstance(self.config["max_workers"], int) or self.config["max_workers"] <= 0:
            import os
            # Default to CPU core count
            default_workers = max(1, os.cpu_count() or 1)
            print(f"Warning: No/invalid max_workers, using default value of {default_workers} (CPU count).")
            self.config["max_workers"] = default_workers
        else:
            print(f"Using specified max_workers: {self.config['max_workers']}.")

    def _setup(self):
        """Initialize PureThinkScorer"""
        # Define think tag patterns to detect
        # Support <think>...</think> and <redacted_reasoning>...</redacted_reasoning>
        self.think_patterns = [
            r'<think\s*>',
            r'</think\s*>',
            r'<think>',
            r'</think>',
            r'<redacted_reasoning\s*>',
            r'</redacted_reasoning\s*>',
        ]
        
        # Compile regex patterns for better performance
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.think_patterns]
        
        print("Setting up PureThinkScorer successfully")

    def score_item(self, data_item: Dict) -> float:
        """Score a single data item"""
        field_name = self.config["field"]
        return _score_item(data_item, field_name, self.compiled_patterns)

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
        tasks = [(line, field_name, self.compiled_patterns) for line in lines]
        
        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Use tqdm to display progress bar
            results = list(tqdm(
                executor.map(_process_single_line, tasks),
                total=num_lines,
                desc=self.config.get('name', 'PureThinkScorer')
            ))
        
        return results
