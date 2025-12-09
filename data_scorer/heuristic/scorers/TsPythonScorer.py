from .base_scorer import BaseScorer
import json
from typing import Dict, List, Optional
from tqdm import tqdm
from .utils import get_total_lines
from concurrent.futures import ProcessPoolExecutor
import re
import importlib

# Try to import tree-sitter
try:
    from tree_sitter import Language, Parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    Parser = None
    Language = None


def _create_python_parser() -> Optional[Parser]:
    """Create Python parser"""
    if not TREE_SITTER_AVAILABLE:
        return None
    
    try:
        # Dynamically import Python language module
        lang_module = importlib.import_module("tree_sitter_python")
        
        # Create Language object
        lang_obj = Language(lang_module.language())
        
        # Create Parser
        parser = Parser(lang_obj)
        return parser
    except (ImportError, Exception):
        return None


def _extract_code_from_markdown(text: str) -> Optional[List[str]]:
    """Extract code blocks from markdown formatted text
    
    Returns:
        List[str]: List of code blocks, returns None if no code blocks found
    """
    # Match markdown code block pattern: ```language ... ``` or ``` ... ```
    pattern = r'```[a-zA-Z0-9+\-#.]*\s*\n?(.*?)```'
    
    # Use re.DOTALL to make . match newlines
    matches = re.findall(pattern, text, re.DOTALL)
    
    if matches:
        # Return all found code blocks
        code_blocks = []
        for code in matches:
            code = code.strip()
            if code:
                code_blocks.append(code)
        
        if code_blocks:
            return code_blocks
    
    # If no code blocks found, return None indicating the entire text might be code
    return None


def _parse_code_snippet(code: str, parser) -> bool:
    """Parse a single code snippet, returns True if syntax is correct, False if syntax error
    
    Args:
        code: Python code to parse
        parser: Tree-sitter Parser object
    """
    if not code or not code.strip():
        return False
    
    if parser is None:
        return False
    
    try:
        tree = parser.parse(bytes(code, "utf8"))
        root = tree.root_node
        return not root.has_error
    except Exception:
        return False


# Helper function for multiprocessing (must be at module level for pickling)
def _process_single_line(args):
    """Helper function to process a single line (for multiprocessing)
    
    Args:
        args: Tuple of (line, field_name)
    
    Returns:
        Dict containing id and score
    """
    line, field_name = args
    
    try:
        item = json.loads(line.strip())
        
        # Extract text from specified field
        text = item.get(field_name, "")
        
        if not text or not isinstance(text, str):
            # If field does not exist or is empty, return 0.0 (invalid)
            return {
                "id": item.get("id", ""),
                "score": 0.0
            }
        
        # Create parser for this process
        parser = _create_python_parser()
        
        # Try to extract code from markdown code blocks
        code_blocks = _extract_code_from_markdown(text)
        
        if code_blocks is not None:
            # Found code blocks, check syntax of all code blocks
            # Only return 1.0 if all code blocks have correct syntax
            for code in code_blocks:
                if not _parse_code_snippet(code, parser):
                    # If any code block has syntax error, return 0.0
                    return {
                        "id": item.get("id", ""),
                        "score": 0.0
                    }
            # All code blocks are correct
            score = 1.0 if len(code_blocks) > 0 else 0.0
        else:
            # No code blocks found, the entire text might be code
            # Use Python parser to parse
            score = 1.0 if _parse_code_snippet(text, parser) else 0.0
        
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


class TsPythonScorer(BaseScorer):
    def _validate_config(self):
        """Validate configuration parameters"""
        # Check the field to extract code from
        if "field" not in self.config or not isinstance(self.config["field"], str):
            self.config["field"] = "output"
            print("Warning: No/invalid field specified, use default value of 'output'.")
        else:
            print(f"Using specified field: {self.config['field']}.")
        
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
        """Initialize scorer"""
        print("Setting up TsPythonScorer successfully")

    def score_item(self, data_item: Dict) -> float:
        """Calculate syntax correctness score for a single data item"""
        # Extract text from specified field
        field_name = self.config["field"]
        text = data_item.get(field_name, "")
        
        if not text or not isinstance(text, str):
            # If field does not exist or is empty, return 0.0 (invalid)
            return 0.0
        
        # Create parser
        parser = _create_python_parser()
        
        # Try to extract code from markdown code blocks
        code_blocks = _extract_code_from_markdown(text)
        
        if code_blocks is not None:
            # Found code blocks, check syntax of all code blocks
            # Only return 1.0 if all code blocks have correct syntax
            for code in code_blocks:
                if not _parse_code_snippet(code, parser):
                    # If any code block has syntax error, return 0.0
                    return 0.0
            # All code blocks are correct
            return 1.0 if len(code_blocks) > 0 else 0.0
        else:
            # No code blocks found, the entire text might be code
            # Use Python parser to parse
            return 1.0 if _parse_code_snippet(text, parser) else 0.0

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
                desc=self.config.get('name', 'TsPythonScorer')
            ))
        
        return results

