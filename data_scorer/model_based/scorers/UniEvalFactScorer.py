import sys
import os
import json
from typing import Dict, List
from tqdm import tqdm


from .UniEval_metric.evaluator import FactEvaluator

from .base_scorer import BaseScorer
from .utils import get_total_lines


class UniEvalFactScorer(BaseScorer):
    """Fact consistency evaluation scorer based on UniEval
    
    Supported dimensions:
    - consistency: Consistency (fact consistency)
    """
    
    def _validate_config(self):
        if "model" not in self.config:
            print("Warning: No model specified. Using default: MingZhong/unieval-fact")
            self.config['model'] = 'MingZhong/unieval-fact'
        else:
            print(f"Using specified model: {self.config['model']}")
        
        # Validate max_length
        if "max_length" not in self.config or not isinstance(self.config["max_length"], int) or self.config["max_length"] <= 0:
            self.config["max_length"] = 1024
            print("Warning: No/invalid max_length, use default value of 1024.")
        else:
            print(f"Using specified max_length: {self.config['max_length']}.")
        
        # Validate batch_size
        if "batch_size" not in self.config or not isinstance(self.config["batch_size"], int) or self.config["batch_size"] <= 0:
            self.config["batch_size"] = 8
            print("Warning: No/invalid batch_size, use default value of 8.")
        else:
            print(f"Using specified batch_size: {self.config['batch_size']}.")
        
        # Validate device
        if "device" not in self.config:
            import torch
            self.config["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            print(f"Using specified device: {self.config['device']}")
        
        # Validate cache_dir
        if "cache_dir" not in self.config:
            self.config["cache_dir"] = None

    def _setup(self):
        """Initialize UniEval evaluator"""
        model_path = self.config.get('model', 'MingZhong/unieval-fact')
        
        # Check if it's a local path and exists
        is_local = os.path.exists(model_path) and os.path.isdir(model_path)
        
        # If it's a local path, try offline mode first
        if is_local:
            try:
                print(f"Using local model path: {model_path}")
                self.evaluator = FactEvaluator(
                    model_name_or_path=model_path,
                    max_length=self.config['max_length'],
                    device=self.config['device'],
                    cache_dir=self.config.get('cache_dir'),
                    local_files_only=True
                )
                print("Setting up UniEvalFactScorer successfully with local model")
                return
            except Exception as e:
                print(f"Warning: Failed to load model in offline mode: {e}")
                print("Attempting to load with online fallback...")
        
        # Try online loading (if local path doesn't exist or offline mode failed)
        try:
            print(f"Attempting to load model from: {model_path}")
            self.evaluator = FactEvaluator(
                model_name_or_path=model_path,
                max_length=self.config['max_length'],
                device=self.config['device'],
                cache_dir=self.config.get('cache_dir'),
                local_files_only=False
            )
            print("Setting up UniEvalFactScorer successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize UniEval FactEvaluator: {e}") from e

    def _extract_data(self, data_item: Dict) -> Dict:
        """Extract evaluation information from data item"""
        # Extract instruction, input, output
        instruction = data_item["instruction"]
        input_text = data_item.get("input", "")
        output = data_item["output"]
        
        # For fact checking, source is the source document (instruction + input)
        source = f"{instruction}\n{input_text}".strip() if input_text else instruction
        
        return {
            "source": source,
            "output": output
        }

    def _check_truncation(self, text: str, data_id: str = "") -> bool:
        """Check if text will be truncated by tokenizer
        
        Args:
            text: Input text to check
            data_id: ID of the data item (for warning message)
        
        Returns:
            True if text will be truncated, False otherwise
        """
        tokenizer = self.evaluator.scorer.tokenizer
        max_length = self.config['max_length']
        
        # Tokenize without truncation to get actual length
        tokens = tokenizer(text, truncation=False, return_tensors='pt')
        actual_length = tokens['input_ids'].shape[1]
        
        if actual_length > max_length:
            print(f"Warning: Data item {data_id} exceeds max_length ({actual_length} > {max_length}). "
                  f"The input will be truncated.")
            return True
        return False

    def score_item(self, data_item: Dict) -> Dict[str, float]:
        """Evaluate a single data item
        
        Returns:
            Dictionary containing consistency scores
        """
        extracted = self._extract_data(data_item)
        
        # Check for truncation
        full_text = extracted["source"] + extracted["output"]
        self._check_truncation(full_text, data_item.get("id", "unknown"))
        
        # Prepare data format
        data = [{
            "source": extracted["source"],
            "system_output": extracted["output"]
        }]
        
        # Evaluate
        eval_scores = self.evaluator.evaluate(
            data,
            print_result=False
        )
        
        return eval_scores[0] if eval_scores else {}

    def evaluate(self, dataset: str) -> List[Dict]:
        """Evaluate the entire dataset"""
        num_lines = get_total_lines(dataset)
        results: List[Dict] = []
        
        # Read all data
        all_data = []
        all_ids = []
        all_full_texts = []
        
        with open(dataset, 'r', encoding='utf-8') as f:
            pbar = tqdm(total=num_lines, desc=self.config.get('name', 'UniEvalFactScorer'))
            try:
                for line in f:
                    item = json.loads(line.strip())
                    item_id = item.get("id", "")
                    all_ids.append(item_id)
                    
                    extracted = self._extract_data(item)
                    all_data.append({
                        "source": extracted["source"],
                        "system_output": extracted["output"]
                    })
                    
                    # Prepare full text for truncation check
                    full_text = extracted["source"] + extracted["output"]
                    all_full_texts.append((full_text, item_id))
                    
                    pbar.update(1)
            finally:
                pbar.close()
        
        # Check for truncations
        print("Checking for potential truncations...")
        truncation_count = 0
        for full_text, item_id in all_full_texts:
            if self._check_truncation(full_text, item_id):
                truncation_count += 1
        
        if truncation_count > 0:
            print(f"Warning: {truncation_count} out of {len(all_data)} data items exceed max_length and will be truncated.")
        
        # Batch evaluation
        batch_size = self.config.get("batch_size", 8)
        
        for i in tqdm(range(0, len(all_data), batch_size), desc="Evaluating batches"):
            batch_data = all_data[i:i+batch_size]
            
            # Evaluate current batch
            eval_scores = self.evaluator.evaluate(
                batch_data,
                print_result=False
            )
            
            # Save results
            for j, score_dict in enumerate(eval_scores):
                result = {"id": all_ids[i + j]}
                # Add consistency scores
                for dim, score in score_dict.items():
                    result[f"UniEval_Fact_{dim.capitalize()}"] = score
                results.append(result)
        
        return results


