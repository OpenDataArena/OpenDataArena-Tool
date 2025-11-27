import torch
from .base_scorer import BaseScorer
import json
from typing import Dict, List, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import numpy as np
from tqdm import tqdm
from .utils import get_total_lines


class SelectitTokenScorer(BaseScorer):
    def _validate_config(self):
        # Check if a local model path is specified in config. If not, use default remote download and notify user.
        if "model" not in self.config:
            print(
                "Warning: No local model specified in config. Downloading the remote huggingface model.")
            self.config["model"] = "meta-llama/Llama-3.1-8B"
        else:
            print(f"Using specified local model: '{self.config['model']}'. ")
        
        if "rp_file" in self.config:
            print(
                f"Using specified rp_file: {self.config['rp_file']}.")
        else:
            print(
                "Warning: No specific rp_file, use default rp_file.")
            self.config['rp_file'] = 'scorers/SelectIT/rating_prompt.txt'
        
        if "k" in self.config and isinstance(self.config["k"], int) and self.config["k"] > 0:
            print(f"Using specified k (number of rating prompts per sample): {self.config['k']}.")
        else:
            self.config['k'] = 5
            print("Warning: No/invalid k specified, use default value of 5.")
        
        # alpha: parameter for adjusting score based on standard deviation (score = avg / (1 + alpha * std))
        if "alpha" in self.config and isinstance(self.config["alpha"], (int, float)) and self.config["alpha"] >= 0:
            print(f"Using specified alpha (std adjustment parameter): {self.config['alpha']}.")
        else:
            self.config['alpha'] = 0.2
            print("Warning: No/invalid alpha specified, use default value of 0.2.")
        
        if "max_length" in self.config and isinstance(self.config["max_length"], int) and self.config["max_length"] > 0:
            print(f"Using specified max_length: {self.config['max_length']}.")
        elif "max_length" in self.config and isinstance(self.config["max_length"], int) and self.config["max_length"] <= 0:
            print(
                "Warning: the specific max_length should > 0. use default value of 2048.")
            self.config['max_length'] = 2048
        else:
            print("Warning: No specific max_length, use default value of 2048.")
            self.config['max_length'] = 2048
        
        if "batch_size" not in self.config or not isinstance(self.config["batch_size"], int) or self.config["batch_size"] <= 0:
            self.config["batch_size"] = 8
            print("Warning: No/invalid batch_size, use default value of 8.")
        else:
            print(f"Using specified batch_size: {self.config['batch_size']}.")

    def _setup(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config["model"], torch_dtype=torch.float16
            )
            to_use_fast = "bloom" in self.config["model"].lower()
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config["model"], use_fast=to_use_fast)
        except Exception as e:
            print(
                f"Load specified model failed ({e}), fall back to meta-llama/Llama-3.1-8B")
            self.model = AutoModelForCausalLM.from_pretrained(
                'meta-llama/Llama-3.1-8B', torch_dtype=torch.float16
            )
            to_use_fast = False
            self.tokenizer = AutoTokenizer.from_pretrained(
                'meta-llama/Llama-3.1-8B', use_fast=to_use_fast)
        
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.pad_token or self.tokenizer.eos_token
        self.model.to(self.device)
        self.model.eval()
        
        # Load rating prompts
        with open(self.config['rp_file'], "r", encoding="utf-8") as f_rp:
            self.rp_lines = f_rp.readlines()
        
        # Dynamically obtain token IDs corresponding to rating levels (1,2,3,4,5)
        # Different model tokenizers may encode numbers as different token IDs
        self.rating_token_ids = []
        for rating in ["1", "2", "3", "4", "5"]:
            token_ids = self.tokenizer.encode(rating, add_special_tokens=False)
            if len(token_ids) > 0:
                self.rating_token_ids.append(token_ids[0])
            else:
                print(f"Warning: Could not tokenize rating '{rating}'")
        
        print(f"Rating token IDs for 1-5: {self.rating_token_ids}")
        print("Setting up SelectitTokenScorer successfully")

    def build_rating_prompts_for_item(self, item: Dict[str, Any]) -> List[str]:
        """Build k rating prompts for a single data item

        Args:
            item: Dictionary containing 'instruction' and 'output' keys.

        Returns:
            List[str]: k rating prompts. Returns empty list if data item is invalid.
        """
        # Validate required fields
        if "instruction" not in item or "output" not in item:
            return []
        
        instruction_tag = "Instruction:"
        response_tag = "Response:"
        suffix = "\nThe answer is: \n"
        
        ins = item["instruction"]
        res = item["output"]
        input_text = item.get("input", "")
        if input_text:
            ins = ins + "\n" + input_text
        
        rating_prompts = []
        for idx in range(self.config['k']):
            prompt = (
                self.rp_lines[idx].rstrip("\n")
                + "\n" + instruction_tag + ins
                + "\n" + response_tag + res
                + suffix
            )
            rating_prompts.append(prompt)
        
        return rating_prompts

    def score_item(self, data_item: Dict[str, Any]) -> float:
        """Score a single data item."""
        return self.score_batch([data_item])[0]

    def score_batch(self, data_items: List[Dict]) -> List[float]:
        """Score a batch of data items.
        
        Args:
            data_items: List of data dictionaries.
            
        Returns:
            List[float]: List of scores for each data item.
        """
        # Generate k rating prompts for each data item
        all_prompts = []
        valid_indices = []
        
        # Initialize all scores to default value 3.0 (for invalid data)
        scores = [3.0] * len(data_items)
        
        for idx, item in enumerate(data_items):
            prompts = self.build_rating_prompts_for_item(item)
            # Only consider it as valid data when prompts is not empty
            if prompts:
                all_prompts.extend(prompts)
                valid_indices.append(idx)
        
        # If no valid data items, return default scores
        if len(valid_indices) == 0:
            return scores
        
        # Batch tokenize all prompts
        tokenized = self.tokenizer(
            all_prompts, 
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.config["max_length"]
        ).to(self.device)
        
        # Check for truncation and issue warning
        input_lengths = (tokenized.attention_mask.sum(dim=1)).cpu().tolist()
        max_input_length = max(input_lengths)
        if max_input_length >= self.config["max_length"]:
            num_truncated = sum(1 for length in input_lengths if length >= self.config["max_length"])
            print(f"Warning: {num_truncated} out of {len(all_prompts)} prompts were truncated to max_length={self.config['max_length']}. "
                  f"Consider increasing max_length for better results.")
        
        # Batch inference
        with torch.no_grad():
            outputs = self.model(**tokenized)
            logits = outputs.logits[:, -1, :]  # shape: [batch_size, vocab_size]
            softmax_logits = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        
        # Extract target token probabilities for each sample and normalize
        probability_vectors = []
        for i in range(len(all_prompts)):
            probs = [float(softmax_logits[i, tid]) for tid in self.rating_token_ids]
            
            # Normalize probability vector (ensure sum is 1)
            prob_sum = sum(probs)
            if prob_sum > 0:
                probs = [p / prob_sum for p in probs]
            else:
                # If all probabilities are 0, use uniform distribution
                probs = [1.0 / len(probs) for _ in probs]
            
            probability_vectors.append(probs)
        
        # Calculate sentence-level score for each data item
        alpha = self.config['alpha']
        k = self.config['k']
        
        for batch_idx, orig_idx in enumerate(valid_indices):
            token_scores = []
            for j in range(k):
                vec_idx = batch_idx * k + j
                probs = probability_vectors[vec_idx]
                
                # Calculate score using expected value: score = sum(rating * probability)
                # rating ranges from 1 to 5, corresponding to probs[0] to probs[4]
                expected_score = sum((i + 1) * probs[i] for i in range(len(probs)))
                token_scores.append(expected_score)
            
            # Adjust final score using mean and standard deviation
            avg = float(np.average(token_scores))
            std = float(np.std(token_scores))
            # alpha is used to penalize inconsistency: if k prompts give scores with large variance, lower the final score
            scores[orig_idx] = avg / (1 + alpha * std)
        
        return scores

    def evaluate(self, dataset) -> List[Dict]:
        """Evaluate the entire dataset with batch processing and progress bar.
        
        Args:
            dataset: Path to the dataset file (JSONL format).
            
        Returns:
            List[Dict]: List of dictionaries with 'id' and 'score'.
        """
        num_lines = get_total_lines(dataset)
        results: List[Dict] = []

        batch_size = self.config.get("batch_size")
        buffer_items = []
        buffer_ids = []

        with open(dataset, 'r', encoding='utf-8') as f:
            pbar = tqdm(total=num_lines, desc=self.config.get(
                'name', 'SelectitTokenScorer'))
            for line in f:
                item = json.loads(line.strip())
                buffer_items.append(item)
                buffer_ids.append(item.get("id", ""))

                if len(buffer_items) == batch_size:
                    batch_scores = self.score_batch(buffer_items)
                    results.extend([
                        {"id": id_, "score": sc}
                        for id_, sc in zip(buffer_ids, batch_scores)
                    ])
                    buffer_items.clear()
                    buffer_ids.clear()
                pbar.update(1)

            # Process remaining samples
            if buffer_items:
                batch_scores = self.score_batch(buffer_items)
                results.extend([
                    {"id": id_, "score": sc}
                    for id_, sc in zip(buffer_ids, batch_scores)
                ])
                buffer_items.clear()
                buffer_ids.clear()
            pbar.close()

        return results
