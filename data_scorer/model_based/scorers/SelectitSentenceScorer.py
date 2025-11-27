import torch
import numpy as np
from .base_scorer import BaseScorer
import json
from typing import Dict, List, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from tqdm import tqdm
from .utils import get_total_lines


class SelectitSentenceScorer(BaseScorer):
    def _validate_config(self):
        # Check if the config specifies a local model path; if not, use the default remote download
        if "model" not in self.config:
            print(
                "Warning: No local model specified in config. Downloading the remote huggingface model.")
            self.config["model"] = "princeton-nlp/QuRater-1.3B"
        else:
            if self.config['model'] == 'princeton-nlp/QuRater-1.3B':
                print("Downloading and use the specific remote huggingface model.")
            elif not os.path.exists(self.config["model"]):
                print(
                    f"Warning: Specified local model path '{self.config['model']}' does not exist. "
                    "Downloading the remote huggingface model: princeton-nlp/QuRater-1.3B"
                )
                self.config["model"] = "princeton-nlp/QuRater-1.3B"
            else:
                print(
                    f"Using specified local model: '{self.config['model']}'.")
        
        if "rp_file" in self.config:
            print(
                f"Using specified rp_file: {self.config['rp_file']}.")
        else:
            print(
                "Warning: No specific rp_file, use default rp_file.")
            self.config['rp_file'] = 'scorers/SelectIT_rating_prompt.txt'
        
        # k: number of rating prompt templates to use per sample
        if "k" in self.config and isinstance(self.config["k"], int) and self.config["k"] > 0:
            print(f"Using specified k (number of rating prompts per sample): {self.config['k']}.")
        else:
            self.config['k'] = 5
            print("Warning: No/invalid k specified, use default value of 5.")
        
        # alpha: parameter to adjust score, controlling the impact of std on final score (score = avg / (1 + alpha * std))
        if "alpha" in self.config and isinstance(self.config["alpha"], (int, float)) and self.config["alpha"] >= 0:
            print(f"Using specified alpha (std adjustment parameter): {self.config['alpha']}.")
        else:
            self.config['alpha'] = 0.2
            print("Warning: No/invalid alpha specified, use default value of 0.2.")
        
        if "max_length" in self.config and isinstance(self.config["max_length"], int) and 0 < self.config["max_length"] <= 2048:
            print(f"Using specified max_length: {self.config['max_length']}.")
        elif "max_length" in self.config and isinstance(self.config["max_length"], int) and self.config["max_length"] <= 0:
            print(
                "Warning: the specific max_length should > 0. use default value of 512.")
            self.config['max_length'] = 512
        elif "max_length" in self.config and isinstance(self.config["max_length"], int) and self.config["max_length"] > 2048:
            print(
                "Warning: the specific max_length should not be larger than 2048. use default value of 512.")
            self.config['max_length'] = 512
        else:
            print("Warning: No specific max_length, use default value of 512.")
            self.config['max_length'] = 512
        
        if "batch_size" not in self.config or not isinstance(self.config["batch_size"], int) or self.config["batch_size"] <= 0:
            self.config["batch_size"] = 16
            print("Warning: No/invalid batch_size, use default value of 16.")
        else:
            print(f"Using specified batch_size: {self.config['batch_size']}.")

    def _setup(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config["model"],
                torch_dtype=torch.float16,
            )
            to_use_fast = "bloom" in self.config["model"].lower()
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config["model"], use_fast=to_use_fast)
        except Exception as e:
            print(
                f"Load specified model failed ({e}), fall back to princeton-nlp/QuRater-1.3B")
            self.model = AutoModelForCausalLM.from_pretrained(
                'princeton-nlp/QuRater-1.3B',
                torch_dtype=torch.float16,
            )
            to_use_fast = False
            self.tokenizer = AutoTokenizer.from_pretrained(
                'princeton-nlp/QuRater-1.3B', use_fast=to_use_fast)

        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.to(self.device)
        self.model.eval()
        
        # Load rating prompt file
        with open(self.config['rp_file'], "r", encoding="utf-8") as f_rp:
            self.rp_lines = f_rp.readlines()
        
        # Dynamically get token IDs for rating levels (1,2,3,4,5)
        # Different model tokenizers will encode numbers to different token IDs
        self.target_token_ids = []
        for rating in ["1", "2", "3", "4", "5"]:
            token_ids = self.tokenizer.encode(rating, add_special_tokens=False)
            if len(token_ids) > 0:
                self.target_token_ids.append(token_ids[0])
            else:
                print(f"Warning: Could not tokenize rating '{rating}'")
        
        print(f"Target token IDs for ratings 1-5: {self.target_token_ids}")
        print("Setting up SelectitSentenceScorer successfully")

    def build_rating_prompts_for_item(self, item: Dict) -> List[str]:
        """Build k rating prompts for a single data item.

        Args:
            item (dict): Data item containing instruction and response/output.

        Returns:
            list[str]: k rating prompts.
        """
        instruction_tag = "Instruction:"
        response_tag = "Response:"
        suffix = "\nThe answer is: \n"
        
        ins = item["instruction"]
        res = item["output"]
        input=item.get("input", "")
        if input:
            ins = ins + "\n" + input
            
        rating_prompts = []
        for idx in range(self.config['k']):
            rp = (
                self.rp_lines[idx].rstrip("\n")
                + "\n" + instruction_tag + ins
                + "\n" + response_tag + res
                + suffix
            )
            rating_prompts.append(rp)
        return rating_prompts

    def score_item(self, data_item: Dict) -> float:
        """Score a single data item"""
        return self.score_batch([data_item])[0]

    def score_batch(self, data_items: List[Dict]) -> List[float]:
        """Score a batch of data items
        
        Args:
            data_items: List of data items
            
        Returns:
            List of scores for each data item
        """
        # Generate k rating prompts for each data item
        all_prompts = []
        valid_indices = []
        
        # Initialize all scores to default value 3.0 (for invalid data)
        scores = [3.0] * len(data_items)
        
        for idx, item in enumerate(data_items):               
            prompts = self.build_rating_prompts_for_item(item)
            # Only consider as valid data when prompts is not empty
            if prompts:
                all_prompts.extend(prompts)
                valid_indices.append(idx)
        
        # If no valid data items, return default scores
        if len(valid_indices) == 0:
            return scores
        
        # Check for truncation: calculate token lengths before tokenization
        truncation_count = 0
        for prompt in all_prompts:
            token_length = len(self.tokenizer.encode(prompt, add_special_tokens=True))
            if token_length > self.config["max_length"]:
                truncation_count += 1
        
        if truncation_count > 0:
            print(f"Warning: {truncation_count}/{len(all_prompts)} prompts exceed max_length "
                  f"({self.config['max_length']}) and will be truncated.")
        
        # Batch inference: process all prompts at once
        probability_vectors = []
        
        # Tokenize and infer all prompts at once
        inputs = self.tokenizer(
            all_prompts, 
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config["max_length"]
        ).to(self.device)
        
        with torch.no_grad():
            logits = self.model(**inputs).logits[:, -1, :]  # shape: [batch_size, vocab_size]
        
        # Calculate softmax probabilities in batch
        probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()  # shape: [batch_size, vocab_size]
        
        # Extract target token probabilities for each sample
        for i in range(len(all_prompts)):
            probability_vectors.append([
                float(probs[i, tid]) for tid in self.target_token_ids
            ])
        
        # Normalize probability vectors (ensure sum equals 1)
        pro_normalized = []
        for vec in probability_vectors:
            vec_sum = sum(vec)
            if vec_sum > 0:
                # Normalize to make probability sum equal to 1
                normalized = [v / vec_sum for v in vec]
            else:
                # If all probabilities are 0, use uniform distribution
                normalized = [1.0 / len(vec) for _ in vec]
            pro_normalized.append(normalized)
        
        # Calculate sentence-level score for each data item
        alpha = self.config['alpha']
        k = self.config['k']
        
        for batch_idx, orig_idx in enumerate(valid_indices):
            token_scores = []
            for j in range(k):
                vec_idx = batch_idx * k + j
                vec = pro_normalized[vec_idx]
                
                # Calculate score using expected value: score = sum(rating * probability)
                # ratings from 1 to 5 correspond to vec[0] to vec[4]
                expected_score = sum((i + 1) * vec[i] for i in range(len(vec)))
                token_scores.append(expected_score)
            
            # Adjust final score using mean and standard deviation
            avg = float(np.average(token_scores))
            std = float(np.std(token_scores))
            # alpha penalizes inconsistency: if k prompts give very different scores, lower the final score
            scores[orig_idx] = avg / (1 + alpha * std)
        
        return scores

    def evaluate(self, dataset) -> List[Dict]:
        """Evaluate the entire dataset"""
        num_lines = get_total_lines(dataset)
        results: List[Dict] = []
        
        batch_size = self.config["batch_size"]
        buffer_items = []
        buffer_ids = []
        
        with open(dataset, 'r', encoding='utf-8') as f:
            pbar = tqdm(total=num_lines, desc=self.config.get(
                'name', 'SelectitSentenceScorer'))
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
            
            # Process remaining data
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
