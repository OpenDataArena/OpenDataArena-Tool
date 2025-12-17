import torch
import numpy as np
from .base_scorer import BaseScorer
import json
from typing import Dict, List, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from tqdm import tqdm
from .utils import get_total_lines


class SelectitModelScorer(BaseScorer):
    def _validate_config(self):
        """Validate configuration parameters"""
        # Check if multiple model paths are provided
        if "models" not in self.config:
            raise ValueError(
                "Error: 'models' must be specified in config. "
                "It should be a list of model paths, e.g., ['model1', 'model2', 'model3']"
            )
        
        if not isinstance(self.config["models"], list) or len(self.config["models"]) == 0:
            raise ValueError(
                "Error: 'models' should be a non-empty list of model paths."
            )
        
        print(f"Using {len(self.config['models'])} models for ensemble scoring:")
        for idx, model_path in enumerate(self.config["models"]):
            print(f"  Model {idx + 1}: {model_path}")
        
        # Check rating prompt file
        if "rp_file" in self.config:
            print(f"Using specified rp_file: {self.config['rp_file']}.")
        else:
            print("Warning: No specific rp_file, use default rp_file.")
            self.config['rp_file'] = 'scorers/SelectIT_rating_prompt.txt'
        
        # k: number of rating prompt templates per sample
        if "k" in self.config and isinstance(self.config["k"], int) and self.config["k"] > 0:
            print(f"Using specified k (number of rating prompts per sample): {self.config['k']}.")
        else:
            self.config['k'] = 5
            print("Warning: No/invalid k specified, use default value of 5.")
        
        # alpha: parameter for adjusting scores
        if "alpha" in self.config and isinstance(self.config["alpha"], (int, float)) and self.config["alpha"] >= 0:
            print(f"Using specified alpha (std adjustment parameter): {self.config['alpha']}.")
        else:
            self.config['alpha'] = 0.2
            print("Warning: No/invalid alpha specified, use default value of 0.2.")
        
        # model_weights: weights for each model
        if "model_weights" in self.config and isinstance(self.config["model_weights"], list):
            if len(self.config["model_weights"]) != len(self.config["models"]):
                print("Warning: model_weights length doesn't match models length, using equal weights.")
                self.config["model_weights"] = [1.0 / len(self.config["models"])] * len(self.config["models"])
            else:
                print(f"Using specified model_weights: {self.config['model_weights']}.")
        else:
            # Default to equal weights
            self.config["model_weights"] = [1.0 / len(self.config["models"])] * len(self.config["models"])
            print(f"Using equal weights for {len(self.config['models'])} models: {self.config['model_weights']}.")
        
        # max_length
        if "max_length" in self.config and isinstance(self.config["max_length"], int) and 0 < self.config["max_length"] <= 2048:
            print(f"Using specified max_length: {self.config['max_length']}.")
        elif "max_length" in self.config and isinstance(self.config["max_length"], int) and self.config["max_length"] <= 0:
            print("Warning: the specific max_length should > 0. use default value of 512.")
            self.config['max_length'] = 512
        elif "max_length" in self.config and isinstance(self.config["max_length"], int) and self.config["max_length"] > 2048:
            print("Warning: the specific max_length should not be larger than 2048. use default value of 512.")
            self.config['max_length'] = 512
        else:
            print("Warning: No specific max_length, use default value of 512.")
            self.config['max_length'] = 512
        
        # batch_size
        if "batch_size" not in self.config or not isinstance(self.config["batch_size"], int) or self.config["batch_size"] <= 0:
            self.config["batch_size"] = 16
            print("Warning: No/invalid batch_size, use default value of 16.")
        else:
            print(f"Using specified batch_size: {self.config['batch_size']}.")

    def _setup(self):
        """Initialize models and tokenizers"""
        # Load all models and corresponding tokenizers
        self.models = []
        self.tokenizers = []
        self.target_token_ids_list = []  # Token IDs for each model
        self.model_devices = []  # Store the device of each model's first layer
        
        for model_idx, model_path in enumerate(self.config["models"]):
            print(f'Loading Model weights from path: {model_path}')
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                
                to_use_fast = "bloom" in model_path.lower()
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path, use_fast=to_use_fast
                )
                tokenizer.padding_side = "left"
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                model.eval()
                self.models.append(model)
                self.tokenizers.append(tokenizer)
                
                # Get the device of the first parameter (for device_map="auto")
                model_device = next(model.parameters()).device
                self.model_devices.append(model_device)
                
                # Dynamically obtain token IDs corresponding to rating levels for current tokenizer
                target_token_ids = []
                for rating in ["1", "2", "3", "4", "5"]:
                    token_ids = tokenizer.encode(rating, add_special_tokens=False)
                    if len(token_ids) > 0:
                        target_token_ids.append(token_ids[0])
                    else:
                        print(f"Warning: Could not tokenize rating '{rating}' for model {model_idx + 1}")
                
                self.target_token_ids_list.append(target_token_ids)
                print(f"Successfully loaded model {model_idx + 1}: {model_path}")
                print(f"  Model device: {model_device}")
                print(f"  Rating token IDs for 1-5: {target_token_ids}")
            except Exception as e:
                raise RuntimeError(f"Failed to load model from {model_path}: {e}")
        
        # Load rating prompt file
        with open(self.config['rp_file'], "r", encoding="utf-8") as f_rp:
            self.rp_lines = f_rp.readlines()
        
        print("Setting up SelectitModelScorer successfully")

    def build_rating_prompts_for_item(self, item: Dict) -> List[str]:
        """Build k rating prompts for a single data item
        
        Args:
            item (dict): Data item containing instruction and response/output
            
        Returns:
            list[str]: k rating prompts
        """
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
            rp = (
                self.rp_lines[idx].rstrip("\n")
                + "\n" + instruction_tag + ins
                + "\n" + response_tag + res
                + suffix
            )
            rating_prompts.append(rp)
        return rating_prompts

    def score_with_single_model(
        self, 
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        prompts: List[str],
        target_token_ids: List[int],
        model_device: torch.device
    ) -> List[float]:
        """Score a batch of prompts using a single model
        
        Args:
            model: Model instance
            tokenizer: Tokenizer instance
            prompts: List of rating prompts
            target_token_ids: Rating token IDs (1,2,3,4,5) for this model
            model_device: The device where the model is located
            
        Returns:
            List of sentence-level scores for each data item
        """
        k = self.config['k']
        alpha = self.config['alpha']
        
        # Batch inference: process all prompts at once
        probability_vectors = []
        
        try:
            # Tokenize all prompts and perform inference
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config["max_length"]
            )
            
            # Check for truncation and warn if any prompt was truncated
            truncated_count = 0
            for i, input_ids in enumerate(inputs.input_ids):
                if len(input_ids) == self.config["max_length"]:
                    truncated_count += 1
            
            if truncated_count > 0:
                print(f"Warning: {truncated_count}/{len(prompts)} prompts were truncated to max_length={self.config['max_length']}. "
                      f"Consider increasing max_length for better accuracy.")
            
            # Move all input tensors to the model's device
            inputs = inputs.to(model_device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits[:, -1, :]  # shape: [batch_size, vocab_size]
                probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()  # shape: [batch_size, vocab_size]
                
                # Extract target token probabilities for each sample
                for i in range(len(prompts)):
                    tmp_res = [float(probs[i, tid]) for tid in target_token_ids]
                    probability_vectors.append(tmp_res)
        except Exception as ex:
            print(f"Error during batch inference: {ex}")
            # Use default uniform distribution
            probability_vectors = [[0.2, 0.2, 0.2, 0.2, 0.2] for _ in range(len(prompts))]
        
        # Normalize probability vectors (ensure sum = 1)
        pro_normalized = []
        for vec in probability_vectors:
            vec_sum = sum(vec)
            if vec_sum > 0:
                # Normalize to make probabilities sum to 1
                normalized = [v / vec_sum for v in vec]
            else:
                # If all probabilities are 0, use uniform distribution
                normalized = [1.0 / len(vec) for _ in vec]
            pro_normalized.append(normalized)
        
        # Calculate sentence-level scores for each data item
        data_num = len(pro_normalized) // k
        sentence_level_scores = []
        
        for idx in range(data_num):
            token_level_scores = []
            for j in range(k):
                vec_idx = idx * k + j
                vec = pro_normalized[vec_idx]
                
                # Calculate score using expected value: score = sum(rating * probability)
                # Ratings from 1 to 5 correspond to vec[0] to vec[4]
                expected_score = sum((i + 1) * vec[i] for i in range(len(vec)))
                token_level_scores.append(expected_score)
            
            # Adjust final score using mean and standard deviation
            avg = float(np.average(token_level_scores))
            std = float(np.std(token_level_scores))
            # Alpha penalizes inconsistency: if k prompts give very different scores, reduce final score
            sentence_level_scores.append(avg / (1 + alpha * std))
        
        return sentence_level_scores

    def score_item(self, data_item: Dict) -> float:
        """Score a single data item"""
        return self.score_batch([data_item])[0]

    def score_batch(self, data_items: List[Dict]) -> List[float]:
        """Score a batch of data items
        
        Args:
            data_items: List of data items
            
        Returns:
            List of weighted combined scores for each data item
        """
        # Generate k rating prompts for each data item
        all_prompts = []
        valid_indices = []
        
        for idx, item in enumerate(data_items):
            prompts = self.build_rating_prompts_for_item(item)
            all_prompts.extend(prompts)
            valid_indices.append(idx)
        
        # If no valid data items, return default scores
        scores = [3.0] * len(data_items)
        if len(valid_indices) == 0:
            return scores
        
        # Score using each model
        model_level_scores = []
        for model, tokenizer, target_token_ids, model_device in zip(
            self.models, self.tokenizers, self.target_token_ids_list, self.model_devices
        ):
            model_scores = self.score_with_single_model(
                model, tokenizer, all_prompts, target_token_ids, model_device
            )
            model_level_scores.append(model_scores)
        
        # Combine scores from multiple models (weighted average)
        # Positions in valid_indices correspond to indices in model_level_scores
        for position, original_idx in enumerate(valid_indices):
            combined_score = sum(
                model_level_scores[model_idx][position] * weight
                for model_idx, weight in enumerate(self.config["model_weights"])
            )
            scores[original_idx] = combined_score
        
        return scores

    def evaluate(self, dataset) -> List[Dict]:
        """Evaluate the entire dataset"""
        num_lines = get_total_lines(dataset)
        results: List[Dict] = []
        
        batch_size = self.config["batch_size"]
        buffer_items = []
        buffer_ids = []
        
        with open(dataset, 'r', encoding='utf-8') as f:
            pbar = tqdm(total=num_lines, desc=self.config.get('name', 'SelectitModelScorer'))
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