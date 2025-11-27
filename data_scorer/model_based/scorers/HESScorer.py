#!/usr/bin/env python3
"""
Offline High-Entropy Sum (HES) scoring for existing Chain-of-Thought (CoT) completions.

This script computes the HES metric per sample by:
  1) Concatenating prompt + completion as a single input string.
  2) Running model forward pass to obtain logits for each token position.
  3) Computing token entropies from logits for the completion span only.
  4) Summing token entropies in the top p percentile (highest entropy), yielding the HES score.

Notes:
  - HES depends on high-quality token-level entropy calculations from logits.
  - For long inputs, this can be memory intensive. Consider smaller batches if
    you run into memory constraints.
"""

import json
import os
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base_scorer import BaseScorer
from .utils import get_total_lines


class HESScorer(BaseScorer):
    """Compute High-Entropy Sum (HES) scores for CoT completions using transformers.

    The HES metric is defined as the sum of token entropies for the top p percentile
    of the most uncertain (highest-entropy) tokens within the completion span.
    """

    def _validate_config(self):
        """Validate configuration parameters"""
        # Validate model path
        if "model" not in self.config:
            raise ValueError("Config must contain 'model' field specifying the model path.")
        else:
            print(f"Using model: {self.config['model']}")

        # Validate percentile_cutoff
        if "percentile_cutoff" not in self.config:
            self.config["percentile_cutoff"] = 0.005
            print("Warning: No percentile_cutoff specified, use default value of 0.005.")
        elif not (0.0 < self.config["percentile_cutoff"] < 1.0):
            print("Warning: percentile_cutoff must be in (0, 1). Using default value of 0.005.")
            self.config["percentile_cutoff"] = 0.005
        else:
            print(f"Using specified percentile_cutoff: {self.config['percentile_cutoff']}.")

        # Validate batch_size
        if "batch_size" not in self.config or not isinstance(self.config["batch_size"], int) or self.config["batch_size"] <= 0:
            self.config["batch_size"] = 8
            print("Warning: No/invalid batch_size, use default value of 8.")
        else:
            print(f"Using specified batch_size: {self.config['batch_size']}.")
        
        # Validate max_length
        if "max_length" not in self.config:
            self.config["max_length"] = 4096
            print("Warning: No max_length specified, use default value of 4096.")
        else:
            print(f"Using specified max_length: {self.config['max_length']}.")

    def _setup(self):
        """Initialize model and tokenizer"""
        try:
            # Detect available device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Detected available device: {self.device}")
            if not torch.cuda.is_available():
                print("Warning: No CUDA device detected. Will run on CPU, which will be very slow.")
            
            # Load model
            print("Loading model, this may take some time...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config["model"],
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config["model"],
                trust_remote_code=True,
                padding_side='left'  # Use left padding for batch processing
            )
            
            # Set pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Save maximum length
            self.max_length = self.config["max_length"]
            print(f"Setting up HESScorer successfully. Max length: {self.max_length}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model: {e}") from e

    def score_item(self, data_item: Dict) -> Dict:
        """Score a single sample
        
        Returns:
            Dictionary containing score, completion_token_length, entropy_threshold, and truncated
        """
        return self.score_batch([data_item])[0]

    @torch.no_grad()
    def score_batch(self, data_items: List[Dict]) -> List[Dict]:
        """Score a batch of samples

        Args:
            data_items: List of dictionaries containing instruction, input, and output fields

        Returns:
            List of dictionaries containing score, completion_token_length, entropy_threshold, and truncated
        """
        # Use fixed field names: instruction + "\n" + input as prompt, output as completion
        batch_prompts = []
        batch_completions = []
        
        for item in data_items:
            instruction=item["instruction"]
            input_text = item.get("input", "")
            output_text=item["output"]
            
            # Concatenate instruction and input with newline as prompt
            if input_text:
                prompt = instruction + "\n" + input_text
            else:
                prompt = instruction
            batch_prompts.append(prompt)
            batch_completions.append(output_text)
        
        return self._calculate_hes_for_batch(batch_prompts, batch_completions)

    def _calculate_hes_for_batch(
        self,
        batch_prompts: Sequence[str],
        batch_completions: Sequence[str],
    ) -> List[Dict]:
        """Compute HES scores for a batch.

        Steps:
          - Concatenate prompt + completion to form the full input per sample.
          - Check input length and truncate if exceeds max_length.
          - Run model forward pass to obtain logits for each token position.
          - Compute per-token entropy from logits for the completion span.
          - Aggregate the top p percentile entropies to get HES score.

        Returns a list of dictionaries containing score, completion_token_length, 
        entropy_threshold, and truncated flag.
        """
        if len(batch_prompts) != len(batch_completions):
            raise ValueError("batch_prompts and batch_completions must have the same length")

        # Prepare full inputs and check/truncate length if necessary
        full_texts: List[str] = []
        prompt_token_lengths: List[int] = []
        truncated_flags: List[bool] = []  # Track which samples were truncated
        
        for p, c in zip(batch_prompts, batch_completions):
            # Tokenize prompt alone to know where completion starts
            prompt_tokens = self.tokenizer.encode(p, add_special_tokens=False)
            prompt_len = len(prompt_tokens)
            
            # Tokenize full text (prompt + completion)
            full_text = p + c
            full_tokens = self.tokenizer.encode(full_text, add_special_tokens=False)
            full_len = len(full_tokens)
            
            # Check if truncation is needed
            if self.max_length is not None and full_len > self.max_length:
                # Need to truncate
                max_input_len = self.max_length
                
                if prompt_len >= max_input_len:
                    # Even prompt is too long, truncate prompt only
                    truncated_tokens = full_tokens[:max_input_len]
                    prompt_len = max_input_len
                else:
                    # Truncate completion part
                    truncated_tokens = full_tokens[:max_input_len]
                
                # Decode back to text
                full_text = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
                truncated_flags.append(True)
            else:
                truncated_flags.append(False)
            
            full_texts.append(full_text)
            prompt_token_lengths.append(prompt_len)

        # Tokenize batch for model input
        batch_size = len(full_texts)
        if batch_size == 1:
            # Single sample processing: no padding, keep original length
            message_inputs = self.tokenizer(
                full_texts[0],
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=False
            ).to(self.device)
        else:
            # Multi-sample processing: use padding for batch processing
            message_inputs = self.tokenizer(
                full_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            ).to(self.device)
        
        input_ids = message_inputs.input_ids
        attention_mask = message_inputs.attention_mask
        
        # Batch inference
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            logits = outputs.logits  # (batch_size, seq_len, vocab_size)

        # Process each sample
        results: List[Dict] = []
        for batch_idx in range(len(batch_prompts)):
            # Safety: If completion is empty, score is 0.
            if len(batch_completions[batch_idx]) == 0:
                results.append({
                    "score": 0.0,
                    "completion_token_length": 0,
                    "entropy_threshold": 0.0,
                    "truncated": truncated_flags[batch_idx]
                })
                continue
            
            # Get actual sequence length (excluding padding)
            seq_len = attention_mask[batch_idx].sum().item()
            
            # Calculate padding offset (for left padding case)
            # Total length - actual length = padding count
            total_len = input_ids.shape[1]
            padding_offset = total_len - seq_len
            
            # Starting position of completion in original sequence
            completion_start = prompt_token_lengths[batch_idx]
            
            # Calculate token entropies for completion part
            token_entropies: List[float] = []
            
            # Iterate through each token position in completion part
            # logits[batch_idx, i, :] are the logits for predicting token at position i+1
            # Note: i here is the index in the padded sequence
            for i in range(padding_offset, total_len - 1):
                # Index of current position in original sequence (after removing padding)
                position_in_original = i - padding_offset
                current_token_position = position_in_original + 1
                
                # Only process the completion part
                if current_token_position >= completion_start:
                    # Get logits for predicting current token
                    current_logits = logits[batch_idx, i, :]
                    
                    # Calculate entropy: add epsilon to prevent NaN from log(0)
                    epsilon = 1e-9
                    probabilities = F.softmax(current_logits, dim=-1)
                    entropy = -torch.sum(probabilities * torch.log2(probabilities + epsilon)).item()
                    
                    token_entropies.append(entropy)
            
            completion_token_length = len(token_entropies)
            
            # Aggregate top p percentile entropies
            if len(token_entropies) == 0:
                results.append({
                    "score": 0.0,
                    "completion_token_length": 0,
                    "entropy_threshold": 0.0,
                    "truncated": truncated_flags[batch_idx]
                })
                continue

            entropies_np = np.asarray(token_entropies, dtype=np.float32)

            # Compute percentile threshold for top p fraction
            # Example: p=0.005 -> threshold at 99.5th percentile
            threshold_percent = (1.0 - self.config["percentile_cutoff"]) * 100.0
            threshold = float(np.percentile(entropies_np, threshold_percent))

            # Select entropies >= threshold. Ensure at least one token contributes.
            selected = entropies_np[entropies_np >= threshold]
            if selected.size == 0:
                # Fallback to selecting the single maximum entropy token.
                selected = entropies_np[np.argmax(entropies_np)][None]

            hes_score = float(np.sum(selected, dtype=np.float64))
            
            results.append({
                "score": hes_score,
                "completion_token_length": completion_token_length,
                "entropy_threshold": threshold,
                "truncated": truncated_flags[batch_idx]
            })
        
        # Clean up CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results

    def evaluate(self, dataset: str) -> List[Dict]:
        """Score the entire dataset

        Args:
            dataset: Path to JSONL dataset file

        Returns:
            List of dictionaries containing id, score, completion_token_length, entropy_threshold, and truncated
        """
        num_lines = get_total_lines(dataset)
        results: List[Dict] = []

        batch_size = self.config.get("batch_size")
        buffer_items = []
        buffer_ids = []
        
        # Count truncated samples
        truncated_count = 0

        with open(dataset, 'r', encoding='utf-8') as f:
            pbar = tqdm(total=num_lines, desc=self.config.get('name', 'HESScorer'))
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line.strip())
                buffer_items.append(item)
                buffer_ids.append(item.get("id", ""))

                if len(buffer_items) == batch_size:
                    batch_results = self.score_batch(buffer_items)
                    # Count truncated samples
                    truncated_count += sum(1 for r in batch_results if r.get("truncated", False))
                    results.extend([
                        {"id": id_, **result}
                        for id_, result in zip(buffer_ids, batch_results)
                    ])
                    pbar.update(len(buffer_items))
                    buffer_items.clear()
                    buffer_ids.clear()

            if buffer_items:
                batch_results = self.score_batch(buffer_items)
                # Count truncated samples
                truncated_count += sum(1 for r in batch_results if r.get("truncated", False))
                results.extend([
                    {"id": id_, **result}
                    for id_, result in zip(buffer_ids, batch_results)
                ])
                pbar.update(len(buffer_ids))
            pbar.close()
        
        # Give warning if samples were truncated
        if truncated_count > 0:
            total_count = len(results)
            truncated_ratio = truncated_count / total_count * 100 if total_count > 0 else 0
            print(f"\nWarning: {truncated_count}/{total_count} ({truncated_ratio:.2f}%) samples were truncated due to exceeding max_length ({self.max_length}).")
            print(f"Suggestion: If the truncation ratio is high, consider increasing the max_length parameter for more accurate scoring results.")

        return results


