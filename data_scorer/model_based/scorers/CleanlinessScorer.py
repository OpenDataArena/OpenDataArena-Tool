#!/usr/bin/env python3
"""
Cleanliness Scorer for Meta-rater framework.

This scorer evaluates the format quality and noise-free content
using a trained classification model with 6 labels (0-5).
"""

import json
import os
from typing import Dict, List

# Disable flash-attn before importing transformers to avoid GLIBC version errors
os.environ.setdefault("DISABLE_FLASH_ATTN", "1")
# Block the import of flash-attn
import sys
sys.modules['flash_attn'] = None
sys.modules['flash_attn_2_cuda'] = None

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .base_scorer import BaseScorer
from .utils import get_total_lines


class CleanlinessScorer(BaseScorer):
    """Compute Cleanliness scores for text using Meta-rater classification model.

    The Cleanliness metric evaluates the format quality and noise-free content.
    The model outputs a classification score from 0 to 5.
    """

    def _validate_config(self):
        """Validate configuration parameters"""
        # Validate model path
        if "model" not in self.config:
            self.config["model"] = "opendatalab/meta-rater-cleanliness-rating"
            print(f"Using default model: {self.config['model']}")
        else:
            print(f"Using model: {self.config['model']}")

        # Validate batch_size
        if "batch_size" not in self.config or not isinstance(self.config["batch_size"], int) or self.config["batch_size"] <= 0:
            self.config["batch_size"] = 16
            print("Warning: No/invalid batch_size, use default value of 16.")
        else:
            print(f"Using specified batch_size: {self.config['batch_size']}.")
        
        # Validate max_model_len
        if "max_model_len" not in self.config:
            self.config["max_model_len"] = 8192
            print("Warning: No max_model_len specified, use default value of 8192.")
        else:
            print(f"Using specified max_model_len: {self.config['max_model_len']}.")

    def _setup(self):
        """Initialize model and tokenizer"""
        try:
            # Detect available device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Detected available device: {self.device}")
            if not torch.cuda.is_available():
                print("Warning: No CUDA device detected, will run on CPU which may be very slow.")
            
            # Load model
            print("Loading model, this may take some time...")
            # Ensure flash-attn is disabled to avoid GLIBC version errors
            os.environ["DISABLE_FLASH_ATTN"] = "1"
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config["model"],
                num_labels=6,
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
            
            # Move model to device
            self.model.to(self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Save max length
            self.max_model_len = self.config["max_model_len"]
            print(f"Setting up CleanlinessScorer successfully. Max model length: {self.max_model_len}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model: {e}") from e

    def score_item(self, data_item: Dict) -> Dict:
        """Score a single sample
        
        Returns:
            Dictionary containing Cleanliness_score
        """
        return self.score_batch([data_item])[0]

    @torch.no_grad()
    def score_batch(self, data_items: List[Dict]) -> List[Dict]:
        """Score a batch of samples

        Args:
            data_items: List of dictionaries containing instruction, input and output fields

        Returns:
            List of dictionaries containing Cleanliness_score
        """
        # Use fixed field names: instruction + "\n" + input + "\n" + output as content
        batch_contents = []
        
        for item in data_items:
            instruction = item["instruction"]
            input_text = item.get("input", "")
            output_text = item["output"]
            
            # Combine instruction, input and output as content
            if input_text:
                content = instruction + "\n" + input_text + "\n" + output_text
            else:
                content = instruction + "\n" + output_text
            batch_contents.append(content)
        
        return self._calculate_score_for_batch(batch_contents)

    def _calculate_score_for_batch(
        self,
        batch_contents: List[str],
    ) -> List[Dict]:
        """Calculate Cleanliness scores for a batch of samples

        Args:
            batch_contents: List of text contents

        Returns:
            List of dictionaries containing score
        """
        # Tokenize batch
        batch_size = len(batch_contents)
        if batch_size == 1:
            # Single sample processing: no padding, keep original length
            inputs = self.tokenizer(
                batch_contents[0],
                return_tensors="pt",
                truncation=True,
                max_length=self.max_model_len,
                padding=False
            ).to(self.device)
            
            # Check if truncation occurred
            if inputs['input_ids'].shape[1] >= self.max_model_len:
                print(f"Warning: input text exceeds max length {self.max_model_len}, truncated")
        else:
            # Multi-sample processing: use padding to ensure batch processing
            inputs = self.tokenizer(
                batch_contents,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_model_len
            ).to(self.device)
            
            # Check if any sample was truncated
            seq_lengths = inputs['attention_mask'].sum(dim=1)
            truncated_count = (seq_lengths >= self.max_model_len).sum().item()
            if truncated_count > 0:
                print(f"Warning: batch has {truncated_count}/{batch_size} samples exceeding max length {self.max_model_len}, truncated")
        
        # Batch inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # (batch_size, num_labels)
            
            # Calculate probability distribution
            probabilities = F.softmax(logits, dim=-1)
            
            # Calculate score using weighted average: class index * probability
            # Classes 0-5 correspond to scores 0-5
            class_indices = torch.arange(6, device=self.device).float()
            scores = torch.sum(probabilities * class_indices, dim=1).cpu().tolist()
        
        # Build results
        results = []
        for score in scores:
            results.append({
                "score": float(score)
            })
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results

    def evaluate(self, dataset: str) -> List[Dict]:
        """Score the entire dataset

        Args:
            dataset: Path to JSONL dataset file

        Returns:
            List of dictionaries containing id and Cleanliness_score
        """
        num_lines = get_total_lines(dataset)
        results: List[Dict] = []

        batch_size = self.config.get("batch_size")
        buffer_items = []
        buffer_ids = []

        with open(dataset, 'r', encoding='utf-8') as f:
            pbar = tqdm(total=num_lines, desc=self.config.get('name', 'CleanlinessScorer'))
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line.strip())
                buffer_items.append(item)
                buffer_ids.append(item.get("id", ""))

                if len(buffer_items) == batch_size:
                    batch_results = self.score_batch(buffer_items)
                    results.extend([
                        {"id": id_, **result}
                        for id_, result in zip(buffer_ids, batch_results)
                    ])
                    pbar.update(len(buffer_items))
                    buffer_items.clear()
                    buffer_ids.clear()

            if buffer_items:
                batch_results = self.score_batch(buffer_items)
                results.extend([
                    {"id": id_, **result}
                    for id_, result in zip(buffer_ids, batch_results)
                ])
                pbar.update(len(buffer_ids))
            pbar.close()

        return results

