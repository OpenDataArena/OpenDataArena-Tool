import torch
from .base_scorer import BaseScorer
import json
from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from tqdm import tqdm
from .utils import get_total_lines


class PPLScorer(BaseScorer):
    def _validate_config(self):
        # Validate model path
        if "model" not in self.config:
            raise ValueError("Error: 'model' must be specified in config.")
        else:
            print(f"Using specified model: '{self.config['model']}'.")

        # Validate max_length
        if "max_length" in self.config and isinstance(self.config["max_length"], int) and 0 < self.config["max_length"]:
            print(f"Using specified max_length: {self.config['max_length']}.")
        elif "max_length" in self.config and isinstance(self.config["max_length"], int) and self.config["max_length"] <= 0:
            print("Warning: the specific max_length should > 0. use default value of 2048.")
            self.config['max_length'] = 2048
        else:
            print("Warning: No specific max_length, use default value of 2048.")
            self.config['max_length'] = 2048

        # Validate batch_size
        if "batch_size" not in self.config or not isinstance(self.config["batch_size"], int) or self.config["batch_size"] <= 0:
            self.config["batch_size"] = 8
            print("Warning: No/invalid batch_size, use default value of 8.")
        else:
            print(f"Using specified batch_size: {self.config['batch_size']}.")

    def _setup(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config['model'])
            self.model = AutoModelForCausalLM.from_pretrained(self.config['model'])
            
            # Critical: Check and set Pad Token
            # Models like Llama and Qwen don't have a pad token by default
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                # Ensure the model's config is also synchronized
                if self.model.config.pad_token_id is None:
                    self.model.config.pad_token_id = self.tokenizer.eos_token_id
            
            self.model.to(self.device)
            self.model.eval()
            print(f"Setting up PPLScorer successfully on {self.device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def score_item(self, data_item: Dict) -> float:
        """Calculate PPL score for a single sample (lower is better)"""
        return self.score_batch([data_item])[0]

    def score_batch(self, data_items: List[Dict]) -> List[float]:
        """
        Calculate PPL scores in batch.
        Critical: Correctly handle padding and attention_mask, ensuring padding tokens in labels are ignored.
        """
        # Extract text content from data items, concatenating instruction, input, and output
        texts = []
        for item in data_items:
            parts = []
            parts.append(item["instruction"])
            parts.append(item.get("input", ""))
            parts.append(item["output"])
            
            # Join with newlines, filtering out empty strings
            text = "\n".join([p for p in parts if p])
            texts.append(text)

        # Batch tokenize with padding and truncation enabled
        encodings = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,        # Critical: Enable padding
            truncation=True,     # Critical: Enable truncation
            max_length=self.config["max_length"]
        ).to(self.device)

        input_ids = encodings.input_ids
        attention_mask = encodings.attention_mask

        # Check for truncation and issue warning if any text was truncated
        max_length = self.config["max_length"]
        for i, text in enumerate(texts):
            # Check if the sequence length reaches max_length (indicating potential truncation)
            seq_length = attention_mask[i].sum().item()
            if seq_length == max_length:
                print(f"Warning: Certain data may have been truncated to max_length={max_length}.")

        # Critical: Create labels and ignore padding tokens
        # Set labels as a copy of input_ids
        labels = input_ids.clone()
        
        # Replace padding token positions in labels with -100
        # PyTorch CrossEntropyLoss has a default ignore_index of -100
        # This ensures padding tokens don't participate in loss calculation
        labels[labels == self.tokenizer.pad_token_id] = -100

        # Calculate PPL for each sample
        perplexities = []
        with torch.no_grad():
            # For batches, we need to calculate PPL for each sample individually
            for i in range(len(texts)):
                # Get input_ids, attention_mask, labels for a single sample
                sample_input_ids = input_ids[i:i+1]
                sample_attention_mask = attention_mask[i:i+1]
                sample_labels = labels[i:i+1]
                
                outputs = self.model(
                    input_ids=sample_input_ids,
                    attention_mask=sample_attention_mask,
                    labels=sample_labels  # labels contain -100 to ignore padding
                )
                
                # outputs.loss is the average cross-entropy loss for all "non-(-100)" tokens in this sample
                loss = outputs.loss
                
                # PPL is e^(loss)
                perplexity = torch.exp(loss).item()
                perplexities.append(perplexity)

        return perplexities

    def evaluate(self, dataset: str) -> List[Dict]:
        """Evaluate the entire dataset and return PPL scores for each sample"""
        num_lines = get_total_lines(dataset)
        results: List[Dict] = []

        batch_size = self.config.get("batch_size")
        buffer_items = []
        buffer_ids = []

        with open(dataset, 'r', encoding='utf-8') as f:
            pbar = tqdm(total=num_lines, desc=self.config.get('name', 'PPLScorer'))
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
                pbar.update(batch_size)

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