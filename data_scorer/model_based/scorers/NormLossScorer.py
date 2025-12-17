import torch
from .base_scorer import BaseScorer
import json
import math
import os
from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from .utils import get_total_lines
import warnings


class NormLossScorer(BaseScorer):
    def _validate_config(self):
        if "model" not in self.config:
            print(
                "Warning: No local model specified in config. Downloading the remote huggingface model.")
            self.config['model'] = 'meta-llama/Llama-3.1-8B'
        else:
            print(f"Using specified local model: '{self.config['model']}'. ")

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
                self.config['model'])
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config['model'])
        except Exception as e:
            print(
                f"Load specified model failed ({e}), fall back to meta-llama/Llama-3.1-8B")
            self.model = AutoModelForCausalLM.from_pretrained(
                'meta-llama/Llama-3.1-8B')
            self.tokenizer = AutoTokenizer.from_pretrained(
                'meta-llama/Llama-3.1-8B')

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.to(self.device)
        self.model.eval()
        print("Setting up NormLossScorer successfully")

    def score_item(self, data_item):
        return self.score_batch([data_item])[0]

    def score_batch(self, data_items: List[Dict]) -> List[float]:
        texts = []
        for item in data_items:
            instruction=item["instruction"]
            input_text=item.get("input_text", "")
            response = item["output"]
            if input_text:
                input_text = instruction + '\n' + input_text + '\n' + response
            else:
                input_text = instruction + '\n' + response
            texts.append(input_text)

        # Batch tokenize
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.config["max_length"]
        ).to(self.device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Check for truncated sequences and issue warning
        max_seq_length = attention_mask.sum(dim=1).max().item()
        if max_seq_length >= self.config["max_length"]:
            truncated_count = (attention_mask.sum(dim=1) >= self.config["max_length"]).sum().item()
            warnings.warn(
                f"Warning: {truncated_count} out of {len(data_items)} sequences were truncated "
                f"to max_length={self.config['max_length']}. Consider increasing max_length for complete processing.",
                UserWarning
            )

        # Create labels, set padding positions to -100
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        scores = []
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Get loss for each token (using natural logarithm)
            # Use ignore_index=-100 to ignore padding tokens
            loss_per_token = torch.nn.functional.cross_entropy(
                outputs.logits[:, :-1, :].contiguous().view(-1, outputs.logits.size(-1)),
                labels[:, 1:].contiguous().view(-1),
                reduction='none',
                ignore_index=-100
            )

            # Reshape to [batch_size, seq_len-1]
            loss_per_token = loss_per_token.view(input_ids.size(0), -1)
            # Adjust attention_mask to match loss_per_token shape (remove first token)
            valid_mask = attention_mask[:, 1:].float()

            # Calculate normalized loss for each sample
            for i in range(input_ids.size(0)):
                # Use valid_mask to calculate loss for valid tokens
                # Note: padding positions in loss_per_token are already handled by ignore_index, returning 0
                masked_loss = loss_per_token[i] * valid_mask[i]
                total_loss = masked_loss.sum().item()
                total_tokens = valid_mask[i].sum().item()

                # Normalize and convert from natural logarithm to log2
                if total_tokens > 0:
                    normalized_loss = (total_loss / total_tokens) / math.log(2)
                else:
                    normalized_loss = 0.0
                
                scores.append(normalized_loss)

        return scores

    def evaluate(self, dataset) -> List[Dict]:
        num_lines = get_total_lines(dataset)
        results: List[Dict] = []

        batch_size = self.config.get("batch_size")
        buffer_items = []
        buffer_ids = []

        with open(dataset, 'r') as f:
            pbar = tqdm(total=num_lines, desc=self.config.get(
                'name', 'NormLossScorer'))
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
                    pbar.update(len(buffer_items))
                    buffer_items.clear()
                    buffer_ids.clear()

            if buffer_items:
                batch_scores = self.score_batch(buffer_items)
                results.extend([
                    {"id": id_, "score": sc}
                    for id_, sc in zip(buffer_ids, batch_scores)
                ])
                pbar.update(len(buffer_items))
                buffer_items.clear()
                buffer_ids.clear()
            pbar.close()

        return results

