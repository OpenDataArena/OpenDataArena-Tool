import torch
import torch.nn as nn
from .base_scorer import BaseScorer
import json
from typing import Dict, List
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import PyTorchModelHubMixin
import os
from tqdm import tqdm
from .utils import get_total_lines


class QualityModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config):
        super(QualityModel, self).__init__()
        self.model = AutoModel.from_pretrained(config["base_model"])
        self.dropout = nn.Dropout(config["fc_dropout"])
        self.fc = nn.Linear(self.model.config.hidden_size,
                            len(config["id2label"]))

    def forward(self, input_ids, attention_mask):
        features = self.model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        dropped = self.dropout(features)
        outputs = self.fc(dropped)
        return torch.softmax(outputs[:, 0, :], dim=1)


class DebertaScorer(BaseScorer):
    def _validate_config(self):
        # Check if the config specifies a local model path. If not, use the default remote model and notify the user.
        if "model" not in self.config:
            print(
                "Warning: No local model specified in config. Downloading the remote huggingface model.")
            self.config['model'] = 'nvidia/quality-classifier-deberta'


        if "max_length" in self.config and isinstance(self.config["max_length"], int) and 0 < self.config["max_length"] <= 2048:
            print(f"Using specified max_length: {self.config['max_length']}.")
        elif "max_length" in self.config and isinstance(self.config["max_length"], int) and self.config["max_length"] <= 0:
            print(
                "Warning: the specified max_length should be > 0. Using default value of 2048.")
            self.config['max_length'] = 2048
        else:
            print("Warning: No max_length specified, using default value of 2048.")
            self.config['max_length'] = 2048

        if "batch_size" not in self.config or not isinstance(self.config["batch_size"], int) or self.config["batch_size"] <= 0:
            self.config["batch_size"] = 32
            print("Warning: No/invalid batch_size specified, using default value of 32.")
        else:
            print(f"Using specified batch_size: {self.config['batch_size']}.")

    def _setup(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config['model'])
            self.model = QualityModel.from_pretrained(
                self.config['model']).to(self.device)
        except Exception as e:
            print(
                f"Failed to load specified model ({e}), falling back to nvidia/quality-classifier-deberta")
            self.tokenizer = AutoTokenizer.from_pretrained(
                'nvidia/quality-classifier-deberta')
            self.model = QualityModel.from_pretrained(
                'nvidia/quality-classifier-deberta').to(self.device)

        self.model.eval()
        print("DebertaScorer setup successfully")

    def score_item(self, data_item):
        return self.score_batch([data_item])[0]

    def score_batch(self, data_items: List[Dict]) -> List[int]:
        texts = []
        for item in data_items:
            text = item["instruction"]
            input_text = item.get("input", "")
            if input_text:  # Only add input if it has actual content
                text += '\n' + input_text
            text += '\n' + item["output"]
            texts.append(text)

        # Batch tokenize
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config["max_length"]
        ).to(self.device)

        # Check if any text was truncated
        truncated_count = 0
        for i, text in enumerate(texts):
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            if len(tokens) > self.config["max_length"]:
                truncated_count += 1
        
        if truncated_count > 0:
            print(f"Warning: {truncated_count} out of {len(texts)} texts were truncated due to exceeding max_length ({self.config['max_length']})")

        with torch.no_grad():
            outputs = self.model(inputs["input_ids"], inputs["attention_mask"])
            predicted_classes = torch.argmax(outputs, dim=1)
            scores = predicted_classes.cpu().tolist()

            # Ensure a list is returned even for a single sample
            if not isinstance(scores, list):
                scores = [scores]

        return scores

    def evaluate(self, dataset) -> List[Dict]:
        num_lines = get_total_lines(dataset)
        results: List[Dict] = []

        batch_size = self.config.get("batch_size")
        buffer_items = []
        buffer_ids = []

        with open(dataset, 'r') as f:
            pbar = tqdm(total=num_lines, desc=self.config.get(
                'name', 'DebertaScorer'))
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
