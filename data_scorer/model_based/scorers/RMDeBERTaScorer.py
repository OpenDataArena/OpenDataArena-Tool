import torch
from .base_scorer import BaseScorer
import json
from typing import Dict, List
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
from tqdm import tqdm
from .utils import get_total_lines


class RMDeBERTaScorer(BaseScorer):
    def _validate_config(self):
        if "model" not in self.config:
            print(
                "Warning: No local model specified in config. Downloading the remote huggingface model.")
            self.config['model'] = 'OpenAssistant/reward-model-deberta-v3-large-v2'
        else:
            print(f"Using specified local model: '{self.config['model']}'. ")

        if "max_length" in self.config and isinstance(self.config["max_length"], int) and 0 < self.config["max_length"]:
            print(f"Using specified max_length: {self.config['max_length']}.")
        else:
            print("Warning: No specific max_length, use default value of 512.")
            self.config['max_length'] = 512

        if "batch_size" not in self.config or not isinstance(self.config["batch_size"], int) or self.config["batch_size"] <= 0:
            self.config["batch_size"] = 32
            print("Warning: No/invalid batch_size, use default value of 32.")
        else:
            print(f"Using specified batch_size: {self.config['batch_size']}.")

    def _setup(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config['model'])
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config['model'])
        except Exception as e:
            print(
                f"Load specified model failed ({e}), fall back to OpenAssistant/reward-model-deberta-v3-large-v2")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                'OpenAssistant/reward-model-deberta-v3-large-v2')
            self.tokenizer = AutoTokenizer.from_pretrained(
                'OpenAssistant/reward-model-deberta-v3-large-v2')

        self.model.to(self.device)
        self.model.eval()
        print("Setting up RMDeBERTaScorer successfully")

    def score_item(self, data_item):
        return self.score_batch([data_item])[0]

    def score_batch(self, data_items: List[Dict]) -> List[float]:
        """
        Score a batch of data items using the reward model.
        """
        # Prepare inputs: pair of (question, answer)
        questions = []
        answers = []
        
        for item in data_items:

            instruction=item["instruction"]
            input_text = item.get("input", "")
            
            # Concatenate instruction and input with newline if input exists
            if input_text:
                question = f"{instruction}\n{input_text}"
            else:
                question = instruction
            
            # answer = item.get("output", "")
            answer = item["output"]
            questions.append(question)
            answers.append(answer)

        # Tokenize all pairs in batch
        inputs = self.tokenizer(
            questions,
            answers,
            padding=True,
            truncation=True,
            max_length=self.config["max_length"],
            return_tensors="pt"
        ).to(self.device)

        # Check for truncation
        max_length_config = self.config["max_length"]
        truncated_indices = []
        for idx, input_ids in enumerate(inputs["input_ids"]):
            # If the actual length equals max_length, it may have been truncated
            actual_length = (input_ids != self.tokenizer.pad_token_id).sum().item()
            if actual_length >= max_length_config:
                truncated_indices.append(idx)
        
        if truncated_indices:
            item_ids = [data_items[i].get("id", f"index_{i}") for i in truncated_indices]
            print(f"Warning: {len(truncated_indices)} item(s) exceeded max_length ({max_length_config}) and were truncated. "
                  f"Item IDs: {item_ids[:5]}{'...' if len(item_ids) > 5 else ''}")

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Get logits and convert to scores
            scores = outputs.logits[:, 0].cpu().detach().tolist()

        return scores

    def evaluate(self, dataset) -> List[Dict]:
        num_lines = get_total_lines(dataset)
        results: List[Dict] = []

        batch_size = self.config.get("batch_size")
        buffer_items = []
        buffer_ids = []

        with open(dataset, 'r') as f:
            pbar = tqdm(total=num_lines, desc=self.config.get(
                'name', 'RMDeBERTaScorer'))
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
