import torch
from .base_scorer import BaseScorer
import json
from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from tqdm import tqdm
from .utils import get_total_lines


class DeitaCScorer(BaseScorer):
    def _validate_config(self):
        if "model" not in self.config:
            print(
                "Warning: No local model specified in config. Downloading the remote huggingface model.")
            self.config['model'] = 'hkust-nlp/deita-complexity-scorer'

        if "max_length" in self.config and isinstance(self.config["max_length"], int) and 0 < self.config["max_length"] <= 2048:
            print(f"Using specified max_length: {self.config['max_length']}.")
        elif "max_length" in self.config and isinstance(self.config["max_length"], int) and self.config["max_length"] <= 0:
            print(
                "Warning: the specific max_length should > 0. use default value of 2048.")
            self.config['max_length'] = 2048
        elif "max_length" in self.config and isinstance(self.config["max_length"], int) and self.config["max_length"] > 2048:
            print(
                "Warning: the specific max_length should not be larger than 2048. use default value of 2048.")
            self.config['max_length'] = 2048
        else:
            print("Warning: No specific max_length, use default value of 2048.")
            self.config['max_length'] = 2048

        if "batch_size" not in self.config or not isinstance(self.config["batch_size"], int) or self.config["batch_size"] <= 0:
            self.config["batch_size"] = 32
            print("Warning: No/invalid batch_size, use default value of 32.")
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
                f"Load specified model failed ({e}), fall back to hkust-nlp/deita-complexity-scorer")
            self.model = AutoModelForCausalLM.from_pretrained(
                'hkust-nlp/deita-complexity-scorer')
            self.tokenizer = AutoTokenizer.from_pretrained(
                'hkust-nlp/deita-complexity-scorer')

        self.model.to(self.device)
        self.model.eval()
        print("Setting up DeitaCScorer successfully")

        self.id2score = {
            29896: "1",
            29906: "2",
            29941: "3",
            29946: "4",
            29945: "5",
            29953: "6"
        }

        self._score_ids = torch.tensor(
            list(self.id2score.keys()), device=self.device, dtype=torch.long)
        self._score_values = torch.tensor(
            [1, 2, 3, 4, 5, 6], device=self.device, dtype=torch.float)

    def score_item(self, data_item):

        return self.score_batch([data_item])[0]

    def score_batch(self, data_items: List[Dict]) -> List[float]:

        complexity_template = (
            "You are a helpful assistant. Please identify the complexity score of the following user query. \n"
            "##Query: {instruction}  \n##Complexity: ")
        user_inputs = []
        for item in data_items:
            instruction = item.get("instruction", "")
            input_text = item.get("input", "")
            if input_text:
                instruction = instruction + "\n" + input_text
            user_inputs.append(complexity_template.format(instruction=instruction))


        # Check for sequences exceeding max_length
        tokenized_no_trunc = self.tokenizer(
            user_inputs,
            padding=False,
            truncation=False,
            return_tensors=None
        )
        max_length = self.config["max_length"]
        for idx, input_ids in enumerate(tokenized_no_trunc["input_ids"]):
            if len(input_ids) > max_length:
                print(f"Warning: Data item at index {idx} has length {len(input_ids)} which exceeds max_length {max_length}. It will be truncated.")

        enc = self.tokenizer(
            user_inputs,
            padding=True,
            truncation=True,
            max_length=self.config["max_length"],
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():

            outputs = self.model.generate(
                **enc,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True
            )

            first_token_scores: torch.Tensor = outputs.scores[0]

            try:
                selected_logits = first_token_scores.index_select(
                    1, self._score_ids)  # [batch, 6]
            except Exception:

                return [3.0] * len(data_items)

            probs = torch.softmax(selected_logits, dim=-1)                # [batch, 6]
            scores = (probs * self._score_values).sum(dim=-1).tolist()    # [batch]

        return scores

    def evaluate(self, dataset) -> List[Dict]:
        num_lines = get_total_lines(dataset)
        results: List[Dict] = []

        batch_size = self.config.get("batch_size")
        buffer_items = []
        buffer_ids = []

        with open(dataset, 'r') as f:
            pbar = tqdm(total=num_lines, desc=self.config.get(
                'name', 'DeitaCScorer'))
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
