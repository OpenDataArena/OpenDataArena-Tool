import torch
from .base_scorer import BaseScorer
import json
from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from .utils import get_total_lines
from tqdm import tqdm


class DeitaQScorer(BaseScorer):
    def _validate_config(self):
        if "model" not in self.config:
            print(
                "Warning: No loacl model specified in config. Downloading the remote huggingface model.")
            self.config['model'] = 'hkust-nlp/deita-quality-scorer'
        else:
            if self.config['model'] == 'hkust-nlp/deita-quality-scorer':
                print("Downloading and use the specific remote huggingface model.")
            elif not os.path.exists(self.config["model"]):
                print(
                    f"Warning: Specified local model path '{self.config['model']}' does not exist. "
                    "Downloading the remote huggingface model: hkust-nlp/deita-quality-scorer"
                )
                self.config['model'] = 'hkust-nlp/deita-quality-scorer'
            else:
                print(
                    f"Using specified local model: '{self.config['model']}'. ")

        if ("max_length" in self.config and isinstance(self.config["max_length"], int)
                and 0 < self.config["max_length"] <= 2048):
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
                f"Warnning: Specified Model Path Does not Work ({e}), Use Remote Model Instead.")
            self.model = AutoModelForCausalLM.from_pretrained(
                'hkust-nlp/deita-quality-scorer')
            self.tokenizer = AutoTokenizer.from_pretrained(
                'hkust-nlp/deita-quality-scorer')

        self.model.to(self.device)
        self.model.eval()
        print("Setting up DeitaQScorer successfully")

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

        score = self.score_batch([data_item])[0]
        return score

    def score_batch(self, data_items: List[Dict]) -> List[float]:
        quality_template = (
            "You are a helpful assistant. Please identify the quality score of the Response corresponding to the Question. \n"
            "#Question#:\n{instruction}\n#Response#:\n{output} \n##Quality: "
        )

        valid_indices = []
        prompts = []
        for idx, item in enumerate(data_items):
            instr = item.get("instruction", "").strip()
            outp = item.get("output", "").strip()
            if not instr:
                print("Warning: 'instruction' is missing or empty. Skipping this item.")
                prompts.append(None)
                continue
            if not outp:
                print("Warning: 'output' is missing or empty. Skipping this item.")
                prompts.append(None)
                continue
            prompts.append(quality_template.format(
                instruction=instr, output=outp))
            valid_indices.append(idx)

        scores = [None] * len(data_items)
        if len(valid_indices) == 0:
            return scores

        enc = self.tokenizer(
            [prompts[i] for i in valid_indices],
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

            first_token_scores = outputs.scores[0]  # [B, V]

            try:
                selected_logits = first_token_scores.index_select(
                    1, self._score_ids)  # [B, 6]
            except Exception:

                for i in valid_indices:
                    scores[i] = 3.0
                return scores

            probs = torch.softmax(selected_logits, dim=-1)         # [B, 6]
            batch_scores = (probs * self._score_values).sum(-1)    # [B]

        for j, idx in enumerate(valid_indices):
            scores[idx] = batch_scores[j].item()

        return scores

    def evaluate(self, dataset) -> List[Dict]:
        num_lines = get_total_lines(dataset)
        results: List[Dict] = []

        batch_size = self.config["batch_size"]
        buf_items, buf_ids = [], []

        with open(dataset, 'r') as f:
            pbar = tqdm(total=num_lines, desc=self.config.get(
                'name', 'DeitaQScorer'))
            for line in f:
                item = json.loads(line.strip())
                buf_items.append(item)
                buf_ids.append(item.get("id", ""))

                if len(buf_items) == batch_size:
                    batch_scores = self.score_batch(buf_items)
                    results.extend(
                        {"id": _id, "Deita_Quality": sc}
                        for _id, sc in zip(buf_ids, batch_scores)
                    )
                    buf_items.clear()
                    buf_ids.clear()
                pbar.update(1)

            if buf_items:
                batch_scores = self.score_batch(buf_items)
                results.extend(
                    {"id": _id, "Deita_Quality": sc}
                    for _id, sc in zip(buf_ids, batch_scores)
                )
                buf_items.clear()
                buf_ids.clear()
            pbar.close()

        return results
