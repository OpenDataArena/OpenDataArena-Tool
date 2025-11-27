import re
import os
import json
import fasttext
from huggingface_hub import hf_hub_download
from .base_scorer import BaseScorer
from typing import Dict, List
from tqdm import tqdm
from .utils import get_total_lines


class TextbookScorer(BaseScorer):
    score_dict = {
        '__label__': 0,
        '__label__Low': 0,
        '__label__Mid': 1,
        '__label__High': 2
    }

    def replace_newlines(self, text: str) -> str:
        return re.sub("\n+", " ", text)

    def _validate_config(self):
        if "model" not in self.config:
            print(
                "Warning: No local model specified in config. Downloading the remote huggingface model.")
            self.config['model'] = 'kenhktsui/llm-data-textbook-quality-fasttext-classifer-v2'
        else:
            print(f"Using specified model: '{self.config['model']}'.")

        if "batch_size" not in self.config or not isinstance(self.config["batch_size"], int) or self.config["batch_size"] <= 0:
            self.config["batch_size"] = 32
            print("Warning: No/invalid batch_size, use default value of 32.")
        else:
            print(f"Using specified batch_size: {self.config['batch_size']}.")

    def _setup(self):
        try:
            if self.config['model'] == 'kenhktsui/llm-data-textbook-quality-fasttext-classifer-v2':
                self.model = fasttext.load_model(hf_hub_download(
                    self.config['model'], "model.bin"))
            else:
                path = f"{str(self.config['model'])}/model.bin"
                self.model = fasttext.load_model(path)
        except Exception as e:
            print(
                f"Load specified model failed ({e}), fall back to kenhktsui/llm-data-textbook-quality-fasttext-classifer-v2")
            self.model = fasttext.load_model(hf_hub_download(
                'kenhktsui/llm-data-textbook-quality-fasttext-classifer-v2', "model.bin"))
        print("Setting up TextbookScorer successfully")

    def score_item(self, data_item):
        return self.score_batch([data_item])[0]

    def score_batch(self, data_items: List[Dict]) -> List[float]:
        texts = []
        for item in data_items:
            instruction = item["instruction"]
            response = item["output"]
            if "input" in item:
                input_text = item["input"]
                text = instruction + '\n' + input_text + '\n' + response
            else:
                text = instruction + '\n' + response
            text = self.replace_newlines(text)
            texts.append(text)

        labels_batch, probs_batch = self.model.predict(texts, k=-1)
        
        scores = []
        for labels, probs in zip(labels_batch, probs_batch):
            score = 0.0
            for label, prob in zip(labels, probs):
                score += self.__class__.score_dict.get(label, 0) * prob
            scores.append(float(score))

        return scores

    def evaluate(self, dataset) -> List[Dict]:
        num_lines = get_total_lines(dataset)
        results: List[Dict] = []

        batch_size = self.config.get("batch_size")
        buffer_items = []
        buffer_ids = []

        with open(dataset, 'r') as f:
            pbar = tqdm(total=num_lines, desc=self.config.get(
                'name', 'TextbookScorer'))
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
