import torch
from .base_scorer import BaseScorer
import json
from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
from .utils import get_total_lines
from tqdm import tqdm


class SkyworkRewardScorer(BaseScorer):
    def _validate_config(self):
        if "model" not in self.config:
            print(
                "Warning: No local model specified in config. Downloading the remote huggingface model.")
            self.config['model'] = 'Skywork/Skywork-Reward-V2-Llama-3.1-8B-40M'
        else:
            if self.config['model'] == 'Skywork/Skywork-Reward-V2-Llama-3.1-8B-40M':
                print("Downloading and use the specific remote huggingface model.")
            elif not os.path.exists(self.config["model"]):
                print(
                    f"Warning: Specified local model path '{self.config['model']}' does not exist. "
                    "Downloading the remote huggingface model: Skywork/Skywork-Reward-V2-Llama-3.1-8B-40M"
                )
                self.config['model'] = 'Skywork/Skywork-Reward-V2-Llama-3.1-8B-40M'
            else:
                print(
                    f"Using specified local model: '{self.config['model']}'. ")

        if "batch_size" not in self.config or not isinstance(self.config["batch_size"], int) or self.config["batch_size"] <= 0:
            self.config["batch_size"] = 32
            print("Warning: No/invalid batch_size, use default value of 32.")
        else:
            print(f"Using specified batch_size: {self.config['batch_size']}.")

        if "max_length" not in self.config or not isinstance(self.config["max_length"], int) or self.config["max_length"] <= 0:
            self.config["max_length"] = 4096
            print("Warning: No/invalid max_length, use default value of 4096.")
        else:
            print(f"Using specified max_length: {self.config['max_length']}.")

    def _setup(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.rank_model = AutoModelForSequenceClassification.from_pretrained(
                self.config['model'],
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
                num_labels=1
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config['model'])
        except Exception as e:
            print(
                f"Warning: Specified model path does not work ({e}), use remote model instead.")
            self.rank_model = AutoModelForSequenceClassification.from_pretrained(
                "Skywork/Skywork-Reward-V2-Llama-3.1-8B-40M",
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
                num_labels=1
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                "Skywork/Skywork-Reward-V2-Llama-3.1-8B-40M")

        self.rank_model.to(self.device)
        self.rank_model.eval()
        print("Setting up SkyworkRewardScorer successfully")

    def score_item(self, data_item):
        return self.score_batch([data_item])[0]

    def score_batch(self, data_items: List[Dict]) -> List[float]:
        """
        Score a batch of data items using the reward model.
        """
        conversations = []
        for item in data_items:
            prompt = item.get("instruction", "")
            output = item.get("output", "")
            conv = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": output}
            ]
            conversations.append(conv)

        # Tokenize all conversations with truncation
        input_ids_list = []
        truncated_indices = []
        max_length_config = self.config["max_length"]
        
        for idx, conv in enumerate(conversations):
            encoded = self.tokenizer.apply_chat_template(
                conv,
                tokenize=True,
                return_tensors="pt",
                add_generation_prompt=False,
                truncation=True,
                max_length=max_length_config
            )
            
            # Check if truncation occurred by comparing actual length with max_length
            actual_length = encoded.shape[1]
            if actual_length >= max_length_config:
                truncated_indices.append(idx)
            
            input_ids_list.append(encoded[0])

        # Warn if any items were truncated
        if truncated_indices:
            item_ids = [data_items[i].get("id", f"index_{i}") for i in truncated_indices]
            print(f"Warning: {len(truncated_indices)} item(s) exceeded max_length ({max_length_config}) and were truncated. "
                  f"Item IDs: {item_ids[:5]}{'...' if len(item_ids) > 5 else ''}")

        # Pad sequences in batch
        batch = self.tokenizer.pad(
            {"input_ids": input_ids_list},
            padding="longest",
            return_tensors="pt",
            max_length=max_length_config
        )
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.rank_model(
                input_ids=input_ids, attention_mask=attention_mask).logits  # [B, 1]
            scores = logits.squeeze(-1).float().tolist()

        return [float(s) for s in scores]

    def evaluate(self, dataset) -> List[Dict]:
        num_lines = get_total_lines(dataset)
        results: List[Dict] = []

        batch_size = self.config["batch_size"]
        buf_items, buf_ids = [], []

        with open(dataset, 'r') as f:
            pbar = tqdm(total=num_lines, desc=self.config.get(
                'name', 'SkyworkRewardScorer'))
            for line in f:
                item = json.loads(line.strip())
                buf_items.append(item)
                buf_ids.append(item.get("id", ""))

                if len(buf_items) == batch_size:
                    batch_scores = self.score_batch(buf_items)
                    results.extend(
                        {"id": _id, "score": sc}
                        for _id, sc in zip(buf_ids, batch_scores)
                    )
                    buf_items.clear()
                    buf_ids.clear()
                pbar.update(1)

            if buf_items:
                batch_scores = self.score_batch(buf_items)
                results.extend(
                    {"id": _id, "score": sc}
                    for _id, sc in zip(buf_ids, batch_scores)
                )
                buf_items.clear()
                buf_ids.clear()
            pbar.close()

        return results
