import torch
from .base_scorer import BaseScorer
import json
from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from .utils import get_total_lines
from tqdm import tqdm


class GraNdScorer(BaseScorer):
    def _validate_config(self):
        if "model" not in self.config:
            print(
                "Warning: No local model specified in config. Downloading the remote huggingface model.")
            self.config['model'] = 'Qwen/Qwen3-8B'
        else:
            if not os.path.exists(self.config["model"]) and '/' not in self.config["model"]:
                print(
                    f"Warning: Specified local model path '{self.config['model']}' does not exist. "
                    "Using as huggingface model name."
                )
            elif os.path.exists(self.config["model"]):
                print(
                    f"Using specified local model: '{self.config['model']}'. ")
            else:
                print(f"Using specified huggingface model: '{self.config['model']}'.")

        if "max_length" not in self.config:
            print("Warning: No max_length specified, use default value of 2048.")
            self.config["max_length"] = 2048
        else:
            print(f"Using specified max_length: {self.config['max_length']}.")
        
        self.max_length = self.config["max_length"]

    def _setup(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config['model'])
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config['model'])
        except Exception as e:
            print(
                f"Warning: Specified Model Path Does not Work ({e}), Use Default Model Instead.")
            self.model = AutoModelForCausalLM.from_pretrained('gpt2')
            self.tokenizer = AutoTokenizer.from_pretrained('gpt2')

        # Ensure tokenizer has pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.to(self.device)
        self.model.eval()  # Set to eval mode by default
        print("Setting up GraNdScorer successfully")

    def score_item(self, data_item: Dict) -> float:
        """Calculate gradient norm for a single data point"""
        instr = data_item["instruction"].strip()
        input = data_item.get("input", "").strip()
        if input:
            instr = instr + "\n" + input
        outp = data_item["output"].strip()
        
        # Build full text: instruction + output
        text = instr + "\n" + outp
        
        # Encode instruction only (to get its length)
        instr_encodings = self.tokenizer(
            instr,
            padding=False,
            truncation=True,
            max_length=self.config["max_length"],
            return_tensors="pt"
        ).to(self.device)
        
        # Encode full text (instruction + output)
        full_encodings = self.tokenizer(
            text,
            padding=False,
            truncation=True,
            max_length=self.config["max_length"],
            return_tensors="pt"
        ).to(self.device)

        input_ids = full_encodings["input_ids"]
        
        # Set labels: ignore instruction part, only compute loss on output part
        labels = input_ids.clone()
        instr_length = instr_encodings["input_ids"].shape[1]
        labels[:, :instr_length] = -100  # -100 is ignored in loss computation

        # Set model to train mode to compute gradients
        self.model.train()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Forward pass
        outputs = self.model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        
        # Backward pass to compute gradients
        loss.backward()
        
        # Calculate L2 norm of all parameter gradients
        total_norm = 0.0
        param_count = 0
        for param in self.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        gradient_norm = total_norm ** (1. / 2)
        
        # Zero gradients and set back to eval mode
        self.model.zero_grad()
        self.model.eval()
        
        return float(gradient_norm)

    def evaluate(self, dataset: str) -> List[Dict]:
        """Evaluate the entire dataset, calculating gradient norm for each data point"""
        num_lines = get_total_lines(dataset)
        results: List[Dict] = []

        with open(dataset, 'r', encoding='utf-8') as f:
            pbar = tqdm(total=num_lines, desc=self.config.get(
                'name', 'GraNdscorer'))
            for line in f:
                item = json.loads(line.strip())
                item_id = item.get("id", "")
                
                # Calculate gradient norm
                gradient_norm = self.score_item(item)
                
                results.append({
                    "id": item_id,
                    "score": gradient_norm
                })
                pbar.update(1)
            pbar.close()

        return results

