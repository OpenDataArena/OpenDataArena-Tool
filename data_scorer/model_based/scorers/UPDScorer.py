import torch
import torch.nn.functional as F
from .base_scorer import BaseScorer
import json
from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from tqdm import tqdm
from .utils import get_total_lines
import numpy as np
import warnings


class UPDScorer(BaseScorer):
    def _validate_config(self):
        """Validate configuration parameters"""
        if "model" not in self.config:
            raise ValueError(
                "model is required in config. Please specify a causal language model path.")
        else:
            if not os.path.exists(self.config["model"]):
                # Possibly a Hugging Face model name
                print(
                    f"Model '{self.config['model']}' will be loaded from Hugging Face Hub.")
            else:
                print(
                    f"Using specified local model: '{self.config['model']}'.")

        if "max_length" in self.config and isinstance(self.config["max_length"], int) and 0 < self.config["max_length"]:
            print(f"Using specified max_length: {self.config['max_length']}.")
        elif "max_length" in self.config and isinstance(self.config["max_length"], int) and self.config["max_length"] <= 0:
            print(
                "Warning: the specific max_length should > 0. use default value of 2048.")
            self.config['max_length'] = 2048
        else:
            print("Warning: No specific max_length, use default value of 2048.")
            self.config['max_length'] = 2048

        if "batch_size" not in self.config or not isinstance(self.config["batch_size"], int) or self.config["batch_size"] <= 0:
            self.config["batch_size"] = 8  # UPD calculation is complex, use smaller default batch_size
            print("Warning: No/invalid batch_size, use default value of 8.")
        else:
            print(f"Using specified batch_size: {self.config['batch_size']}.")

    def _setup(self):
        """Load model and tokenizer"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config['model'],
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config['model'])
            
            # Set pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model '{self.config['model']}': {e}")

        if self.device == "cuda" and not hasattr(self.model, 'device_map'):
            self.model.to(self.device)
        
        self.model.eval()
        
        # Get vocabulary size
        self.vocab_size = self.model.config.vocab_size
        self.log_vocab_size = np.log(self.vocab_size)
        
        print(f"Setting up UPDScorer successfully")
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Device: {self.device}")

    def _compute_token_upd(self, loss: float, entropy: float) -> float:
        """Compute UPD value for a single token
        
        Args:
            loss: Cross-entropy loss Lt = -Log(P(yt|x, y<t))
            entropy: Shannon entropy Ht
            
        Returns:
            UPD value
        """
        # Sigmoid transformation
        sigmoid_loss = torch.sigmoid(torch.tensor(loss)).item()
        
        # Compute max(0, 1 - Ht/log(v_size))
        entropy_term = max(0.0, 1.0 - entropy / self.log_vocab_size)
        
        # UPDt = σ(Lt) × max(0, 1 - Ht/log(v_size))
        upd = sigmoid_loss * entropy_term
        
        return upd

    def score_item(self, data_item: Dict) -> float:
        """Score a single sample"""
        return self.score_batch([data_item])[0]

    def score_batch(self, data_items: List[Dict]) -> List[float]:
        """Compute UPD scores for a batch in parallel"""
        if len(data_items) == 0:
            return []
        
        # Prepare batch data
        instructions = []
        output_texts = []
        full_texts = []
        
        for item in data_items:
            instruction = item["instruction"]
            input_text = item.get("input", "")
            output = item["output"]
            
            if input_text:
                instruction_full = instruction + "\n" + input_text + "\n"
            else:
                instruction_full = instruction + "\n"
            
            instructions.append(instruction_full)
            output_texts.append(output)
            full_texts.append(instruction_full + output)
        
        # Batch tokenize instructions (to get instruction length for each sample)
        instruction_tokens = self.tokenizer(
            instructions,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True
        )
        instruction_lengths = (instruction_tokens.attention_mask.sum(dim=1)).tolist()
        
        # Batch tokenize full texts without truncation first to detect truncation
        full_tokens_no_trunc = self.tokenizer(
            full_texts,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
            truncation=False
        )
        
        # Check for truncation
        truncated_indices = []
        for idx, tokens in enumerate(full_tokens_no_trunc.input_ids):
            actual_length = (tokens != self.tokenizer.pad_token_id).sum().item()
            if actual_length > self.config["max_length"]:
                truncated_indices.append(idx)
        
        if truncated_indices:
            warnings.warn(
                f"Warning: {len(truncated_indices)} out of {len(data_items)} samples exceed max_length={self.config['max_length']} and will be truncated. "
                f"Sample indices: {truncated_indices[:10]}{'...' if len(truncated_indices) > 10 else ''}",
                UserWarning
            )
        
        # Batch tokenize full texts with truncation
        full_tokens = self.tokenizer(
            full_texts,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=self.config["max_length"]
        )
        
        input_ids = full_tokens.input_ids.to(self.device)
        attention_mask = full_tokens.attention_mask.to(self.device)
        
        batch_size = input_ids.shape[0]
        seq_length = input_ids.shape[1]
        
        # Batch forward pass
        with torch.no_grad():
            model_outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            logits = model_outputs.logits  # shape: (batch_size, seq_len, vocab_size)
        
        # Compute UPD scores for each sample
        batch_scores = []
        
        for batch_idx in range(batch_size):
            instruction_length = instruction_lengths[batch_idx]
            
            # Find the actual length of current sample (non-padding part)
            sample_length = attention_mask[batch_idx].sum().item()
            
            # If instruction is too long or no output, set score to 0
            if instruction_length >= self.config["max_length"] or instruction_length >= sample_length:
                batch_scores.append(0.0)
                continue
            
            # Compute UPD for each token in the output part
            upd_scores = []
            
            # Iterate over each token in the output part (starting from instruction_length)
            for i in range(instruction_length, sample_length):
                # Check if it's padding (double insurance)
                if attention_mask[batch_idx, i].item() == 0:
                    break
                
                true_token_id = input_ids[batch_idx, i].item()
                
                # Skip padding token
                if true_token_id == self.tokenizer.pad_token_id:
                    continue
                
                # Get logits for predicting this token (output at position i-1 predicts token at position i)
                current_logits = logits[batch_idx, i - 1, :]
                
                # Compute probability distribution
                probs = F.softmax(current_logits, dim=-1)
                
                # 1. Compute cross-entropy loss Lt = -log(P(yt|x, y<t))
                token_prob = probs[true_token_id].item()
                token_loss = -np.log(token_prob + 1e-10)
                
                # 2. Compute Shannon entropy Ht = -Σ P(y) * log(P(y))
                log_probs = F.log_softmax(current_logits, dim=-1)
                entropy = -(probs * log_probs).sum().item()
                
                # 3. Compute UPD
                upd = self._compute_token_upd(token_loss, entropy)
                upd_scores.append(upd)
            
            # Compute average UPD
            if upd_scores:
                avg_upd = np.mean(upd_scores)
            else:
                avg_upd = 0.0
            
            batch_scores.append(avg_upd)
        
        return batch_scores

    def evaluate(self, dataset: str) -> List[Dict]:
        """Evaluate the entire dataset"""
        num_lines = get_total_lines(dataset)
        results: List[Dict] = []

        batch_size = self.config.get("batch_size")
        buffer_items = []
        buffer_ids = []

        with open(dataset, 'r', encoding='utf-8') as f:
            pbar = tqdm(total=num_lines, desc=self.config.get(
                'name', 'UPDScorer'))
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
