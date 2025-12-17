from typing import List, Optional, Dict, Any

import torch
import torch.nn as nn
import json
import os
from transformers import LlamaPreTrainedModel, LlamaModel, PreTrainedTokenizerFast
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from tqdm import tqdm

from .base_scorer import BaseScorer
from .utils import get_total_lines


class INFORMForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, self.num_labels)
        )
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class InfOrmScorer(BaseScorer):
    def _validate_config(self):
        if "model" not in self.config:
            print(
                "Warning: No local model specified in config. Downloading the remote huggingface model.")
            self.config['model'] = 'infly/INF-ORM-Llama3.1-70B'


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
            self.rank_model = INFORMForSequenceClassification.from_pretrained(
                self.config['model'],
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
                device_map="auto" if torch.cuda.is_available() else None,
                num_labels=1
            )
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
                self.config['model'])
            # Set pad_token (Llama models don't have pad_token by default, use eos_token)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            # Update pad_token_id in model config
            self.rank_model.config.pad_token_id = self.tokenizer.pad_token_id
        except Exception as e:
            print(
                f"Warning: Specified Model Path Does not Work ({e}), Use Remote Model Instead.")
            self.rank_model = INFORMForSequenceClassification.from_pretrained(
                "infly/INF-ORM-Llama3.1-70B",
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
                device_map="auto" if torch.cuda.is_available() else None,
                num_labels=1
            )
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
                "infly/INF-ORM-Llama3.1-70B")
            # Set pad_token (Llama models don't have pad_token by default, use eos_token)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            # Update pad_token_id in model config
            self.rank_model.config.pad_token_id = self.tokenizer.pad_token_id

        if not torch.cuda.is_available() or self.rank_model.device.type != "cuda":
            self.rank_model.to(self.device)
        self.rank_model.eval()
        print("Setting up InfOrmScorer successfully")

    def score_item(self, data_item):
        return self.score_batch([data_item])[0]

    def score_batch(self, data_items: List[Dict]) -> List[float]:

        input_ids_list = []
        truncation_count = 0

        for item in data_items:
            prompt = item["instruction"]
            input_text = item.get("input", "")
            if input_text:
                prompt = prompt + '\n' + input_text
            output = item["output"]
            conv = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": output}
            ]

            # First encode without truncation to check original length
            encoded_full = self.tokenizer.apply_chat_template(
                conv,
                tokenize=True,
                return_tensors="pt",
                add_generation_prompt=False,
                truncation=False
            )
            
            original_length = encoded_full.shape[1]
            if original_length > self.config["max_length"]:
                truncation_count += 1
                # Encode with truncation
                encoded = self.tokenizer.apply_chat_template(
                    conv,
                    tokenize=True,
                    return_tensors="pt",
                    add_generation_prompt=False,
                    truncation=True,
                    max_length=self.config["max_length"]
                )
            else:
                encoded = encoded_full
            
            input_ids_list.append(encoded[0])
        
        if truncation_count > 0:
            print(f"Warning: {truncation_count}/{len(data_items)} samples exceeded max_length ({self.config['max_length']}) and were truncated.")

        batch = self.tokenizer.pad(
            {"input_ids": input_ids_list},
            padding=True,
            return_tensors="pt"
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
                'name', 'InfOrmScorer'))
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

