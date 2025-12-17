import torch
from .base_scorer import BaseScorer
import json
from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from tqdm import tqdm
from .utils import get_total_lines


class InstagScorer(BaseScorer):
    def _validate_config(self):
        # Check if a local model path is specified in config, otherwise use default remote download
        if "model" not in self.config:
            print(
                "Warning: No local model specified in config. Downloading the remote huggingface model.")
            self.config['model'] = 'OFA-Sys/InsTagger'
        else:
            print(f"Using specified local model: '{self.config['model']}'. ")

        # Validate max_new_tokens configuration
        # Model max length is 2048
        MAX_MODEL_LENGTH = 2048
        
        if "max_new_tokens" in self.config and isinstance(self.config["max_new_tokens"], int) and self.config["max_new_tokens"] > 0:
            if self.config["max_new_tokens"] >= MAX_MODEL_LENGTH:
                print(
                    f"Warning: max_new_tokens ({self.config['max_new_tokens']}) exceeds model max_length ({MAX_MODEL_LENGTH}). "
                    f"Setting max_new_tokens to {MAX_MODEL_LENGTH - 1}.")
                self.config['max_new_tokens'] = MAX_MODEL_LENGTH - 1
            else:
                print(
                    f"Using specified max_new_tokens: {self.config['max_new_tokens']}.")
        elif "max_new_tokens" in self.config and isinstance(self.config["max_new_tokens"], int) and self.config["max_new_tokens"] <= 0:
            print(
                "Warning: the specific max_new_tokens should > 0. use default value of 512.")
            self.config['max_new_tokens'] = 512
        else:
            print("Warning: No specific max_new_tokens, use default value of 512.")
            self.config['max_new_tokens'] = 512

        # Validate batch_size configuration
        if "batch_size" not in self.config or not isinstance(self.config["batch_size"], int) or self.config["batch_size"] <= 0:
            self.config["batch_size"] = 8
            print("Warning: No/invalid batch_size, use default value of 8.")
        else:
            print(f"Using specified batch_size: {self.config['batch_size']}.")

    def _setup(self):
        """Initialize model and tokenizer"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config['model'])
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config['model'])
        except Exception as e:
            print(
                f"Load specified model failed ({e}), fall back to OFA-Sys/InsTagger")
            self.model = AutoModelForCausalLM.from_pretrained(
                'OFA-Sys/InsTagger')
            self.tokenizer = AutoTokenizer.from_pretrained(
                'OFA-Sys/InsTagger')

        # Set pad_token if not available
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # For Causal LM, must use left padding
        self.tokenizer.padding_side = "left"

        self.model.to(self.device)
        self.model.eval()
        print("Setting up InstagScorer successfully")

    def make_prompt(self, query: str) -> str:
        """Generate prompt text"""
        prompt = f"Please identify tags of user intentions in the following user query and provide an explanation for each tag. Please respond in the JSON format {{\"tag\": str, \"explanation\": str}}.\nUser query: {query}"
        messages = [("USER", prompt), ("ASSISTANT", None)]
        seps = [" ", "</s>"]
        ret = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions." + \
            seps[0]
        for i, (role, message) in enumerate(messages):
            if message:
                ret += role + ": " + message + seps[i % 2]
            else:
                ret += role + ":"
        return ret

    def score_item(self, data_item: Dict) -> int:
        """Score a single data item"""
        return self.score_batch([data_item])[0]

    def score_batch(self, data_items: List[Dict]) -> List[int]:
        """Score data items in batch"""
        # Extract instructions from data items
        queries = []
        for item in data_items:
            query = item["instruction"]
            input_text = item.get("input", "")
            if input_text:
                query = query + "\n" + input_text
            queries.append(query)

        # Generate prompts in batch
        input_strs = [self.make_prompt(query) for query in queries]
        # Model max length is 2048, need to reserve space for max_new_tokens generation
        max_input_length = 2048 - self.config['max_new_tokens']
        input_tokens = self.tokenizer(
            input_strs, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=max_input_length
        ).to(self.device)
        
        # Check if any data was truncated
        truncated_count = 0
        for i in range(input_tokens['input_ids'].shape[0]):
            # Count non-padding tokens
            actual_length = (input_tokens['input_ids'][i] != self.tokenizer.pad_token_id).sum().item()
            if actual_length >= max_input_length:
                truncated_count += 1
        
        if truncated_count > 0:
            print(f"Warning: {truncated_count} out of {len(input_strs)} samples were truncated due to max_length={max_input_length}")

        with torch.no_grad():
            output = self.model.generate(
                input_tokens['input_ids'],
                attention_mask=input_tokens['attention_mask'],
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=False,
                max_new_tokens=self.config['max_new_tokens'],
                num_return_sequences=1,
                return_dict_in_generate=True,
            )

        num_input_tokens = input_tokens["input_ids"].shape[1]
        output_tokens = output.sequences
        generated_tokens = output_tokens[:, num_input_tokens:]
        generated_texts = self.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True)

        # Parse output and calculate scores
        scores = []
        for generated_text in generated_texts:
            string_output = generated_text.strip()
            try:
                json_output = json.loads(string_output)
                if isinstance(json_output, list):
                    complexity_score = len(json_output)
                elif isinstance(json_output, dict):
                    complexity_score = 1
                else:
                    complexity_score = 0
            except json.JSONDecodeError:
                complexity_score = 0
            scores.append(int(complexity_score))

        return scores

    def evaluate(self, dataset) -> List[Dict]:
        """Evaluate the entire dataset"""
        num_lines = get_total_lines(dataset)
        results: List[Dict] = []

        batch_size = self.config.get("batch_size")
        buffer_items = []
        buffer_ids = []

        with open(dataset, 'r') as f:
            pbar = tqdm(total=num_lines, desc=self.config.get(
                'name', 'InstagScorer'))
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

            # Process remaining data
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
