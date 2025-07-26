# import re
import torch
from .base_scorer import BaseScorer
import json
from typing import Dict, List
from transformers import AutoTokenizer
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from .utils import get_total_lines
import json
from tqdm import tqdm


class IFDScorer(BaseScorer):
    def _validate_config(self):
        if "model" not in self.config:
            print(
                "Warning: No loacl model specified in config. Downloading the remote huggingface model.")
            self.config['model'] = 'openai-community/gpt2'
        else:
            if not os.path.exists(self.config["model"]):
                print(
                    f"Warning: Specified local model path '{self.config['model']}' does not exist. "
                    "Downloading the remote huggingface model: openai-community/gpt2"
                )
                self.config['model'] = 'openai-community/gpt2'
            else:
                print(
                    f"Using specified local model: '{self.config['model']}'. "
                )

        if "max_length" in self.config and isinstance(self.config["max_length"], int) and self.config["max_length"] > 0:
            print(
                f"Using specified max_length: {self.config['max_length']}.")
        else:
            print(
                "Warning: No specific max_length, use default value of 2048.")
            self.config['max_length'] = 2048

        if "template" in self.config and isinstance(self.config["template"], str):
            print(
                f"Using specified template: {self.config['template']}.")
        else:
            print(
                "Warning: No specific template, use default value of qwen2.")
            self.config['template_no_input'] = "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
            self.config['template'] = "<|im_start|>user\n{instruction}\n{input}<|im_end|>\n<|im_start|>assistant\n"

    def _setup(self):
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config['model'], device_map="auto", cache_dir='../cache', output_hidden_states=True)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config['model'], cache_dir='./cache')
        except:
            print(
                "Warning: Failed to load model from remote. Loading openai-community/gpt2 model.")
            self.model = AutoModelForCausalLM.from_pretrained(
                'openai-community/gpt2', device_map="auto", cache_dir='../cache', output_hidden_states=True)
            self.tokenizer = AutoTokenizer.from_pretrained(
                'openai-community/gpt2', cache_dir='./cache')
        self.model.eval()

        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            print("Warning: No GPU available. Using CPU.")
            self.device = "cpu"
        print("Setting up IFDScorer successfully")

    def get_perplexity_and_embedding_whole_text(self, text, max_length):
        try:
            input_ids = self.tokenizer.encode(
                text, return_tensors="pt", truncation=True, max_length=max_length).to(self.device)
            with torch.no_grad():
                outputs = self.model(input_ids, labels=input_ids.contiguous())
            loss = outputs.loss
            perplexity = torch.exp(loss)

            return perplexity.to('cpu').item(), loss.to('cpu').item()
        except Exception as e:
            print(f"Error in get_perplexity_and_embedding_whole_text: {e}")
            return 0, 0

    def get_perplexity_and_embedding_part_text(self, text, target_span):
        try:
            input_ids = self.tokenizer.encode(
                text, return_tensors="pt", truncation=True, max_length=self.config['max_length']).to(self.device)

            start_index = text.rfind(target_span)
            start_token = len(self.tokenizer.encode(text[:start_index]))
            end_token = input_ids.shape[1]

            labels = input_ids.clone()
            labels[0, :start_token] = -100

            with torch.no_grad():
                outputs = self.model(input_ids, labels=labels)

            loss = outputs.loss
            perplexity = torch.exp(loss)

            return perplexity.to('cpu').item(), loss.to('cpu').item()
        except Exception as e:
            print(f"Error in get_perplexity_and_embedding_part_text: {e}")
            return 0, 0

    def score_item(self, data_item):
        instruction = data_item.get("instruction", "")
        _input = data_item.get("input", "")
        output = data_item.get("output", "")
        if output is None or (isinstance(output, str) and len(output) == 0):
            print(f"data_item's output is empty: {data_item}, return -1")
            return -1
        if _input is None or (isinstance(_input, str) and len(_input) == 0):
            prompt = self.config['template_no_input'].format(
                instruction=instruction)
        else:
            prompt = self.config['template'].format(
                instruction=instruction, input=_input)
        whole_text = prompt + output

        try:
            instruct_input_ids = self.tokenizer.encode(
                prompt, return_tensors="pt", truncation=True, max_length=self.config['max_length']).to(self.device)
            instruct_len = instruct_input_ids.shape[1]

            ppl_out_alone, loss_out_alone = self.get_perplexity_and_embedding_whole_text(
                output, self.config['max_length']-instruct_len+1)
            ppl_out_condition, loss_out_condition = self.get_perplexity_and_embedding_part_text(
                whole_text, output)
        except Exception as e:
            print(f"Error in score_item: {e}")
            return -1
        return ppl_out_condition/ppl_out_alone

    def evaluate(self, dataset) -> List[Dict]:
        num_lines = get_total_lines(dataset)
        results = []
        with open(dataset, 'r') as f:
            for line in tqdm(f, total=num_lines, desc=self.config['name']):
                item = json.loads(line.strip())
                res = {
                    "id": item.get("id", ""),
                    "IFD": self.score_item(item)
                }
                results.append(res)
        return results
