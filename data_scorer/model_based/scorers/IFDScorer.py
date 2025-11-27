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

        if "batch_size" in self.config and isinstance(self.config["batch_size"], int) and self.config["batch_size"] > 0:
            print(
                f"Using specified batch_size: {self.config['batch_size']}.")
        else:
            print(
                "Warning: No specific batch_size, use default value of 1.")
            self.config['batch_size'] = 1

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
            # Tokenize without truncation first to check length
            full_tokens = self.tokenizer.encode(text, return_tensors="pt")
            if full_tokens.shape[1] > max_length:
                print(f"Warning: Text length ({full_tokens.shape[1]} tokens) exceeds max_length ({max_length}), truncating.")
            
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
            # Tokenize without truncation first to check length
            full_tokens = self.tokenizer.encode(text, return_tensors="pt")
            if full_tokens.shape[1] > self.config['max_length']:
                print(f"Warning: Text length ({full_tokens.shape[1]} tokens) exceeds max_length ({self.config['max_length']}), truncating.")
            
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

    def get_batch_perplexity_whole_text(self, texts, max_length):
        """Compute perplexity for a batch of texts"""
        try:
            # Check for truncation before tokenizing
            for idx, text in enumerate(texts):
                full_tokens = self.tokenizer.encode(text, return_tensors="pt")
                if full_tokens.shape[1] > max_length:
                    print(f"Warning: Text {idx} length ({full_tokens.shape[1]} tokens) exceeds max_length ({max_length}), truncating.")
            
            # Batch encoding
            encodings = self.tokenizer(
                texts, 
                return_tensors="pt", 
                truncation=True, 
                max_length=max_length,
                padding=True
            ).to(self.device)
            
            input_ids = encodings['input_ids']
            attention_mask = encodings['attention_mask']
            
            # Create labels, ignoring padding parts
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
            
            # Calculate perplexity for each sample
            perplexities = []
            losses = []
            
            for i in range(len(texts)):
                # Get valid tokens for each sample
                valid_mask = attention_mask[i] == 1
                sample_input_ids = input_ids[i:i+1, valid_mask]
                sample_labels = sample_input_ids.clone()
                
                with torch.no_grad():
                    sample_outputs = self.model(sample_input_ids, labels=sample_labels)
                
                loss = sample_outputs.loss
                perplexity = torch.exp(loss)
                perplexities.append(perplexity.to('cpu').item())
                losses.append(loss.to('cpu').item())
            
            return perplexities, losses
        except Exception as e:
            print(f"Error in get_batch_perplexity_whole_text: {e}")
            return [0] * len(texts), [0] * len(texts)

    def get_batch_perplexity_part_text(self, texts, target_spans):
        """Compute conditional perplexity for a batch of texts"""
        try:
            perplexities = []
            losses = []
            
            for idx, (text, target_span) in enumerate(zip(texts, target_spans)):
                # Check for truncation
                full_tokens = self.tokenizer.encode(text, return_tensors="pt")
                if full_tokens.shape[1] > self.config['max_length']:
                    print(f"Warning: Text {idx} length ({full_tokens.shape[1]} tokens) exceeds max_length ({self.config['max_length']}), truncating.")
                
                input_ids = self.tokenizer.encode(
                    text, return_tensors="pt", truncation=True, max_length=self.config['max_length']).to(self.device)
                
                start_index = text.rfind(target_span)
                start_token = len(self.tokenizer.encode(text[:start_index]))
                
                labels = input_ids.clone()
                labels[0, :start_token] = -100
                
                with torch.no_grad():
                    outputs = self.model(input_ids, labels=labels)
                
                loss = outputs.loss
                perplexity = torch.exp(loss)
                
                perplexities.append(perplexity.to('cpu').item())
                losses.append(loss.to('cpu').item())
            
            return perplexities, losses
        except Exception as e:
            print(f"Error in get_batch_perplexity_part_text: {e}")
            return [0] * len(texts), [0] * len(texts)

    def score_item(self, data_item):
        instruction = data_item["instruction"]
        _input = data_item.get("input", "")
        output = data_item["output"]
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

    def score_batch(self, data_items):
        """Score multiple data items in batch"""
        batch_outputs = []
        batch_prompts = []
        batch_whole_texts = []
        valid_indices = []
        
        # Prepare batch data
        for idx, data_item in enumerate(data_items):
            instruction = data_item["instruction"]
            _input = data_item.get("input", "")
            output = data_item["output"]
            
            if output is None or (isinstance(output, str) and len(output) == 0):
                print(f"data_item's output is empty: {data_item}, return -1")
                continue
            
            if _input is None or (isinstance(_input, str) and len(_input) == 0):
                prompt = self.config['template_no_input'].format(
                    instruction=instruction)
            else:
                prompt = self.config['template'].format(
                    instruction=instruction, input=_input)
            
            whole_text = prompt + output
            
            batch_outputs.append(output)
            batch_prompts.append(prompt)
            batch_whole_texts.append(whole_text)
            valid_indices.append(idx)
        
        if len(batch_outputs) == 0:
            return [-1] * len(data_items)
        
        try:
            # Calculate prompt length to determine max output length
            max_output_length = self.config['max_length']
            for prompt in batch_prompts:
                instruct_input_ids = self.tokenizer.encode(
                    prompt, return_tensors="pt", truncation=True, max_length=self.config['max_length'])
                instruct_len = instruct_input_ids.shape[1]
                max_output_length = min(max_output_length, self.config['max_length'] - instruct_len + 1)
            
            # Batch compute perplexity of outputs alone
            ppl_out_alone_list, _ = self.get_batch_perplexity_whole_text(
                batch_outputs, max_output_length)
            
            # Batch compute conditional perplexity
            ppl_out_condition_list, _ = self.get_batch_perplexity_part_text(
                batch_whole_texts, batch_outputs)
            
            # Calculate IFD scores
            scores = [-1] * len(data_items)
            for i, idx in enumerate(valid_indices):
                if ppl_out_alone_list[i] > 0:
                    scores[idx] = ppl_out_condition_list[i] / ppl_out_alone_list[i]
                else:
                    scores[idx] = -1
            
            return scores
        except Exception as e:
            print(f"Error in score_batch: {e}")
            return [-1] * len(data_items)

    def evaluate(self, dataset) -> List[Dict]:
        num_lines = get_total_lines(dataset)
        results = []
        batch_size = self.config.get('batch_size', 1)
        
        if batch_size == 1:
            # Single sample processing mode (maintain backward compatibility)
            with open(dataset, 'r') as f:
                for line in tqdm(f, total=num_lines, desc=self.config['name']):
                    item = json.loads(line.strip())
                    res = {
                        "id": item.get("id", ""),
                        "score": self.score_item(item)
                    }
                    results.append(res)
        else:
            # Batch processing mode
            with open(dataset, 'r') as f:
                batch_items = []
                batch_ids = []
                
                for line in tqdm(f, total=num_lines, desc=self.config['name']):
                    item = json.loads(line.strip())
                    batch_items.append(item)
                    batch_ids.append(item.get("id", ""))
                    
                    if len(batch_items) >= batch_size:
                        # Process current batch
                        scores = self.score_batch(batch_items)
                        for item_id, score in zip(batch_ids, scores):
                            results.append({
                                "id": item_id,
                                "score": score
                            })
                        batch_items = []
                        batch_ids = []
                
                # Process the last incomplete batch
                if batch_items:
                    scores = self.score_batch(batch_items)
                    for item_id, score in zip(batch_ids, scores):
                        results.append({
                            "id": item_id,
                            "score": score
                        })
        
        return results
