import torch
from .base_scorer import BaseScorer
import json
import os
from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from .utils import get_total_lines
import numpy as np

class QuRateScorer(BaseScorer):
    def _validate_config(self):
        """Validate configuration parameters"""
        # Validate model parameter
        if "model" not in self.config:
            print(
                "Warning: No local model specified in config. Downloading the remote huggingface model.")
            self.config['model'] = 'princeton-nlp/QuRater-1.3B'
        else:
            if self.config['model'] == 'princeton-nlp/QuRater-1.3B':
                print("Downloading and use the specific remote huggingface model.")
            elif not os.path.exists(self.config["model"]):
                print(
                    f"Warning: Specified local model path '{self.config['model']}' does not exist. "
                    "Downloading the remote huggingface model: princeton-nlp/QuRater-1.3B"
                )
                self.config['model'] = 'princeton-nlp/QuRater-1.3B'
            else:
                print(
                    f"Using specified local model: '{self.config['model']}'. ")
        
        # Validate labels parameter
        if "labels" not in self.config or not isinstance(self.config["labels"], list):
            self.config["labels"] = ["writing_style", "required_expertise", "facts_and_trivia", "educational_value"]
            print("Warning: No/invalid labels, use default value of ['writing_style', 'required_expertise', 'facts_and_trivia', 'educational_value'].")

        if len(self.config["labels"]) == 0:
            self.config["labels"] = ["writing_style", "required_expertise", "facts_and_trivia", "educational_value"]
            print("Warning: Labels list cannot be empty, use default value of ['writing_style', 'required_expertise', 'facts_and_trivia', 'educational_value'].")
        
        print(f"Using labels: {self.config['labels']}")
        
        # Validate chunk_size (tokens)
        if "chunk_size" in self.config and isinstance(self.config["chunk_size"], int) and self.config["chunk_size"] > 0:
            print(f"Using specified chunk_size: {self.config['chunk_size']}.")
        else:
            print("Warning: No/invalid chunk_size, use default value of 512.")
            self.config['chunk_size'] = 512
        
        # Validate batch_size
        if "batch_size" not in self.config or not isinstance(self.config["batch_size"], int) or self.config["batch_size"] <= 0:
            self.config["batch_size"] = 8
            print("Warning: No/invalid batch_size, use default value of 8.")
        else:
            print(f"Using specified batch_size: {self.config['batch_size']}.")
        
        # Validate device_batch_size
        if "device_batch_size" not in self.config or not isinstance(self.config["device_batch_size"], int) or self.config["device_batch_size"] <= 0:
            self.config["device_batch_size"] = 16
            print("Warning: No/invalid device_batch_size, use default value of 16.")
        else:
            print(f"Using specified device_batch_size: {self.config['device_batch_size']}.")

    def _setup(self):
        """Initialize model and tokenizer"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config['model'], use_fast=True)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = 0
            
            # Load model - using standard transformers implementation
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config['model'],
                torch_dtype=torch.bfloat16,
                trust_remote_code=False  # Use standard implementation without custom code
            )
            if self.model.config.pad_token_id is None:
                self.model.config.pad_token_id = 0
            self.model.eval()
            self.model.to(self.device)
            
            # Validate number of labels
            self.num_labels = len(self.config['labels'])
            if self.num_labels != self.model.config.num_labels:
                raise ValueError(
                    f"Number of labels ({self.num_labels}) does not match "
                    f"model config ({self.model.config.num_labels})"
                )
            
            print("Setting up QuRateScorer successfully")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def _tokenize_and_chunk(self, text: str) -> tuple:
        """Tokenize and chunk the text"""
        # Tokenize the text
        tokens = self.tokenizer(
            text, 
            truncation=False, 
            padding=False, 
            add_special_tokens=False
        ).input_ids
        
        # Split tokens into chunks
        chunks = torch.tensor(tokens, dtype=torch.long).split(self.config['chunk_size'])
        chunks_token_ids = [chunk.tolist() for chunk in chunks]
        chunks_token_counts = [len(chunk) for chunk in chunks]
        
        return chunks_token_ids, chunks_token_counts

    @torch.inference_mode()
    def _score_chunks(self, chunks_token_ids: List[List[int]], chunks_token_counts: List[int]) -> torch.Tensor:
        """Score text chunks"""
        chunks_token_counts_tensor = torch.tensor(chunks_token_counts, dtype=torch.long)
        sorted_indices = torch.argsort(chunks_token_counts_tensor)
        
        scores = torch.zeros(len(chunks_token_ids), self.num_labels, dtype=torch.float32)
        
        for batch_indices in sorted_indices.split(self.config['device_batch_size']):
            max_len = chunks_token_counts_tensor[batch_indices].max().item()
            
            input_ids = torch.zeros((len(batch_indices), max_len), dtype=torch.long)
            attention_mask = torch.zeros((len(batch_indices), max_len), dtype=torch.long)
            
            for i, j in enumerate(batch_indices):
                seq = torch.tensor(chunks_token_ids[j], dtype=torch.long)
                input_ids[i, :len(seq)] = seq
                attention_mask[i, :len(seq)] = 1
            
            outputs = self.model(
                input_ids.to(self.device), 
                attention_mask=attention_mask.to(self.device), 
                use_cache=False
            )
            scores[batch_indices] = outputs.logits.float().cpu()
        
        return scores

    def score_item(self, data_item: Dict) -> Dict:
        """Score a single sample"""
        return self.score_batch([data_item])[0]

    def score_batch(self, data_items: List[Dict]) -> List[Dict]:
        """Score a batch of samples"""
        batch_results = []
        
        for item in data_items:
            # Build text
            instruction = item["instruction"]
            input_text = item.get("input", "")
            response = item["output"]
            
            if input_text:
                text = instruction + '\n' + input_text + '\n' + response
            else:
                text = instruction + '\n' + response
            
            # Tokenize and chunk
            chunks_token_ids, chunks_token_counts = self._tokenize_and_chunk(text)
            
            # Score chunks
            chunk_scores = self._score_chunks(chunks_token_ids, chunks_token_counts)
            
            # Calculate weighted average scores
            result = {
                "length": sum(chunks_token_counts),
                "num_chunks": len(chunks_token_ids)
            }
            
            for i, label in enumerate(self.config['labels']):
                label_scores = chunk_scores[:, i].numpy()
                # Calculate weighted average using token counts as weights
                weighted_avg = np.average(label_scores, weights=chunks_token_counts)
                result[f"{label}_score"] = float(weighted_avg)
                result[f"{label}_chunks"] = label_scores.tolist()
            
            batch_results.append(result)
        
        return batch_results

    def evaluate(self, dataset: str) -> List[Dict]:
        """Score the entire dataset"""
        num_lines = get_total_lines(dataset)
        results: List[Dict] = []
        
        batch_size = self.config.get("batch_size")
        buffer_items = []
        buffer_ids = []
        
        with open(dataset, 'r', encoding='utf-8') as f:
            pbar = tqdm(total=num_lines, desc=self.config.get('name', 'QuRateScorer'))
            
            for line in f:
                item = json.loads(line.strip())
                buffer_items.append(item)
                buffer_ids.append(item.get("id", ""))
                
                if len(buffer_items) == batch_size:
                    batch_scores = self.score_batch(buffer_items)
                    
                    for id_, scores_dict in zip(buffer_ids, batch_scores):
                        result = {"id": id_}
                        result.update(scores_dict)
                        results.append(result)
                    
                    pbar.update(len(buffer_items))
                    buffer_items.clear()
                    buffer_ids.clear()
            
            # Process remaining samples
            if buffer_items:
                batch_scores = self.score_batch(buffer_items)
                
                for id_, scores_dict in zip(buffer_ids, batch_scores):
                    result = {"id": id_}
                    result.update(scores_dict)
                    results.append(result)
                
                pbar.update(len(buffer_items))
                buffer_items.clear()
                buffer_ids.clear()
            
            pbar.close()
        
        return results
