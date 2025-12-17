import torch
import numpy as np
import json
import os
from typing import Dict, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from .base_scorer import BaseScorer
from .utils import get_total_lines, get_distance_function


class MIWVScorer(BaseScorer):
    """
    MIWV (Most Influential Weight Vector) Scorer
    
    Calculation method:
    1. Find the most similar data for each sample based on the embedding file
    2. Calculate one-shot ICL loss (using the most similar data as example)
    3. Calculate zero-shot loss
    4. MIWV = one-shot loss - zero-shot loss
    """
    
    def _validate_config(self):
        # Validate model path
        if "model" not in self.config:
            raise ValueError("Model path must be specified in config")
        
        if not os.path.exists(self.config["model"]):
            raise ValueError(f"Model path does not exist: {self.config['model']}")
        
        print(f"Using model: {self.config['model']}")
        
        # Validate embedding file
        if "embedding_path" not in self.config:
            raise ValueError("Embedding path must be specified in config")
        
        if not os.path.exists(self.config["embedding_path"]):
            raise ValueError(f"Embedding file does not exist: {self.config['embedding_path']}")
        
        print(f"Using embedding file: {self.config['embedding_path']}")
        
        # Validate batch_size
        if "batch_size" not in self.config or not isinstance(self.config["batch_size"], int) or self.config["batch_size"] <= 0:
            self.config["batch_size"] = 8
            print("Warning: batch_size not specified or invalid, using default value 8")
        else:
            print(f"Using batch_size: {self.config['batch_size']}")
        
        # Validate max_length
        if "max_length" not in self.config or not isinstance(self.config["max_length"], int) or self.config["max_length"] <= 0:
            self.config["max_length"] = 2048
            print("Warning: max_length not specified or invalid, using default value 2048")
        else:
            print(f"Using max_length: {self.config['max_length']}")
        
        # Validate distance_metric
        if "distance_metric" not in self.config:
            self.config["distance_metric"] = "cosine"
            print("Warning: distance_metric not specified, using default value cosine")
        else:
            valid_metrics = ["euclidean", "squared_euclidean", "manhattan", "cosine"]
            if self.config["distance_metric"] not in valid_metrics:
                print(
                    f"Warning: Invalid distance_metric '{self.config['distance_metric']}', "
                    f"using default value cosine. Available metrics: {', '.join(valid_metrics)}"
                )
                self.config["distance_metric"] = "cosine"
            else:
                print(f"Using distance_metric: {self.config['distance_metric']}")
    
    def _setup(self):
        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer
        try:
            print("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config['model'],
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config['model'],
                trust_remote_code=True,
            )
            
            # Set pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            if not torch.cuda.is_available() or self.model.device.type != "cuda":
                self.model.to(self.device)
            
            self.tokenizer.padding_side="right"
            self.model.eval()
            print("Model loaded successfully")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
        
        # Load embedding file
        try:
            print("Loading embedding file...")
            self.embeddings = np.load(self.config['embedding_path'])
            print(f"Embedding shape: {self.embeddings.shape}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding file: {e}")
        
        # Get distance metric and calculation function
        distance_metric = self.config["distance_metric"]
        self.distance_metric = distance_metric
        self.distance_func = get_distance_function(distance_metric)
        print(f"Using distance metric: {distance_metric}")
        
        # Calculate distance matrix and find the most similar sample for each sample (sample with minimum distance)
        print("Computing most similar samples...")
        self._compute_most_similar_indices()
        print("Most similar samples computation completed")
        
        print("MIWVScorer setup completed")
    
    def _compute_most_similar_indices(self):
        """Compute the most similar sample index for each sample (sample with minimum distance, excluding itself)"""
        n_samples = len(self.embeddings)
        self.most_similar_indices = []
        
        distance_metric = self.distance_metric
        
        # For efficiency, use vectorized computation for cosine distance
        if distance_metric == "cosine":
            # Normalize embeddings
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            # Avoid division by zero
            norms = np.where(norms == 0, 1, norms)
            normalized_embeddings = self.embeddings / norms
            
            # Calculate cosine similarity matrix
            similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
            
            # Cosine distance = 1 - cosine similarity
            distance_matrix = 1.0 - similarity_matrix
            
            # Set diagonal to inf to exclude self
            np.fill_diagonal(distance_matrix, np.inf)
            
            # Find the index of minimum value in each row (minimum distance = most similar)
            self.most_similar_indices = np.argmin(distance_matrix, axis=1).tolist()
            
        else:
            # For other distance metrics, compute one by one
            for i in tqdm(range(n_samples), desc="Computing most similar samples"):
                min_dist = np.inf
                min_idx = -1
                
                for j in range(n_samples):
                    if i == j:
                        continue
                    
                    # Use distance function to calculate distance
                    dist = self.distance_func(self.embeddings[i], self.embeddings[j])
                    
                    if dist < min_dist:
                        min_dist = dist
                        min_idx = j
                
                self.most_similar_indices.append(min_idx)
    
    def _build_text_from_item(self, prompt: str, output: str, 
                              example_prompt: str = None, example_output: str = None) -> str:
        """Build text from data (using direct string concatenation)
        
        Args:
            prompt: Instruction of current sample
            output: Output of current sample
            example_prompt: Instruction of ICL example (optional)
            example_output: Output of ICL example (optional)
        
        Returns:
            Constructed text
        """
        # Use direct string concatenation, not chat_template
        if example_prompt is not None and example_output is not None:
            # One-shot ICL
            full_text = (
                f"User: {example_prompt}\n"
                f"Assistant: {example_output}\n"
                f"User: {prompt}\n"
                f"Assistant: {output}"
            )
        else:
            # Zero-shot
            full_text = f"User: {prompt}\nAssistant: {output}"
        
        return full_text
    
    def _get_prompt_length(self, prompt: str, 
                          example_prompt: str = None, example_output: str = None) -> int:
        """Get the token length of the prompt part (using direct string concatenation)"""
        # Use direct string concatenation, not chat_template
        if example_prompt is not None and example_output is not None:
            prompt_text = (
                f"User: {example_prompt}\n"
                f"Assistant: {example_output}\n"
                f"User: {prompt}\n"
                f"Assistant: "
            )
        else:
            prompt_text = f"User: {prompt}\nAssistant: "
        
        encoding = self.tokenizer(prompt_text, return_tensors="pt", truncation=False, padding=False)
        return encoding.input_ids.shape[1]
    
    def _compute_batch_loss(self, texts: List[str], prompt_lengths: List[int]) -> tuple[List[float], List[bool]]:
        """Compute loss in batch (using model forward to get loss directly)
        
        Args:
            texts: List of complete texts
            prompt_lengths: List of prompt lengths for each text
        
        Returns:
            (List of loss for each sample, List of whether each sample was truncated)
        """
        # First check original lengths (without truncation, process one by one)
        original_lengths = []
        for text in texts:
            encoding = self.tokenizer(text, truncation=False, padding=False)
            original_lengths.append(len(encoding.input_ids))
        
        # Batch tokenize (with truncation)
        encodings = self.tokenizer(
            texts,
            return_tensors="pt",
            max_length=self.config["max_length"],
            truncation=True,
            padding=True,
        )
        
        input_ids = encodings.input_ids.to(self.device)
        attention_mask = encodings.attention_mask.to(self.device)
        
        # Detect which samples were truncated
        truncated = [orig_len > self.config["max_length"] for orig_len in original_lengths]
        
        # Warn if any samples were truncated
        if any(truncated):
            num_truncated = sum(truncated)
            print(f"Warning: {num_truncated} sample(s) in this batch exceeded max_length ({self.config['max_length']}) and were truncated")
        
        # Construct labels, set prompt part to -100
        labels = input_ids.clone()
        
        for i in range(len(texts)):
            # Get the actual sequence length of current sample (excluding padding)
            actual_length = attention_mask[i].sum().item()
            
            # For truncated samples, prompt length may not be accurate
            if truncated[i]:
                # When truncated, prompt_lengths[i] is the length before truncation
                # If the prompt itself exceeds max_length, the entire sequence might be prompt
                # To be safe, we limit prompt length to not exceed actual_length-1,
                # keeping at least one token for calculating output loss
                prompt_len = min(prompt_lengths[i], max(0, actual_length - 1))
            else:
                # Not truncated, use original prompt length
                prompt_len = prompt_lengths[i]
            
            # Ensure prompt_len does not exceed actual sequence length
            prompt_len = min(prompt_len, actual_length)
            
            # Set label of prompt part to -100 (do not calculate loss)
            if prompt_len > 0:
                labels[i, :prompt_len] = -100
            
            # Set padding part to -100 as well
            labels[i, attention_mask[i] == 0] = -100
        
        losses = []
        
        with torch.no_grad():
            # Use model forward to calculate loss directly
            # Here we need to calculate loss for each sample separately
            for i in range(len(texts)):
                sample_input_ids = input_ids[i:i+1]  # [1, seq_len]
                sample_attention_mask = attention_mask[i:i+1]  # [1, seq_len]
                sample_labels = labels[i:i+1]  # [1, seq_len]
                
                # Check if there are valid labels (not all -100)
                if (sample_labels != -100).sum() == 0:
                    # No valid output tokens, set loss to 0
                    losses.append(0.0)
                    continue
                
                outputs = self.model(
                    input_ids=sample_input_ids,
                    attention_mask=sample_attention_mask,
                    labels=sample_labels
                )
                
                # Get loss directly
                loss = outputs.loss
                losses.append(loss.item())
        
        return losses, truncated
    
    def score_item(self, data_item: Dict, item_index: int, all_data: List[Dict]) -> float:
        """
        Calculate MIWV score for a single sample
        
        Args:
            data_item: Data sample
            item_index: Index of sample in the dataset
            all_data: Complete dataset (for getting the most similar sample)
        
        Returns:
            MIWV score
        """
        scores, _ = self.score_batch([data_item], [item_index], all_data)
        return scores[0]
    
    def score_batch(self, data_items: List[Dict], item_indices: List[int], all_data: List[Dict]) -> tuple[List[float], List[bool]]:
        """
        Calculate MIWV scores in batch
        
        Args:
            data_items: List of data samples
            item_indices: List of sample indices in the dataset
            all_data: Complete dataset (for getting the most similar samples)
        
        Returns:
            (List of MIWV scores, List of truncation flags)
        """
        # Construct all zero-shot and one-shot texts
        zero_shot_texts = []
        zero_shot_prompt_lengths = []
        one_shot_texts = []
        one_shot_prompt_lengths = []
        
        for data_item, item_idx in zip(data_items, item_indices):
            prompt = data_item["instruction"]
            input_text = data_item.get("input", "")
            if input_text:
                prompt = prompt + '\n' + input_text
            output = data_item["output"]
            
            # Construct zero-shot text
            zero_shot_text = self._build_text_from_item(prompt, output)
            zero_shot_texts.append(zero_shot_text)
            zero_shot_prompt_lengths.append(self._get_prompt_length(prompt))
            
            # Get most similar sample and construct one-shot text
            most_similar_idx = self.most_similar_indices[item_idx]
            similar_item = all_data[most_similar_idx]
            similar_prompt = similar_item["instruction"]
            similar_input_text = similar_item.get("input", "")
            if similar_input_text:
                similar_prompt = similar_prompt + '\n' + similar_input_text
            similar_output = similar_item["output"]
            
            one_shot_text = self._build_text_from_item(
                prompt, output, similar_prompt, similar_output
            )
            one_shot_texts.append(one_shot_text)
            one_shot_prompt_lengths.append(
                self._get_prompt_length(prompt, similar_prompt, similar_output)
            )
        
        # Calculate zero-shot loss in batch
        zero_shot_losses, zero_shot_truncated = self._compute_batch_loss(zero_shot_texts, zero_shot_prompt_lengths)
        
        # Calculate one-shot loss in batch
        one_shot_losses, one_shot_truncated = self._compute_batch_loss(one_shot_texts, one_shot_prompt_lengths)
        
        # Calculate MIWV score = one-shot loss - zero-shot loss
        miwv_scores = [
            one_shot - zero_shot 
            for one_shot, zero_shot in zip(one_shot_losses, zero_shot_losses)
        ]
        
        # Mark as truncated if either zero-shot or one-shot is truncated
        truncated = [z_trunc or o_trunc for z_trunc, o_trunc in zip(zero_shot_truncated, one_shot_truncated)]
        
        return miwv_scores, truncated
    
    def evaluate(self, dataset: str) -> List[Dict]:
        """
        Evaluate the entire dataset (using batch parallel processing)
        
        Args:
            dataset: Path to jsonl file
        
        Returns:
            List of evaluation results
        """
        # First read all data
        print("Reading dataset...")
        all_data = []
        with open(dataset, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                all_data.append(item)
        
        print(f"Dataset size: {len(all_data)}")
        
        # Validate that dataset size matches embedding size
        if len(all_data) != len(self.embeddings):
            raise ValueError(
                f"Dataset size ({len(all_data)}) does not match embedding size ({len(self.embeddings)})"
            )
        
        results: List[Dict] = []
        batch_size = self.config["batch_size"]
        
        # Process data in batches
        num_lines = len(all_data)
        buffer_items = []
        buffer_indices = []
        buffer_ids = []
        
        pbar = tqdm(total=num_lines, desc=self.config.get('name', 'MIWVScorer'))
        
        for i in range(num_lines):
            item = all_data[i]
            buffer_items.append(item)
            buffer_indices.append(i)
            buffer_ids.append(item.get("id", i))
            
            # When buffer is full or it's the last batch, perform batch scoring
            if len(buffer_items) == batch_size or i == num_lines - 1:
                # Calculate MIWV scores in batch
                miwv_scores, truncated_flags = self.score_batch(buffer_items, buffer_indices, all_data)
                
                # Save results without truncated field
                for j, (item_id, miwv_score, idx) in enumerate(zip(buffer_ids, miwv_scores, buffer_indices)):
                    most_similar_idx = int(self.most_similar_indices[idx])
                    most_similar_id = all_data[most_similar_idx].get("id", most_similar_idx)
                    results.append({
                        "id": item_id,
                        "score": miwv_score,
                        "most_similar_idx": most_similar_idx,
                        "most_similar_id": most_similar_id
                    })
                
                # Update progress bar
                pbar.update(len(buffer_items))
                
                # Clear buffer
                buffer_items.clear()
                buffer_indices.clear()
                buffer_ids.clear()
        
        pbar.close()
        
        return results

