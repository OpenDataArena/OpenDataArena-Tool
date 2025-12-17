from .base_scorer import BaseScorer
from typing import Dict, List
import numpy as np
import os
import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.metrics.pairwise import cosine_distances
from tqdm import tqdm


class Task2VecScorer(BaseScorer):
    def _validate_config(self):
        """Validate configuration parameters"""
        # Validate model path
        if "model" not in self.config:
            self.config['model'] = 'openai-community/gpt2'
            print("Warning: No local model specified in config, using default 'openai-community/gpt2'.")
        else:
            # Check if local path exists
            if not os.path.exists(self.config["model"]):
                print(
                    f"Warning: Specified local model path '{self.config['model']}' does not exist. "
                    "Using default 'openai-community/gpt2'."
                )
                self.config['model'] = 'openai-community/gpt2'
            else:
                print(f"Using specified local model: {self.config['model']}")
        
        # Validate last_layer_only parameter
        if "last_layer_only" not in self.config:
            self.config['last_layer_only'] = False
            print("Warning: No last_layer_only specified, using default value of False.")
        else:
            if not isinstance(self.config["last_layer_only"], bool):
                print(
                    f"Warning: Invalid last_layer_only '{self.config['last_layer_only']}', "
                    f"using default value of False."
                )
                self.config['last_layer_only'] = False
            else:
                print(f"Using specified last_layer_only: {self.config['last_layer_only']}")
        
        # Validate max_length parameter
        if "max_length" not in self.config:
            self.config['max_length'] = 512
            print("Warning: No max_length specified, using default value of 512.")
        else:
            if not isinstance(self.config["max_length"], int) or self.config["max_length"] <= 0:
                print(
                    f"Warning: Invalid max_length '{self.config['max_length']}', "
                    f"using default value of 512."
                )
                self.config['max_length'] = 512
            else:
                print(f"Using specified max_length: {self.config['max_length']}")

    def _setup(self):
        """Initialize model and tokenizer"""
        print(f"Loading model from: {self.config['model']}")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        try:
            self.model = GPT2LMHeadModel.from_pretrained(self.config['model'])
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.config['model'])
            print(f"Model and tokenizer loaded successfully from {self.config['model']}")
        except Exception as e:
            print(f"Warning: Failed to load model from '{self.config['model']}': {e}")
            print("Falling back to remote model 'openai-community/gpt2'")
            self.model = GPT2LMHeadModel.from_pretrained('openai-community/gpt2')
            self.tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2')
        
        # Set padding token for GPT-2 (use eos_token as pad_token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"Set pad_token to eos_token: {self.tokenizer.eos_token}")
        
        # Also set model's pad_token_id
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            print(f"Set model pad_token_id to: {self.model.config.pad_token_id}")
        
        self.model.to(self.device)
        self.last_layer_only = self.config['last_layer_only']
        
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print("Setting up Task2VecScorer successfully")

    def compute_fim(self, text, last_layer_only=False, check_truncation=False):
        """Compute diagonal elements of Fisher Information Matrix (FIM) for a single text
        
        Args:
            text: A single text string
            last_layer_only: Whether to only consider the last layer parameters
            check_truncation: Whether to check and warn if text is truncated
            
        Returns:
            Tensor of FIM diagonal elements and truncation flag (is_truncated)
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Get parameters to compute
        if last_layer_only:
            # Only consider the last trainable parameter
            params = [p for p in self.model.parameters() if p.requires_grad][-1:]
            param_shapes = [(p.numel(), p.shape) for p in params]
        else:
            # Consider all trainable parameters
            params = [p for p in self.model.parameters() if p.requires_grad]
            param_shapes = [(p.numel(), p.shape) for p in params]

        total_params = sum(size for size, _ in param_shapes)

        # Encode text
        inputs = self.tokenizer(
            text, 
            return_tensors='pt', 
            truncation=True, 
            max_length=self.config['max_length']
        ).to(self.model.device)
        
        # Check if text was truncated
        is_truncated = False
        if check_truncation:
            # Tokenize without truncation to check original length
            full_tokens = self.tokenizer(text, return_tensors='pt', truncation=False)
            original_length = full_tokens['input_ids'].shape[1]
            truncated_length = inputs['input_ids'].shape[1]
            if original_length > truncated_length:
                is_truncated = True

        # Forward pass with gradient computation enabled
        with torch.enable_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # shape: [1, seq_len, vocab_size]

            # Get true target tokens (next token in the input sequence)
            target_token_ids = inputs.input_ids[:, 1:]  # shape: [1, seq_len-1]

            # Compute log probabilities
            log_probs = torch.log_softmax(logits[:, :-1], dim=-1)
            log_prob_targets = torch.gather(
                log_probs, 2, target_token_ids.unsqueeze(-1)
            ).squeeze(-1)  # shape: [1, seq_len-1]

            # Get sequence length
            seq_len = log_prob_targets.shape[1]
            
            if seq_len == 0:
                # If sequence length is 0, return zero vector
                return torch.zeros(total_params), is_truncated
            
            # Initialize FIM
            fim_diag = torch.zeros(total_params, device=self.model.device)
            
            # Compute gradient for each position
            for t in range(seq_len):
                # Compute gradient at current position
                grads = torch.autograd.grad(
                    outputs=log_prob_targets[0, t],
                    inputs=params,
                    retain_graph=(t < seq_len - 1),  # Last position doesn't need to retain computation graph
                    create_graph=False,
                    allow_unused=True
                )

                # Accumulate squared gradients
                offset = 0
                for param, grad in zip(params, grads):
                    if grad is not None:
                        grad_flat = grad.detach().flatten()
                        fim_diag[offset:offset + grad_flat.numel()] += grad_flat ** 2
                        offset += grad_flat.numel()

            # Average over sequence length
            fim_diag /= seq_len

        return fim_diag.detach().cpu(), is_truncated

    def score_item(self, data_item: Dict) -> float:
        """Task2VecScorer scores the entire dataset, not individual samples"""
        raise NotImplementedError(
            "Task2VecScorer computes a single score for the entire dataset. "
            "Use evaluate() method instead."
        )

    def evaluate(self, dataset) -> Dict:
        """Evaluate the entire dataset based on Task2Vec and compute average cosine distance
        
        Args:
            dataset: Dataset file path (jsonl format)
        
        Returns:
            Dictionary containing evaluation results and statistics
        """
        print(f"Loading dataset from: {dataset}")
        
        texts = []
        anomalous_indices = set()

        # Load and encode data
        with open(dataset, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                try:
                    data = json.loads(line.strip())
                    instruction = data["instruction"]
                    input_text = data.get("input", "")
                    response = data['output']
                    if input_text:
                        text = instruction + '\n' + input_text + '\n' + response
                    else:
                        text = instruction + '\n' + response
                    if text:
                        texts.append(text)
                    else:
                        print(f"Skipping incomplete record at line {idx}: {data}")
                        anomalous_indices.add(idx)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON line at {idx}: {line}")
                    anomalous_indices.add(idx)

        print(f"Loaded {len(texts)} valid samples from dataset")
        print(f"Skipped {len(anomalous_indices)} anomalous records")
        
        print(f"Computing Fisher Information Matrix for {len(texts)} samples...")
        print(f"Last layer only: {self.config['last_layer_only']}")
        
        # Compute FIM embeddings for each sample
        embeddings = []
        truncated_count = 0
        truncated_indices = []
        
        for idx, text in enumerate(tqdm(texts, desc="Processing samples")):
            fim_embedding, is_truncated = self.compute_fim(
                text, last_layer_only=self.config['last_layer_only'], check_truncation=True
            )
            embeddings.append(fim_embedding)
            
            if is_truncated:
                truncated_count += 1
                truncated_indices.append(idx)
        
        # Warn if any texts were truncated
        if truncated_count > 0:
            print(f"\nWarning: {truncated_count} out of {len(texts)} samples ({truncated_count/len(texts)*100:.2f}%) "
                  f"were truncated due to exceeding max_length={self.config['max_length']}")
            if truncated_count <= 10:
                print(f"Truncated sample indices: {truncated_indices}")
            else:
                print(f"First 10 truncated sample indices: {truncated_indices[:10]}")

        # Convert to numpy array and compute cosine distance matrix
        embeddings = torch.stack(embeddings).detach().cpu().numpy()
        print(f"Embeddings shape: {embeddings.shape}")
        
        print("Computing cosine distance matrix...")
        cosine_dist_matrix = cosine_distances(embeddings)
        
        # Compute average distance for each sample with respect to other samples
        n_samples = len(embeddings)
        sample_distances = []
        
        for i in range(n_samples):
            # Get distances between i-th sample and all other samples
            dists = cosine_dist_matrix[i]
            # Exclude self (diagonal element)
            dists = np.delete(dists, i)
            # Compute average distance
            avg_dist = np.mean(dists).item()
            sample_distances.append(avg_dist)
        
        # Compute average distance of all samples as final score
        score = np.mean(sample_distances).item() if sample_distances else 0.0
        
        print(f"Task2Vec Score: {score}")
        
        return {
            "score": score,
            "num_samples": len(texts),
            "num_anomalous": len(anomalous_indices),
            "num_truncated": truncated_count,
            "truncation_rate": truncated_count / len(texts) if len(texts) > 0 else 0.0,
            "last_layer_only": self.config['last_layer_only'],
            "embedding_dim": embeddings.shape[1] if len(embeddings.shape) > 1 else 0
        }