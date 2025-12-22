import torch
from .base_scorer import BaseScorer
import json
from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from .utils import get_total_lines
from tqdm import tqdm


class EffectiveRankScorer(BaseScorer):
    def _validate_config(self):
        if "model" not in self.config:
            print(
                "Warning: No local model specified in config. Downloading the remote huggingface model.")
            self.config['model'] = 'Qwen/Qwen3-8B'


        if "max_length" not in self.config:
            print("Warning: No max_length specified, use default value of 2048.")
            self.config["max_length"] = 2048


        # Validate start_layer_index and num_layers parameters
        if "start_layer_index" not in self.config:
            print("Warning: No start_layer_index specified, will use the last layer by default.")
            self.config["start_layer_index"] = None
        else:
            start_idx = self.config["start_layer_index"]
            if start_idx is not None:
                if not isinstance(start_idx, int) or start_idx < 0:
                    print(f"Warning: Invalid start_layer_index '{start_idx}', must be a non-negative integer. Using last layer instead.")
                    self.config["start_layer_index"] = None
                else:
                    print(f"Using specified start_layer_index: {self.config['start_layer_index']}.")
            else:
                print("Using last layer (start_layer_index=None).")

        if "num_layers" not in self.config:
            print("Warning: No num_layers specified, will use default value of 1.")
            self.config["num_layers"] = 1
        else:
            num_layers = self.config["num_layers"]
            if num_layers is not None:
                if not isinstance(num_layers, int) or num_layers <= 0:
                    print(f"Warning: Invalid num_layers '{num_layers}', must be a positive integer. Using default value of 1 instead.")
                    self.config["num_layers"] = 1
                else:
                    print(f"Using specified num_layers: {self.config['num_layers']}.")
            else:
                print("Warning: num_layers is None, will use default value of 1.")
                self.config["num_layers"] = 1

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
        
        # Get total number of layers
        self.total_layers = self._get_total_layers()
        if self.total_layers is None:
            print("Warning: Cannot determine total layers, will use last layer by default.")
            self.config["start_layer_index"] = None
            self.config["num_layers"] = 1  # num_layers defaults to 1, but will use the last layer since total layers is unknown
        else:
            print(f"Total transformer layers: {self.total_layers}")
            # Validate if start_layer_index and num_layers are within valid range
            start_idx = self.config.get("start_layer_index")
            num_layers = self.config.get("num_layers", 1)  # Default to 1
            
            # If start_layer_index is specified, validate range
            if start_idx is not None:
                if start_idx >= self.total_layers:
                    print(f"Warning: start_layer_index {start_idx} is out of range (0-{self.total_layers-1}). Using last layer instead.")
                    self.config["start_layer_index"] = None
                elif start_idx + num_layers > self.total_layers:
                    print(f"Warning: start_layer_index {start_idx} + num_layers {num_layers} exceeds total layers {self.total_layers}. Using last layer instead.")
                    self.config["start_layer_index"] = None
            # If only num_layers is specified but not start_layer_index, treat as invalid
            elif start_idx is None and num_layers is not None and num_layers != 1:
                print(f"Warning: num_layers specified but start_layer_index not specified. Using last layer instead.")
                # Keep num_layers=1, but will use the last layer when start_layer_index is None
        
        print("Setting up EffectiveRankScorer successfully")

    def _get_total_layers(self):
        """Get the total number of transformer layers"""
        model = self.model
        
        # Try different model architectures
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            return len(model.transformer.h)
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            return len(model.model.layers)
        elif hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
            return len(model.gpt_neox.layers)
        else:
            # Try to find layers directly
            for attr_name in ['layers', 'h', 'blocks']:
                if hasattr(model, attr_name):
                    layers = getattr(model, attr_name)
                    if len(layers) > 0:
                        return len(layers)
        return None

    def _get_all_layers(self):
        """Get all transformer layers"""
        model = self.model
        layers = None
        
        # Try different model architectures
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            # GPT-2, Qwen, etc. architectures
            layers = model.transformer.h
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            # LLaMA, Qwen, etc. architectures
            layers = model.model.layers
        elif hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
            # GPT-NeoX architecture
            layers = model.gpt_neox.layers
        else:
            # Try to find layers directly
            for attr_name in ['layers', 'h', 'blocks']:
                if hasattr(model, attr_name):
                    layers = getattr(model, attr_name)
                    break
        
        if layers is None or len(layers) == 0:
            raise ValueError("Cannot find transformer layers")
        
        return layers

    def _get_layer_attention_params(self, layer):
        """Get attention parameters (Q, K, V, O) of a specific transformer layer
        
        Args:
            layer: Transformer layer object
        """
        attention_module = None
        
        # Find attention module
        if hasattr(layer, 'attn'):
            attention_module = layer.attn
        elif hasattr(layer, 'attention'):
            attention_module = layer.attention
        elif hasattr(layer, 'self_attn'):
            attention_module = layer.self_attn
        else:
            raise ValueError("Cannot find attention module")
        
        # Extract Q, K, V, O parameters
        params = {}
        
        # Try different parameter naming conventions
        if hasattr(attention_module, 'q_proj'):
            params['Q'] = attention_module.q_proj
            params['K'] = attention_module.k_proj
            params['V'] = attention_module.v_proj
            if hasattr(attention_module, 'o_proj'):
                params['O'] = attention_module.o_proj
            elif hasattr(attention_module, 'out_proj'):
                params['O'] = attention_module.out_proj
        elif hasattr(attention_module, 'c_attn'):
            # GPT-2 style: c_attn contains Q, K, V, and c_proj is O
            params['c_attn'] = attention_module.c_attn
            params['O'] = attention_module.c_proj
        else:
            raise ValueError("Cannot find Q, K, V, O parameters")
        
        return params

    def _compute_effective_rank(self, grad_matrix):
        """Compute the Effective Rank of a matrix
        
        Effective Rank computation steps:
        1. Perform singular value decomposition on the gradient matrix
        2. Normalize singular values to form a probability distribution
        3. Calculate Shannon entropy (using natural logarithm)
        4. Take exp to get Effective Rank
        
        Args:
            grad_matrix: Gradient matrix
            
        Returns:
            Effective Rank value
        """
        if grad_matrix is None:
            return 0.0
        
        # Convert gradient matrix to 2D
        if grad_matrix.dim() > 2:
            grad_matrix = grad_matrix.view(grad_matrix.size(0), -1)
        
        # Perform singular value decomposition
        try:
            # Use torch.linalg.svd (torch.svd is deprecated)
            U, S, Vh = torch.linalg.svd(grad_matrix, full_matrices=False)
            
            # Filter out very small singular values (avoid numerical instability)
            S = S[S > 1e-10]
            
            if len(S) == 0:
                return 0.0
            
            # Normalize singular values to form a probability distribution
            S_sum = torch.sum(S)
            if S_sum <= 0:
                return 0.0
            
            p = S / S_sum  # Probability distribution
            
            # Calculate Shannon entropy (using natural logarithm)
            # H = -sum(p_i * ln(p_i))
            # Avoid ln(0) by only computing non-zero probability values
            p = p[p > 1e-10]  # Filter out very small probability values
            if len(p) == 0:
                return 0.0
            
            # Renormalize to ensure probability distribution sums to 1
            p = p / torch.sum(p)
            
            entropy = -torch.sum(p * torch.log(p)).item()
            
            # Effective Rank = exp(H)
            effective_rank = torch.exp(torch.tensor(entropy, device=self.device)).item()
            
            return effective_rank
        except Exception as e:
            print(f"Warning: Effective Rank computation failed ({e}), returning 0.0")
            return 0.0

    def score_item(self, data_item: Dict) -> Dict[str, float]:
        """Calculate Effective Rank scores for a single data point, returning Q, K, V, O values
        
        Returns:
            Dict containing four key-value pairs: 'Q_EffectiveRank', 'K_EffectiveRank', 'V_EffectiveRank', 'O_EffectiveRank'
        """
        instr = data_item["instruction"].strip()
        input = data_item.get("input", "").strip()
        if input:
            instr = instr + "\n" + input
        outp = data_item["output"].strip()

        # Build input text: instruction(+input) + output
        # NOTE: We only compute loss/gradients on the output part (SFT-style) by masking the prompt tokens in labels.
        # Use "\n" as the separator to be consistent with typical instruction-tuning formatting.
        prompt_text = instr + "\n"
        text = prompt_text + outp

        # Check original text length before truncation
        pre_tokenized = self.tokenizer(text, add_special_tokens=True)
        original_length = len(pre_tokenized["input_ids"])
        if original_length > self.config["max_length"]:
            print(f"Warning: Data exceeds max_length ({original_length} > {self.config['max_length']}). Text will be truncated.")

        # Encode prompt only (to get its length, including the separator) for label masking
        prompt_encodings = self.tokenizer(
            prompt_text,
            padding=False,
            truncation=True,
            max_length=self.config["max_length"],
            return_tensors="pt"
        ).to(self.device)

        # Encode full text (prompt + output)
        encodings = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.config["max_length"],
            return_tensors="pt"
        ).to(self.device)

        input_ids = encodings["input_ids"]
        attention_mask = encodings.get("attention_mask", None)
        
        # Set labels: ignore prompt part, only compute loss on output part
        labels = input_ids.clone()
        prompt_length = prompt_encodings["input_ids"].shape[1]
        labels[:, :prompt_length] = -100  # -100 is ignored in loss computation
        # Also ignore padding tokens if present (important for future batching)
        if attention_mask is not None:
            labels = labels.masked_fill(attention_mask == 0, -100)

        # Set model to training mode to compute gradients
        self.model.train()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Forward pass
        outputs = self.model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        
        # Check if loss is valid
        if torch.isnan(loss) or torch.isinf(loss):
            print("Warning: Loss is NaN or Inf, skipping this sample")
            # Clean up memory
            del loss, outputs, encodings, input_ids, labels
            self.model.zero_grad()
            self.model.eval()
            if self.device == "cuda":
                torch.cuda.empty_cache()
            return {'Q_EffectiveRank': 0.0, 'K_EffectiveRank': 0.0, 'V_EffectiveRank': 0.0, 'O_EffectiveRank': 0.0}
        
        # Backward pass to compute gradients
        loss.backward()
        
        # Get all layers
        try:
            all_layers = self._get_all_layers()
        except Exception as e:
            print(f"Warning: Cannot get transformer layers ({e}), returning zero values")
            # Clean up memory
            del loss, outputs, encodings, input_ids, labels
            self.model.zero_grad()
            self.model.eval()
            if self.device == "cuda":
                torch.cuda.empty_cache()
            return {'Q_EffectiveRank': 0.0, 'K_EffectiveRank': 0.0, 'V_EffectiveRank': 0.0, 'O_EffectiveRank': 0.0}
        
        # Determine layers to compute
        start_idx = self.config.get("start_layer_index")
        num_layers = self.config.get("num_layers", 1)  # Default to 1
        
        # If start_layer_index is not specified or total layers is unknown, only compute last layer
        if start_idx is None or self.total_layers is None:
            target_layers = [all_layers[-1]]  # Only compute last layer
        else:
            # Compute specified range of layers
            end_idx = start_idx + num_layers
            target_layers = all_layers[start_idx:end_idx]
        
        # Store Q, K, V, O Effective Rank for all layers
        q_ranks = []
        k_ranks = []
        v_ranks = []
        o_ranks = []
        
        # Iterate through target layers
        for layer in target_layers:
            try:
                attention_params = self._get_layer_attention_params(layer)
            except Exception as e:
                print(f"Warning: Cannot get attention parameters for a layer ({e}), skipping this layer")
                continue
            
            # Handle standard format: Q, K, V, O stored separately
            if 'Q' in attention_params and 'K' in attention_params and 'V' in attention_params:
                # Q
                if 'Q' in attention_params:
                    q_param = attention_params['Q']
                    if q_param.weight.grad is not None:
                        q_grad = q_param.weight.grad
                        q_rank = self._compute_effective_rank(q_grad)
                        q_ranks.append(q_rank)
                
                # K
                if 'K' in attention_params:
                    k_param = attention_params['K']
                    if k_param.weight.grad is not None:
                        k_grad = k_param.weight.grad
                        k_rank = self._compute_effective_rank(k_grad)
                        k_ranks.append(k_rank)
                
                # V
                if 'V' in attention_params:
                    v_param = attention_params['V']
                    if v_param.weight.grad is not None:
                        v_grad = v_param.weight.grad
                        v_rank = self._compute_effective_rank(v_grad)
                        v_ranks.append(v_rank)
                
                # O
                if 'O' in attention_params:
                    o_param = attention_params['O']
                    if o_param.weight.grad is not None:
                        o_grad = o_param.weight.grad
                        o_rank = self._compute_effective_rank(o_grad)
                        o_ranks.append(o_rank)
            
            # Handle GPT-2 style: c_attn contains Q, K, V
            elif 'c_attn' in attention_params:
                c_attn = attention_params['c_attn']
                if c_attn.weight.grad is not None:
                    grad_matrix = c_attn.weight.grad
                    # c_attn weight shape is [3 * embed_dim, embed_dim]
                    # Need to split into Q, K, V parts (split along first dimension)
                    embed_dim = grad_matrix.size(1)  # Use size(1) to get embed_dim
                    q_grad = grad_matrix[:embed_dim, :]
                    k_grad = grad_matrix[embed_dim:2*embed_dim, :]
                    v_grad = grad_matrix[2*embed_dim:3*embed_dim, :]
                    
                    q_rank = self._compute_effective_rank(q_grad)
                    k_rank = self._compute_effective_rank(k_grad)
                    v_rank = self._compute_effective_rank(v_grad)
                    
                    q_ranks.append(q_rank)
                    k_ranks.append(k_rank)
                    v_ranks.append(v_rank)
                
                # O parameter (c_proj)
                if 'O' in attention_params:
                    o_param = attention_params['O']
                    if o_param.weight.grad is not None:
                        o_grad = o_param.weight.grad
                        o_rank = self._compute_effective_rank(o_grad)
                        o_ranks.append(o_rank)
        
        # Calculate average values
        q_avg = sum(q_ranks) / len(q_ranks) if len(q_ranks) > 0 else 0.0
        k_avg = sum(k_ranks) / len(k_ranks) if len(k_ranks) > 0 else 0.0
        v_avg = sum(v_ranks) / len(v_ranks) if len(v_ranks) > 0 else 0.0
        o_avg = sum(o_ranks) / len(o_ranks) if len(o_ranks) > 0 else 0.0
        
        # Clean up memory to avoid GPU memory leaks
        del loss, outputs, encodings, input_ids, labels
        self.model.zero_grad()
        self.model.eval()
        
        # If using GPU, empty cache
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        return {
            'Q_EffectiveRank': float(q_avg),
            'K_EffectiveRank': float(k_avg),
            'V_EffectiveRank': float(v_avg),
            'O_EffectiveRank': float(o_avg)
        }

    def evaluate(self, dataset: str) -> List[Dict]:
        """Evaluate the entire dataset and calculate Effective Rank scores for each data point"""
        num_lines = get_total_lines(dataset)
        results: List[Dict] = []

        with open(dataset, 'r', encoding='utf-8') as f:
            pbar = tqdm(total=num_lines, desc=self.config.get(
                'name', 'EffectiveRankScorer'))
            for line in f:
                item = json.loads(line.strip())
                item_id = item.get("id", "")
                
                # Calculate Effective Rank scores, returns Q, K, V, O values
                effective_rank_scores = self.score_item(item)
                
                result = {
                    "id": item_id,
                    **effective_rank_scores  # Expand dict to include Q_EffectiveRank, K_EffectiveRank, V_EffectiveRank, O_EffectiveRank
                }
                results.append(result)
                pbar.update(1)
            pbar.close()

        return results

