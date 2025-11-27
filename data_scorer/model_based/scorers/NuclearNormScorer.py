import torch
from .base_scorer import BaseScorer
import json
from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from .utils import get_total_lines
from tqdm import tqdm


class NuclearNormScorer(BaseScorer):
    def _validate_config(self):
        if "model" not in self.config:
            print(
                "Warning: No local model specified in config. Downloading the remote huggingface model.")
            self.config['model'] = 'Qwen/Qwen3-8B'


        if "max_length" not in self.config:
            print("Warning: No max_length specified, use default value of 2048.")
            self.config["max_length"] = 2048
        else:
            print(f"Using specified max_length: {self.config['max_length']}.")
            self.max_length = self.config["max_length"]

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
            self.config["num_layers"] = 1  # Default to 1, will use last layer when total_layers is unknown
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
            # If only num_layers is specified without start_layer_index, treat as invalid
            elif start_idx is None and num_layers is not None and num_layers != 1:
                print(f"Warning: num_layers specified but start_layer_index not specified. Using last layer instead.")
                # Keep num_layers=1, will use last layer when start_layer_index is None
        
        print("Setting up NuclearNormScorer successfully")

    def _get_total_layers(self):
        """Get total number of transformer layers"""
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
            # GPT-2, Qwen architectures
            layers = model.transformer.h
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            # LLaMA, Qwen architectures
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
        """Get attention parameters (Q, K, V, O) from specified transformer layer
        
        Args:
            layer: transformer layer object
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

    def _compute_nuclear_norm(self, grad_matrix):
        """Compute nuclear norm of matrix (sum of singular values)"""
        if grad_matrix is None:
            return 0.0
        
        # Convert gradient matrix to 2D
        if grad_matrix.dim() > 2:
            grad_matrix = grad_matrix.view(grad_matrix.size(0), -1)
        
        # Perform singular value decomposition
        try:
            # Use torch.linalg.svd (torch.svd is deprecated)
            U, S, Vh = torch.linalg.svd(grad_matrix, full_matrices=False)
            # Nuclear norm = sum of all singular values
            nuclear_norm = torch.sum(S).item()
            return nuclear_norm
        except Exception as e:
            print(f"Warning: SVD computation failed ({e}), returning 0.0")
            return 0.0

    def score_item(self, data_item: Dict) -> Dict[str, float]:
        """Compute nuclear norm scores for a single data point, returning Q, K, V, O values
        
        Returns:
            Dict containing four key-value pairs: 'Q_NuclearNorm', 'K_NuclearNorm', 'V_NuclearNorm', 'O_NuclearNorm'
        """
        instr = data_item["instruction"].strip()
        input = data_item.get("input", "").strip()
        if input:
            instr = instr + "\n" + input
        outp = data_item["output"].strip()
        
        # Build input text: instruction + output
        text = instr + " " + outp
        
        # Tokenize without truncation first to check original length
        full_encodings = self.tokenizer(
            text,
            padding=False,
            truncation=False,
            return_tensors="pt"
        )
        original_length = full_encodings["input_ids"].size(1)
        
        # Encode input with truncation
        encodings = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.config["max_length"],
            return_tensors="pt"
        ).to(self.device)

        input_ids = encodings["input_ids"]
        actual_length = input_ids.size(1)
        
        # Check if truncation occurred and issue warning
        if original_length > self.config["max_length"]:
            item_id = data_item.get("id", "unknown")
            print(f"Warning: Data item (id={item_id}) was truncated from {original_length} to {self.config['max_length']} tokens")
        
        # Set labels as input_ids (for loss computation)
        # Only compute loss on output part, mask out instruction part
        labels = input_ids.clone()
        
        # Encode instruction part to get its length
        instr_encodings = self.tokenizer(
            instr + " ",  # Include separator
            padding=False,
            truncation=True,
            max_length=self.config["max_length"],
            return_tensors="pt"
        )
        instr_length = instr_encodings["input_ids"].size(1)
        
        # Set instruction part labels to -100 (ignore this part in loss computation)
        labels[:, :instr_length] = -100

        # Set model to training mode to compute gradients
        self.model.train()
        
        # Zero out gradients
        self.model.zero_grad()
        
        # Forward pass
        outputs = self.model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        
        # Backward pass to compute gradients
        loss.backward()
        
        # Get all layers
        try:
            all_layers = self._get_all_layers()
        except Exception as e:
            print(f"Warning: Cannot get transformer layers ({e}), returning zero values")
            self.model.zero_grad()
            self.model.eval()
            return {'Q_NuclearNorm': 0.0, 'K_NuclearNorm': 0.0, 'V_NuclearNorm': 0.0, 'O_NuclearNorm': 0.0}
        
        # Determine which layers to compute
        start_idx = self.config.get("start_layer_index")
        num_layers = self.config.get("num_layers", 1)  # Default to 1
        
        # If start_layer_index is not specified or total_layers is unknown, only compute last layer
        if start_idx is None or self.total_layers is None:
            target_layers = [all_layers[-1]]  # Only compute last layer
        else:
            # Compute specified range of layers
            end_idx = start_idx + num_layers
            target_layers = all_layers[start_idx:end_idx]
        
        # Store Q, K, V, O nuclear norms for all layers
        q_norms = []
        k_norms = []
        v_norms = []
        o_norms = []
        
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
                        q_norm = self._compute_nuclear_norm(q_grad)
                        q_norms.append(q_norm)
                
                # K
                if 'K' in attention_params:
                    k_param = attention_params['K']
                    if k_param.weight.grad is not None:
                        k_grad = k_param.weight.grad
                        k_norm = self._compute_nuclear_norm(k_grad)
                        k_norms.append(k_norm)
                
                # V
                if 'V' in attention_params:
                    v_param = attention_params['V']
                    if v_param.weight.grad is not None:
                        v_grad = v_param.weight.grad
                        v_norm = self._compute_nuclear_norm(v_grad)
                        v_norms.append(v_norm)
                
                # O
                if 'O' in attention_params:
                    o_param = attention_params['O']
                    if o_param.weight.grad is not None:
                        o_grad = o_param.weight.grad
                        o_norm = self._compute_nuclear_norm(o_grad)
                        o_norms.append(o_norm)
            
            # Handle GPT-2 style: c_attn contains Q, K, V
            elif 'c_attn' in attention_params:
                c_attn = attention_params['c_attn']
                if c_attn.weight.grad is not None:
                    grad_matrix = c_attn.weight.grad
                    # c_attn weight shape is [3 * embed_dim, embed_dim]
                    # Need to split into Q, K, V parts along the first dimension
                    total_dim = grad_matrix.size(0)
                    embed_dim = total_dim // 3
                    q_grad = grad_matrix[:embed_dim, :]
                    k_grad = grad_matrix[embed_dim:2*embed_dim, :]
                    v_grad = grad_matrix[2*embed_dim:3*embed_dim, :]
                    
                    q_norm = self._compute_nuclear_norm(q_grad)
                    k_norm = self._compute_nuclear_norm(k_grad)
                    v_norm = self._compute_nuclear_norm(v_grad)
                    
                    q_norms.append(q_norm)
                    k_norms.append(k_norm)
                    v_norms.append(v_norm)
                
                # O parameter (c_proj)
                if 'O' in attention_params:
                    o_param = attention_params['O']
                    if o_param.weight.grad is not None:
                        o_grad = o_param.weight.grad
                        o_norm = self._compute_nuclear_norm(o_grad)
                        o_norms.append(o_norm)
        
        # Compute average values
        q_avg = sum(q_norms) / len(q_norms) if len(q_norms) > 0 else 0.0
        k_avg = sum(k_norms) / len(k_norms) if len(k_norms) > 0 else 0.0
        v_avg = sum(v_norms) / len(v_norms) if len(v_norms) > 0 else 0.0
        o_avg = sum(o_norms) / len(o_norms) if len(o_norms) > 0 else 0.0
        
        # Zero out gradients and set back to evaluation mode
        self.model.zero_grad()
        self.model.eval()
        
        return {
            'Q_NuclearNorm': float(q_avg),
            'K_NuclearNorm': float(k_avg),
            'V_NuclearNorm': float(v_avg),
            'O_NuclearNorm': float(o_avg)
        }

    def evaluate(self, dataset: str) -> List[Dict]:
        """Evaluate entire dataset, computing nuclear norm scores for each data point"""
        num_lines = get_total_lines(dataset)
        results: List[Dict] = []

        with open(dataset, 'r', encoding='utf-8') as f:
            pbar = tqdm(total=num_lines, desc=self.config.get(
                'name', 'NuclearNormScorer'))
            for line in f:
                item = json.loads(line.strip())
                item_id = item.get("id", "")
                
                # Compute nuclear norm scores, returning Q, K, V, O values
                nuclear_norm_scores = self.score_item(item)
                
                result = {
                    "id": item_id,
                    **nuclear_norm_scores  # Expand dict, contains Q_NuclearNorm, K_NuclearNorm, V_NuclearNorm, O_NuclearNorm
                }
                results.append(result)
                pbar.update(1)
            pbar.close()

        return results

