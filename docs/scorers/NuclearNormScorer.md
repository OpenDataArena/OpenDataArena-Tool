# NuclearNormScorer

## Overview

The **Nuclear Norm Scorer** is a gradient-based evaluation tool designed to assess the quality of instruction-following and reasoning data through spectral analysis of layer-wise gradients. This scorer computes the nuclear norm (sum of singular values) of gradient matrices derived from attention layer parameters (Q, K, V, O) during model fine-tuning, providing insights into gradient stability and data quality.

Based on the research paper ["How Instruction and Reasoning Data shape Post-Training: Data Quality through the Lens of Layer-wise Gradients"](https://arxiv.org/abs/2504.10766), this method reveals that higher-quality data typically exhibits lower nuclear norms, indicating more stable and focused gradient updates. The nuclear norm serves as a complementary metric to effective rank, capturing the magnitude-weighted diversity of gradient singular values.

## Metric Definition:

* **Definition:** 
  
  The Nuclear Norm is computed through singular value decomposition (SVD) of gradient matrices:
  
  ```
  Nuclear_Norm = Σ(σ_i)
  
  where σ_i are singular values from SVD
  ```
  
  Where:
  - `σ_i` represents the i-th singular value from SVD of the gradient matrix
  - The nuclear norm is the sum of all singular values
  - Also known as the trace norm or Schatten 1-norm

* **Explanation:** Nuclear Norm quantifies the magnitude and complexity of gradient updates:
  
  * A **lower Nuclear Norm** indicates that gradients have **smaller magnitudes** and are **more concentrated**, suggesting **higher data quality** and more stable learning dynamics. This implies the training sample induces focused, efficient gradient updates.
  * A **higher Nuclear Norm** suggests that gradients have **larger magnitudes** or are **more dispersed** across dimensions, potentially indicating **noisier training signals** or less efficient learning patterns.
  
  The metric is computed separately for Query (Q), Key (K), Value (V), and Output (O) projection matrices in attention layers. Compared to effective rank which measures dimensionality, nuclear norm captures the overall scale of gradient updates, providing complementary insights into training dynamics.

## YAML Configuration

```yaml
name: NuclearNormScorer
model: Qwen/Qwen3-8B
max_length: 2048
start_layer_index: 16
num_layers: 4
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"NuclearNormScorer"` | Scorer identifier, must be set to `NuclearNormScorer` |
| `model` | string | `"Qwen/Qwen3-8B"` | Path to the language model, either a local directory or a HuggingFace model ID. The model is used to compute gradients during forward and backward passes. If loading fails, defaults to `gpt2` as fallback. |
| `max_length` | integer | `2048` | Maximum token length for input sequences. Text exceeding this length will be truncated with a warning message. Recommended values: 1024-4096 depending on GPU memory and model context length. |
| `start_layer_index` | integer/None | `None` | The starting index of transformer layers to analyze (0-indexed). If `None`, only the last layer is analyzed. If specified, layers from `start_layer_index` to `start_layer_index + num_layers - 1` are analyzed. Must be within `[0, total_layers - 1]`. |
| `num_layers` | integer | `1` | Number of consecutive layers to analyze starting from `start_layer_index`. Must be a positive integer. If `start_layer_index` is `None`, this parameter is ignored. The final nuclear norm scores are averaged across all specified layers. |

**Note:** The combination of `start_layer_index` and `num_layers` allows flexible analysis of specific layer ranges. For example, to analyze the last 4 layers of a 32-layer model, set `start_layer_index: 28` and `num_layers: 4`.

## Underlying Model

The Nuclear Norm Scorer requires a **causal language model** for gradient computation. By default, it uses:

- **Primary model:** `Qwen/Qwen3-8B` - A powerful instruction-tuned language model
- **Fallback model:** `gpt2` - Used if the specified model fails to load

The scorer is architecture-agnostic and supports various transformer models including:
- GPT-style models (GPT-2, GPT-Neo, etc.)
- LLaMA family models
- Qwen family models
- GPT-NeoX architecture models

The model is used in training mode to compute gradients but **is not updated**—gradients are computed solely for analysis purposes. The scorer automatically detects the model architecture and locates attention layer parameters (Q, K, V, O projections) accordingly.

**GPU Recommendation:** This scorer requires significant computational resources. A GPU with at least 16GB memory is recommended for models like Qwen3-8B. Larger models may require multiple GPUs or gradient checkpointing.

## Scoring Process

1. **Data Preparation:**
   - Concatenate the `instruction`, `input` (if present), and `output` fields into a single text sequence
   - Text format: `instruction + " " + output` (with input appended to instruction if provided)
   - Check original text length before truncation and display detailed warning if truncation occurs, including item ID and token counts

2. **Tokenization:**
   - Tokenize the concatenated text using the model's tokenizer
   - Apply padding and truncation to `max_length`
   - Set labels equal to input_ids for language modeling loss computation
   - **Label Masking:** Set instruction portion labels to `-100` to compute loss only on the output portion, focusing gradient analysis on response generation

3. **Forward Pass:**
   - Set the model to training mode
   - Zero out any existing gradients
   - Perform forward pass to compute the language modeling loss (only on output tokens due to label masking)

4. **Backward Pass:**
   - Compute gradients via backpropagation: `loss.backward()`
   - Gradients are accumulated in the `.grad` attribute of each parameter

5. **Layer Selection:**
   - Retrieve all transformer layers from the model
   - Determine target layers based on `start_layer_index` and `num_layers`:
     - If `start_layer_index` is `None`: Use only the **last layer**
     - Otherwise: Use layers `[start_layer_index : start_layer_index + num_layers]`

6. **Gradient Extraction:**
   - For each target layer, extract attention module parameters:
     - **Standard format:** Separate Q, K, V, O projection weights
     - **GPT-2 format:** Combined QKV in `c_attn` (split by dimension), separate O in `c_proj`
   - Retrieve gradient matrices from `.weight.grad` of each projection

7. **Nuclear Norm Computation (for each Q, K, V, O gradient matrix):**
   - **Step 7.1:** Reshape gradient matrix to 2D if necessary
   - **Step 7.2:** Perform Singular Value Decomposition (SVD): `U, S, Vh = torch.linalg.svd(grad_matrix)`
   - **Step 7.3:** Compute Nuclear Norm as the sum of all singular values: `Nuclear_Norm = sum(S)`
   - **Note:** Unlike effective rank which normalizes and computes entropy, nuclear norm directly sums the singular values, capturing both magnitude and diversity

8. **Aggregation:**
   - Collect nuclear norms for Q, K, V, O across all target layers
   - Compute average nuclear norm for each projection type
   - Return four averaged scores: Q_NuclearNorm, K_NuclearNorm, V_NuclearNorm, O_NuclearNorm

9. **Memory Cleanup:**
   - Clear gradients, delete temporary tensors
   - Set model back to evaluation mode
   - Empty CUDA cache if using GPU to prevent memory leaks (automatically handled)

## Output Format

For each input sample, the scorer returns:

```json
{
  "id": "sample_identifier",
  "Q_NuclearNorm": 245.67,
  "K_NuclearNorm": 198.34,
  "V_NuclearNorm": 289.12,
  "O_NuclearNorm": 223.45
}
```

- `id`: The unique identifier of the sample, extracted from the input data's `id` field
- `Q_NuclearNorm`: Nuclear norm of gradients from Query projection matrices, averaged across all specified layers. Lower values indicate more focused query attention updates.
- `K_NuclearNorm`: Nuclear norm of gradients from Key projection matrices, averaged across all specified layers. Reflects the magnitude of key representation updates.
- `V_NuclearNorm`: Nuclear norm of gradients from Value projection matrices, averaged across all specified layers. Indicates the scale of value transformation updates.
- `O_NuclearNorm`: Nuclear norm of gradients from Output projection matrices, averaged across all specified layers. Represents the overall magnitude of attention output transformations.

**Interpretation Guidelines:**

- **Higher-quality data** generally correlates with **lower nuclear norms** across all four projection types, indicating more stable and efficient gradient updates
- **Lower nuclear norms** suggest that gradients are concentrated and focused, reflecting better training stability
- **Higher nuclear norms** may indicate noisier or less efficient learning patterns
- **Zero values** indicate computation failures (missing gradients or SVD errors) and should be investigated
- **Relative comparisons** between samples are more meaningful than absolute values, as nuclear norms depend on model architecture, layer depth, and parameter dimensions
- **Complementary to Effective Rank:** While effective rank measures gradient dimensionality (higher is better for quality), nuclear norm measures gradient magnitude (lower is better for quality)

**Example Output:**

```json
[
  {
    "id": 1,
    "Q_NuclearNorm": 156.34,
    "K_NuclearNorm": 142.89,
    "V_NuclearNorm": 178.23,
    "O_NuclearNorm": 165.91
  },
  {
    "id": 2,
    "Q_NuclearNorm": 387.56,
    "K_NuclearNorm": 412.34,
    "V_NuclearNorm": 456.78,
    "O_NuclearNorm": 398.12
  }
]
```

## Citation

```bibtex
@article{li2025instruction,
  title={How instruction and reasoning data shape post-training: Data quality through the lens of layer-wise gradients},
  author={Li, Ming and Li, Yanhong and Li, Ziyue and Zhou, Tianyi},
  journal={arXiv preprint arXiv:2504.10766},
  year={2025}
}
```

