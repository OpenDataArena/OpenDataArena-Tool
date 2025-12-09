# EffectiveRankScorer

## Overview

The **Effective Rank Scorer** is a gradient-based evaluation tool designed to assess the quality of instruction-following and reasoning data through spectral analysis of layer-wise gradients. This scorer computes the effective rank of gradient matrices derived from attention layer parameters (Q, K, V, O) during model fine-tuning, providing insights into the richness and complexity of gradient structures induced by training data.

Based on the research paper ["How Instruction and Reasoning Data shape Post-Training: Data Quality through the Lens of Layer-wise Gradients"](https://arxiv.org/abs/2504.10766), this method reveals that higher-quality data typically exhibits higher effective ranks, indicating richer gradient structures and more complex learning patterns. The effective rank metric demonstrates better robustness and resolution than nuclear norm in capturing subtle quality differences between instruction and reasoning data.

## Metric Definition:

* **Definition:** 
  
  The Effective Rank is computed through singular value decomposition (SVD) of gradient matrices:
  
  ```
  Effective_Rank = exp(H)
  
  where H = -Σ(p_i * ln(p_i))  (Shannon entropy)
  
  and p_i = σ_i / Σ(σ_j)  (normalized singular values)
  ```
  
  Where:
  - `σ_i` represents the i-th singular value from SVD of the gradient matrix
  - `p_i` forms a probability distribution from normalized singular values
  - `H` is the Shannon entropy computed using natural logarithm
  - `Effective_Rank` is the exponential of the entropy

* **Explanation:** 
  
  Effective Rank quantifies the dimensionality and richness of the gradient space:
  
  * A **higher Effective Rank** indicates that the gradients span a more diverse set of directions in parameter space, suggesting that the training sample induces **richer gradient structures** and potentially **higher data quality**.
  * A **lower Effective Rank** suggests that gradients are concentrated in fewer dimensions, indicating **simpler learning patterns** and potentially **lower data complexity**.
  
  The metric is computed separately for Query (Q), Key (K), Value (V), and Output (O) projection matrices in attention layers, providing fine-grained insights into how different attention components respond to training data.

## YAML Configuration

```yaml
name: EffectiveRankScorer
model: Qwen/Qwen3-8B
max_length: 2048
start_layer_index: 16
num_layers: 4
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"EffectiveRankScorer"` | Identifier for the scorer |
| `model` | string | `"Qwen/Qwen3-8B"` | HuggingFace model path or local directory for the causal language model used to compute gradients |
| `max_length` | integer | `2048` | Maximum sequence length for tokenization and gradient computation |
| `start_layer_index` | integer | `None` | Starting index of transformer layers to analyze (0-indexed). If `None`, only the last layer is analyzed |
| `num_layers` | integer | `1` | Number of consecutive layers to analyze starting from `start_layer_index`. Scores are averaged across all specified layers |

## Underlying Model

The scorer uses causal language models from the HuggingFace ecosystem to compute gradients through backpropagation. By default, it uses **Qwen/Qwen3-8B**, but can be configured to use any autoregressive transformer model. The model is used in training mode to compute gradients but is not updated—gradients are computed solely for analysis purposes.

## Scoring Process

1. **Input Processing**: For each data sample, concatenate the `instruction`, `input` (if present), and `output` fields into a single text sequence

2. **Tokenization**: Tokenize the concatenated text using the model's tokenizer with padding and truncation to `max_length`

3. **Forward Pass**: Set model to training mode and compute the language modeling loss through forward propagation

4. **Backward Pass**: Compute gradients via backpropagation using `loss.backward()` to accumulate gradients in parameter `.grad` attributes

5. **Layer Selection**: Determine target layers based on `start_layer_index` and `num_layers` parameters (defaults to last layer only)

6. **Gradient Extraction**: For each target layer, extract gradient matrices from attention projection parameters (Q, K, V, O)

7. **Effective Rank Computation**: For each gradient matrix:
   - Perform Singular Value Decomposition (SVD)
   - Normalize singular values to form probability distribution
   - Compute Shannon entropy: `H = -Σ(p_i * ln(p_i))`
   - Calculate Effective Rank: `exp(H)`

8. **Aggregation**: Average effective ranks across all specified layers for each projection type (Q, K, V, O)

## Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "Q_EffectiveRank": 128.45,
  "K_EffectiveRank": 115.32,
  "V_EffectiveRank": 142.67,
  "O_EffectiveRank": 135.89
}
```

- `id`: Unique identifier of the sample
- `Q_EffectiveRank`: Effective rank of Query projection gradients, averaged across specified layers
- `K_EffectiveRank`: Effective rank of Key projection gradients, averaged across specified layers
- `V_EffectiveRank`: Effective rank of Value projection gradients, averaged across specified layers
- `O_EffectiveRank`: Effective rank of Output projection gradients, averaged across specified layers

## Citation

```bibtex
@article{li2025instruction,
  title={How instruction and reasoning data shape post-training: Data quality through the lens of layer-wise gradients},
  author={Li, Ming and Li, Yanhong and Li, Ziyue and Zhou, Tianyi},
  journal={arXiv preprint arXiv:2504.10766},
  year={2025}
}
```
