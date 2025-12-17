# GraNdScorer

## Overview

The **GraNd Scorer** (Gradient Normed) is a model-based evaluation tool designed to measure the importance and informativeness of individual training examples for supervised fine-tuning (SFT) data. Introduced in the paper [Paul et al., 2021](https://proceedings.neurips.cc/paper/2021/hash/ac56f8fe9eea3e4a365f29f0f1957c55-Abstract.html), this method identifies valuable training examples by computing gradient norms early in the training process. The core insight is that examples producing larger gradient norms during early training phases tend to be more informative and crucial for model generalization.

## Metric Definition:

* **Definition:** The GraNd score is computed as the L2 norm of all parameter gradients after a single forward-backward pass on a data sample:
  
  ```
  GraNd(x) = ||∇θ L(x; θ)||₂
  ```
  
  where `L(x; θ)` is the loss for example `x` with model parameters `θ`, and `||·||₂` denotes the L2 norm.

* **Explanation:** This metric quantifies how much a single training example would change the model's parameters if used for training.
  
  * A **higher GraNd score** indicates that the example produces a large gradient, suggesting it contains **informative or challenging content** that can significantly improve the model.
  * A **lower GraNd score** suggests the example is **easy or redundant** relative to the current model state, contributing less to model improvement.

## YAML Configuration

```yaml
name: GraNdScorer
model: Qwen/Qwen3-8B
max_length: 2048
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"GraNdScorer"` | Identifier for the scorer |
| `model` | string | `"Qwen/Qwen3-8B"` | Path to local model or HuggingFace model name for computing gradients. Can be any causal language model compatible with HuggingFace `AutoModelForCausalLM`. Falls back to `gpt2` if the specified model fails to load |
| `max_length` | integer | `2048` | Maximum sequence length for tokenization. Controls the maximum number of tokens processed per example. Longer sequences may provide more accurate scores but require more memory |

## Underlying Model

The scorer uses causal language models (CLMs) to compute gradient norms. By default, it uses [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), but can work with any compatible language model.

The model is used in training mode to enable gradient computation, but no actual parameter updates are performed.

## Scoring Process

The GraNd scoring process follows these steps for each SFT data sample:

1. **Text Preparation**: Concatenate the `instruction`, `input` (if present), and `output` fields to form the complete text sequence.

2. **Tokenization**: 
   - Tokenize the instruction portion separately to determine its length
   - Tokenize the full text (instruction + output) for model input
   - Apply truncation at `max_length` if necessary

3. **Label Masking**: Create labels where:
   - Instruction tokens are set to `-100` (ignored in loss computation)
   - Output tokens retain their original token IDs (used for loss computation)
   - This ensures gradients are computed only with respect to the output/response generation

4. **Gradient Computation**:
   - Set model to training mode and zero existing gradients
   - Perform forward pass to compute the loss on output tokens
   - Perform backward pass to compute parameter gradients

5. **Gradient Norm Calculation**: Compute the L2 norm across all parameter gradients:
   ```
   gradient_norm = sqrt(Σᵢ ||∇θᵢ||₂²)
   ```
   where the sum is over all model parameters with computed gradients

6. **Cleanup**: Zero gradients and return model to evaluation mode

## Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 245.7382
}
```

- `id`: The unique identifier of the data sample (from the input data's `id` field)
- `score`: The computed GraNd score (gradient L2 norm) as a floating-point number. Higher values indicate more informative/important examples. Values are non-negative (L2 norms are always ≥ 0). Magnitude depends on model size and architecture

## Citation

```bibtex
@article{paul2021deep,
  title={Deep learning on a data diet: Finding important examples early in training},
  author={Paul, Mansheej and Ganguli, Surya and Dziugaite, Gintare Karolina},
  journal={Advances in neural information processing systems},
  volume={34},
  pages={20596--20607},
  year={2021}
}
```
