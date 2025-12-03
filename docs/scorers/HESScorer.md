# HESScorer

## Overview

The **High-Entropy Sum (HES) Scorer** is a training-free metric designed to evaluate the quality and complexity of Chain-of-Thought (CoT) reasoning samples. Proposed in the paper ["Unified Data Selection for LLM Reasoning"](https://openreview.net/pdf?id=heVn5cNfje), HES addresses a critical limitation in traditional metrics: they perform coarse-grained, global evaluation that treats all tokens equally, diluting the signal from truly critical reasoning steps.

Unlike metrics such as average entropy or perplexity that average over all tokens, HES focuses exclusively on **high-entropy forking tokens**—the key decision points in the reasoning process where the model faces multiple plausible paths. By summing only the entropy of the top 0.5% highest-entropy tokens, HES captures the genuine complexity and learning value of reasoning samples while filtering out predictable, trivial content.

## Metric Definition:

* **Definition:** 
  
  HES is calculated by summing the entropy values of the top *p* percentile (default *p* = 0.5%) of tokens with the highest entropy within the completion (reasoning) part of a sample:
  
  ```
  HES = Σ H(token_i) for token_i ∈ Top_p%(entropies)
  ```
  
  where `H(token_i)` is the Shannon entropy (in bits) of the token probability distribution at position *i*.

* **Explanation:** 
  
  * A **higher HES score** indicates greater diversity and complexity of reasoning patterns at critical forking points, suggesting **higher learning value** and more challenging reasoning paths.
  * A **lower HES score** suggests fewer critical decision points or more deterministic reasoning, indicating **simpler or lower-quality** reasoning samples.
  
  The key insight is that in long CoT reasoning, truly difficult-to-predict tokens are in the minority. By focusing on these high-entropy tokens rather than averaging across all tokens, HES effectively identifies samples with substantial reasoning complexity.

## YAML Configuration

```yaml
name: HESScorer
model: Qwen/Qwen2.5-7B-Instruct
percentile_cutoff: 0.005
batch_size: 8
max_length: 4096
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"HESScorer"` | Identifier for the scorer |
| `model` | string | `"Qwen/Qwen2.5-7B-Instruct"` | HuggingFace model path for the causal language model used to compute token entropies |
| `percentile_cutoff` | float | `0.005` | Fraction of highest-entropy tokens to include in HES calculation (0.5% by default) |
| `batch_size` | integer | `8` | Number of samples to process in parallel per forward pass |
| `max_length` | integer | `4096` | Maximum sequence length for tokenization (prompt + completion) |

## Underlying Model

The HES Scorer requires a **causal language model** to compute token-level entropy from logits. Unlike task-specific scorers, HES is **model-agnostic** and can work with any decoder-only language model from the Hugging Face transformers library that supports `AutoModelForCausalLM`.

### Recommended Models:
* Models aligned with your training objective (e.g., if training Qwen-based models, use Qwen variants for scoring)
* Instruction-tuned models often provide better signal for reasoning tasks
* Examples: `Qwen/Qwen2.5-7B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct`, `mistralai/Mistral-7B-Instruct-v0.3`

### Model Requirements:
* Must be a causal language model (decoder-only architecture)
* Should be accessible via Hugging Face `AutoModelForCausalLM`
* Must support `torch_dtype=torch.bfloat16` and `device_map="auto"`
* Must have a corresponding tokenizer via `AutoTokenizer`

The scorer automatically handles model loading, tokenization, and entropy computation. No additional training or fine-tuning of the model is required—HES is entirely **training-free**.

## Scoring Process

The HES scoring pipeline operates through the following steps:

### 1. **Input Preparation**
   * Each sample consists of `instruction`, `input`, and `output` fields
   * The prompt is constructed as: `instruction + "\n" + input` (or just `instruction` if input is empty)
   * The completion is the `output` field containing the CoT reasoning

### 2. **Tokenization and Length Management**
   * Concatenate prompt + completion to form the full text
   * Tokenize separately to determine prompt length (where completion starts)
   * Check total length against `max_length`
   * If exceeding limit, truncate from the end (completion is truncated, prompt preserved when possible)
   * Flag truncated samples for later analysis

### 3. **Model Forward Pass**
   * Process samples in batches for efficiency
   * Use left-padding for batch processing (right-padding for single samples)
   * Run model inference with `torch.no_grad()` to obtain logits for each token position
   * Output shape: `(batch_size, sequence_length, vocabulary_size)`

### 4. **Entropy Calculation**
   * For each token position in the **completion span only** (excluding prompt):
     * Extract logits for predicting that token
     * Convert to probabilities: `p = softmax(logits)`
     * Compute Shannon entropy: `H = -Σ(p_i × log₂(p_i))`
     * Add small epsilon (1e-9) to prevent log(0)
   * Collect all token entropies for the completion

### 5. **HES Aggregation**
   * Compute the percentile threshold: `threshold = percentile(entropies, (1 - p) × 100)`
   * Select tokens with entropy ≥ threshold
   * If no tokens selected (edge case), use the single maximum entropy token
   * Sum the selected high-entropy values to get the final HES score

### 6. **Result Collection**
   * Record HES score, completion token count, entropy threshold, and truncation flag
   * Clear CUDA cache between batches to manage memory

## Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 245.73,
  "completion_token_length": 512,
  "entropy_threshold": 8.42,
  "truncated": false
}
```

- `id`: Unique identifier from the input sample
- `score`: HES score (sum of entropies for top percentile highest-entropy tokens). Higher values indicate more complex reasoning
- `completion_token_length`: Number of tokens in the completion (reasoning output)
- `entropy_threshold`: Entropy cutoff value at the (1-p) percentile for selecting high-entropy tokens
- `truncated`: Flag indicating whether the sample exceeded max_length and was truncated

## Citation

```bibtex
@misc{anonymous2025unified,
  title={Unified Data Selection for {LLM} Re{ASON}ing},
  author={Anonymous},
  year={2025},
  note={Manuscript under review. Submitted to ICLR 2025},
  howpublished={\url{https://openreview.net/forum?id=heVn5cNfje}}
}
```
