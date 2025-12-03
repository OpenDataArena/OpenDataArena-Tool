# MIWVScorer

## Overview

The **MIWV (Model Instruction Weakness Value) Scorer** is a model-based evaluation tool designed to quantify the importance of instruction data in enhancing a language model's capabilities. Proposed in the paper [Jiang et al., 2025](https://arxiv.org/abs/2511.07074), this method identifies the most beneficial data for instruction tuning by measuring the discrepancies in the model's responses when using In-Context Learning (ICL). 

The key insight behind MIWV is that high-quality instruction data should maximize the performance gains for a given LLM during instruction tuning. Rather than focusing solely on data quality scores, MIWV evaluates how much a specific instruction sample can help improve the model's weakness areas by comparing its performance with and without relevant contextual examples.

## Metric Definition:

* **Definition:** 
  
  MIWV(x) = Loss_one-shot(x) - Loss_zero-shot(x)
  
  where:
  - `Loss_zero-shot(x)` is the cross-entropy loss of generating the output given only the instruction
  - `Loss_one-shot(x)` is the cross-entropy loss of generating the output with the most similar sample as an ICL example
  - The most similar sample is determined by embedding similarity using a specified distance metric

* **Explanation:** This metric quantifies how much the model struggles with an instruction even when provided with a relevant example:
  
  * A **higher MIWV score** (positive value) indicates that the model performs **worse** with the ICL example than without it, suggesting this is a **weakness area** where the model needs improvement. Such data is **more valuable** for instruction tuning.
  * A **lower MIWV score** (negative or close to zero) suggests the ICL example helps the model, indicating the instruction is already within the model's capabilities and may be **less critical** for training.

## YAML Configuration

```yaml
name: MIWVScorer
model: /path/to/your/model
embedding_path: /path/to/embeddings.npy
batch_size: 8
max_length: 2048
distance_metric: cosine
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"MIWVScorer"` | Identifier for the scorer |
| `model` | string | (required) | Path to the causal language model used for loss computation. Can be any HuggingFace-compatible model path |
| `embedding_path` | string | (required) | Path to a `.npy` file containing precomputed embeddings for all samples in the dataset. The embedding array shape should be `(num_samples, embedding_dim)`, and the order must match the dataset order |
| `batch_size` | integer | `8` | Number of samples to process simultaneously. Larger values speed up computation but require more GPU memory |
| `max_length` | integer | `2048` | Maximum token length for input sequences. Sequences exceeding this length will be truncated with a warning |
| `distance_metric` | string | `"cosine"` | Metric used to find the most similar sample for ICL. Available options: `cosine` (cosine distance), `euclidean` (Euclidean distance), `squared_euclidean` (squared Euclidean distance), `manhattan` (Manhattan distance) |

## Underlying Model

The MIWV Scorer can work with **any causal language model** that supports the HuggingFace `AutoModelForCausalLM` interface. The choice of model depends on your specific use case and instruction tuning target. Common choices include:

- Base models (e.g., `meta-llama/Llama-2-7b`, `Qwen/Qwen2.5-7B`)
- Instruction-tuned models (e.g., `Qwen/Qwen2.5-7B-Instruct`)

**Note:** The embeddings used for similarity computation should be generated separately using an appropriate embedding model (e.g., sentence transformers, model hidden states) and saved as a numpy array before running the scorer.

## Scoring Process

1. **Embedding-Based Similarity Computation:**
   - Load precomputed embeddings for all samples in the dataset
   - For each sample, compute distances to all other samples using the specified distance metric
   - Identify the most similar sample (minimum distance, excluding self)

2. **Text Construction:**
   - **Zero-shot format:** `User: {instruction}\nAssistant: {output}`
   - **One-shot format:** 
     ```
     User: {similar_instruction}
     Assistant: {similar_output}
     User: {instruction}
     Assistant: {output}
     ```

3. **Loss Computation:**
   - Tokenize both zero-shot and one-shot texts with the target model
   - Compute cross-entropy loss only on the output tokens (prompt tokens are masked with -100)
   - Use batch processing for efficiency

4. **MIWV Calculation:**
   - MIWV = one-shot loss - zero-shot loss
   - Positive scores indicate data that exposes model weaknesses

5. **Result Aggregation:**
   - Store MIWV score along with metadata about the most similar sample

## Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 0.2345,
  "most_similar_idx": 42,
  "most_similar_id": "sample_042"
}
```

- `id`: Unique identifier of the evaluated sample. Extracted from the `id` field in the input data, or uses the sample index if not present
- `score`: The MIWV score (float). Higher values indicate more important data for instruction tuning. Positive values suggest the model struggles even with ICL examples (weakness area)
- `most_similar_idx`: Integer index of the most similar sample in the dataset that was used as the ICL example
- `most_similar_id`: The unique identifier of the most similar sample (for traceability and debugging)

## Citation

```bibtex
@article{jiang2025importance,
  title={Importance-Aware Data Selection for Efficient LLM Instruction Tuning},
  author={Jiang, Tingyu and Li, Shen and Song, Yiyao and Zhang, Lan and Zhu, Hualei and Zhao, Yuan and Xu, Xiaohang and Taura, Kenjiro and Wang, Hao Henry},
  journal={arXiv preprint arXiv:2511.07074},
  year={2025}
}
```

