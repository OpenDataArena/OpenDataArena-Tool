# AskLlmScorer

## Overview

The **AskLlmScorer** is a sample-wise data quality evaluation metric that leverages Large Language Models (LLMs) to assess the quality of individual training samples. It computes the **average log probability** of a specified positive token (e.g., "yes") given a prompt and the data sample, effectively measuring how likely the model believes the sample is high quality.

## Metric Definition:

* **Definition:**

  Given a prompt P, data sample D, and target positive token(s) Y, the scorer computes:
  
  For a single-token yes_token:
  ```
  Score = log P(yes_token | prompt, data)
  ```
  
  For a multi-token yes_token:
  ```
  Score = (1/T) × Σ log P(token_i | prompt, data, token_1...token_{i-1})
  ```
  
  where T is the number of tokens in the yes_token.

* **Explanation:** The metric quantifies data quality through conditional probability:
  
  * A **higher score** (closer to 0) indicates the LLM assigns **high probability** to the positive response, suggesting the sample is likely high quality.
  * A **lower score** (more negative) indicates the LLM assigns **low probability** to the positive response, suggesting the sample may be low quality.
  * Scores typically range from approximately -10 to 0, with scores above -2 generally indicating good quality.

* **Key Advantages:**
  
  * **Customizable prompts:** Allows flexible quality criteria through prompt engineering
  * **Multi-token support:** Handles both single and multi-token positive responses
  * **Probabilistic interpretation:** Provides interpretable quality scores based on model confidence

## YAML Configuration

```yaml
name: AskLlmScorer
model: Qwen/Qwen2.5-7B
prompt: "Is the following data high quality? Please answer yes or no.\n\n"
yes_token: "yes"
batch_size: 8
max_length: 2048
model_dtype: bfloat16
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"AskLlmScorer"` | Identifier for the scorer |
| `model` | string | `"Qwen/Qwen2.5-7B"` | HuggingFace model identifier or local path to the evaluation LLM |
| `prompt` | string | `"Is the following data high quality? Please answer yes or no.\n\n"` | Prompt template that precedes the data sample |
| `yes_token` | string | `"yes"` | Target token(s) representing positive/high-quality response |
| `batch_size` | integer | `8` | Number of samples to process in parallel per batch |
| `max_length` | integer | `2048` | Maximum sequence length in tokens; longer sequences are truncated |
| `model_dtype` | string | `bfloat16` | Precision for model loading: `"float32"`, `"bfloat16"`, or `"float16"` |

## Underlying Model

The scorer uses causal language models from the HuggingFace ecosystem to compute token-level probabilities. By default, it uses **Qwen/Qwen2.5-7B**, but can be configured to use any autoregressive language model with strong instruction-following capabilities. Larger models generally provide more accurate quality judgments but require more computational resources.

## Scoring Process

1. **Data Preparation**: For each sample, construct the full text by concatenating instruction, input (if present), and output fields.

2. **Prompt Construction**: Build the evaluation prompt by appending the data sample to the configured prompt template, followed by the yes_token.

3. **Batch Tokenization**: Tokenize samples in batches with padding and truncation. Track prompt lengths to identify yes_token positions.

4. **Model Inference**: Run forward pass through the LLM to obtain logits for all token positions in batch mode.

5. **Log Probability Computation**: For each sample, extract log probabilities of yes_token tokens given the prompt and data context. Compute average log probability across all yes_token tokens. Critical computations use float32 precision for numerical stability.

6. **Score Assignment**: Return the average log probability as the quality score. Handle edge cases (truncation, empty tokens) by assigning a score of -100.0.

## Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": -0.523
}
```

- `id`: Unique identifier for the sample (from input data's `id` field, or empty string if not present)
- `score`: Average log probability of yes_token given prompt and data. Higher (less negative) scores indicate higher quality. Typical range: -10 to 0, with scores above -2 generally indicating good quality

## Citation

```bibtex
@article{sachdeva2024train,
  title={How to train data-efficient llms},
  author={Sachdeva, Noveen and Coleman, Benjamin and Kang, Wang-Cheng and Ni, Jianmo and Hong, Lichan and Chi, Ed H and Caverlee, James and McAuley, Julian and Cheng, Derek Zhiyuan},
  journal={arXiv preprint arXiv:2402.09668},
  year={2024}
}
```
