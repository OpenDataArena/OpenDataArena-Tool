# TokenEntropyScorer

## Overview

The **Token Entropy Scorer** is a statistical evaluation tool designed to measure the lexical diversity and information richness of SFT (Supervised Fine-Tuning) data by computing token-level entropy. This scorer analyzes the distribution of tokens in instruction-response pairs and quantifies their unpredictability using Shannon entropy. Higher entropy scores indicate more diverse token usage and potentially richer information content, while lower scores suggest repetitive or predictable token patterns.

Unlike model-based approaches, this scorer relies purely on statistical analysis of token distributions using the `tiktoken` tokenizer, making it computationally efficient and model-agnostic.

## Metric Definition:

* **Definition:** 

  Token Entropy is computed using Shannon's entropy formula:
  
  ```
  H(X) = -Σ p(x) * log₂(p(x))
  ```
  
  where `p(x)` is the probability (frequency) of each unique token in the text.

* **Explanation:** This metric quantifies the unpredictability or diversity of token usage in the data:
  
  * A **higher Token Entropy score** indicates greater lexical diversity, with tokens distributed more evenly across the vocabulary. This suggests the data contains varied and information-rich content.
  * A **lower Token Entropy score** suggests repetitive token usage or limited vocabulary diversity, potentially indicating redundant or template-like content.

## YAML Configuration

```yaml
name: TokenEntropyScorer
encoder: o200k_base
max_workers: 8
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"TokenEntropyScorer"` | Identifier for the scorer instance. Used for logging and progress tracking |
| `encoder` | string | `"o200k_base"` | The tiktoken encoder to use for tokenization. Common options: `o200k_base` (GPT-4o and newer), `cl100k_base` (GPT-4, GPT-3.5-turbo), `p50k_base` (older GPT-3 models) |
| `max_workers` | integer | CPU cores | Number of parallel processes to use for scoring. Higher values speed up processing but consume more CPU resources |

## Underlying Model

This scorer does **not require a deep learning model**. It uses the `tiktoken` library for tokenization, which provides efficient byte-pair encoding (BPE) tokenizers compatible with OpenAI models. The tokenizer is used purely for splitting text into tokens; no neural network inference is involved.

## Scoring Process

The Token Entropy Scorer evaluates each data sample through the following steps:

1. **Text Concatenation:** For each data item, the instruction, optional input, and output (response) are concatenated:
   ```
   text = instruction + '\n' + input + '\n' + output
   ```
   (If no input field exists, it is omitted)

2. **Tokenization:** The concatenated text is tokenized using the specified tiktoken encoder, producing a sequence of token IDs:
   ```python
   tokens = encoder.encode(text, disallowed_special=())
   ```

3. **Token Frequency Analysis:** The frequency of each unique token is counted to build a probability distribution:
   ```
   p(token_i) = count(token_i) / total_tokens
   ```

4. **Entropy Calculation:** Shannon entropy is computed across all unique tokens:
   ```
   H = -Σ p(token_i) * log₂(p(token_i))
   ```

5. **Parallel Processing:** When evaluating datasets, the scorer uses `ProcessPoolExecutor` to distribute work across multiple CPU cores for efficient batch processing.

## Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 8.234
}
```

- `id`: The unique identifier of the data sample, extracted from the input data's `id` field
- `score`: The computed token entropy value. Higher values indicate greater lexical diversity. A score of 0.0 indicates either empty text or an error during processing

## Citation

```bibtex
@inproceedings{zhuang2025meta,
  title        = {Meta-rater: A multi-dimensional data selection method for pre-training language models},
  author       = {Zhuang, Xinlin and Peng, Jiahui and Ma, Ren and Wang, Yinfan and Bai, Tianyi and Wei, Xingjian and Jiantao, Qiu and Zhang, Chi and Qian, Ying and He, Conghui},
  booktitle    = {Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages        = {10856--10896},
  year         = {2025}
}
```
