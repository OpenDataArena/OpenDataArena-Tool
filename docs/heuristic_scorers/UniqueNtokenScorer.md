# UniqueNtokenScorer

## Overview

The **Unique N-token Scorer** is a statistical evaluation tool designed to measure the lexical diversity of SFT (Supervised Fine-Tuning) data by calculating the ratio of unique token-level n-grams to total token-level n-grams in the combined text. Unlike word-based n-gram analysis, this scorer operates at the **token level** using tiktoken encoders, which provides a more granular and tokenizer-aware assessment of text diversity that aligns with how modern language models process text.

This metric is particularly valuable for evaluating training data quality from a model-centric perspective, as it captures repetitive patterns at the subword level. Higher unique token n-gram ratios indicate more diverse token sequences and less repetitive patterns, which is generally desirable for training data quality. The scorer supports parallel processing for efficient evaluation of large datasets.

## Metric Definition:

* **Definition:** 

  ```
  Unique_Ntoken_Score = |unique token n-grams| / |total token n-grams|
  ```

  where token n-grams are extracted from the tokenized text (using tiktoken encoder) of combined instruction, input, and output fields.

* **Explanation:** This metric quantifies the lexical diversity of text at the token level by measuring the proportion of distinct token n-gram patterns.

  * A **higher Unique N-token Score** (closer to 1) indicates **greater token-level diversity** with minimal repetition, suggesting rich and varied token sequences.
  * A **lower Unique N-token Score** (closer to 0) indicates **more repetitive token patterns**, suggesting redundant or formulaic sequences at the subword level.
  * A score of **0** is assigned when the text has fewer tokens than the specified n value.

## YAML Configuration

```yaml
name: UniqueNtokenScorer
encoder: o200k_base
n: 2
max_workers: 8
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"UniqueNtokenScorer"` | Identifier for the scorer. Used for logging and result tracking. |
| `encoder` | string | `"o200k_base"` | The tiktoken encoder to use for tokenization. Options: `"o200k_base"` (GPT-4o and newer models, 200k vocabulary), `"cl100k_base"` (GPT-4 and GPT-3.5-turbo, 100k vocabulary), `"p50k_base"` (Codex and older GPT-3 models, 50k vocabulary), `"r50k_base"` (Older GPT-3 models like davinci, 50k vocabulary). The choice of encoder should align with the tokenizer of the target model for training. |
| `n` | integer | `2` | The n-gram size to use for token-level analysis. `n=1` measures unique token diversity (unigrams), `n=2` measures unique token pair diversity (bigrams), `n=3` measures unique three-token sequence diversity (trigrams). Higher values of `n` capture longer token patterns but require more tokens in the text. |
| `max_workers` | integer | CPU count | The number of parallel workers for multiprocessing. Automatically defaults to the system's CPU core count if not specified or invalid. Higher values can speed up processing but may increase memory usage. Recommended to set based on available CPU cores and memory. |

## Underlying Model

This scorer does **not require a deep learning model**. It uses **tiktoken**, OpenAI's fast byte pair encoding (BPE) tokenizer library, for token-level text processing. The default encoder is `o200k_base`, which is used by GPT-4o and newer models. Users can specify alternative tiktoken encoders based on their target model's tokenization scheme.

Tiktoken is a highly efficient tokenization library that provides:
- Fast encoding and decoding
- Multiple pre-trained BPE vocabularies
- Consistency with OpenAI's language models

## Scoring Process

1. **Text Extraction:** For each data item, extract the `instruction`, `input` (optional), and `output` fields.

2. **Text Combination:** Concatenate the fields with newline separators:
   - If input exists: `instruction + '\n' + input + '\n' + output`
   - Otherwise: `instruction + '\n' + output`

3. **Token Encoding:** Use tiktoken's encoder to convert the text into a sequence of token IDs based on the specified encoder (e.g., `o200k_base`).

4. **Token N-gram Generation:** Generate all token-level n-grams from the token ID sequence:
   - For each position `i` in the token list, extract a tuple of `n` consecutive token IDs: `(token[i], token[i+1], ..., token[i+n-1])`
   - Continue until reaching the end of the token sequence

5. **Uniqueness Calculation:** 
   - Count the total number of token n-grams
   - Count the number of unique token n-grams (using a set of tuples)
   - Calculate the ratio: `unique_count / total_count`

6. **Edge Cases:**
   - If the token count is less than `n`, return a score of 0.0
   - If encoding fails, return a score of 0.0 and log the error

7. **Parallel Processing:** The scorer uses `ProcessPoolExecutor` to process multiple data items in parallel, with progress tracking via `tqdm`.

## Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 0.89
}
```

- `id`: The unique identifier of the data item, extracted from the `id` field in the input data. Returns an empty string if the `id` field is not present.
- `score`: The Unique N-token Ratio, ranging from 0.0 to 1.0. A value of **1.0** indicates every token n-gram in the text is unique (maximum diversity), **0.5** indicates half of the token n-grams are unique, and **0.0** indicates all token n-grams are identical (maximum repetition) or text has fewer than `n` tokens.
- `error` (optional): Only present if an error occurred during processing. Contains the error message for debugging purposes.

## Citation

```bibtex
@misc{opendataarena_tool_2025,
  author       = {OpenDataArena},
  title        = {{OpenDataArena-Tool}},
  year         = {2025},
  url          = {https://github.com/OpenDataArena/OpenDataArena-Tool},
  note         = {GitHub repository},
  howpublished = {\url{https://github.com/OpenDataArena/OpenDataArena-Tool}},
}
```
