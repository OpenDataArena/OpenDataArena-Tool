# TokenLengthScorer

## Overview

The **Token Length Scorer** is a lightweight, efficient evaluation tool designed to measure the token length of SFT (Supervised Fine-Tuning) data samples. Unlike model-based scorers, this metric provides a deterministic, tokenization-based measurement that quantifies the raw token count of specified fields in each data sample. Token length serves as a fundamental characteristic for data selection, filtering, and quality control in instruction-tuning pipelines, helping practitioners identify samples that are too short (potentially low-information) or too long (potentially noisy or verbose).

This scorer leverages OpenAI's **tiktoken** library to efficiently tokenize text and supports parallel processing for large-scale datasets.

## Metric Definition:

* **Definition:** 
  
  Token_Length = number of tokens after encoding the concatenated text of specified fields using a tiktoken encoder.

* **Explanation:** This metric counts the total number of tokens in a data sample after concatenating the specified fields (e.g., instruction, input, output) with newline separators.
  
  * A **higher Token Length** indicates a longer sample, which may contain more comprehensive information but could also be verbose or redundant.
  * A **lower Token Length** suggests a more concise sample, which could be efficient but might lack necessary detail.

## YAML Configuration

```yaml
name: TokenLengthScorer
encoder: o200k_base
fields:
  - instruction
  - input
  - output
max_workers: 8
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"TokenLengthScorer"` | Identifier for the scorer |
| `encoder` | string | `"o200k_base"` | The tiktoken encoder to use for tokenization. Common options: `o200k_base` (GPT-4o), `cl100k_base` (GPT-4, GPT-3.5-turbo), `p50k_base` (Codex models). Automatically falls back to `o200k_base` if the specified encoder fails to load. |
| `fields` | list | `["instruction", "input", "output"]` | The fields to extract from each data sample for token counting. Fields are concatenated with newline separators (`\n`) before tokenization. Only non-empty fields present in the data item are included. |
| `max_workers` | integer | Number of CPU cores | The number of parallel worker processes for data processing. Higher values can significantly speed up processing for large datasets. Recommended range: 4-16 depending on CPU availability and dataset size. |

## Underlying Model

This scorer **does not use a language model**. Instead, it relies on the **tiktoken** tokenization library, which provides fast and accurate byte-pair encoding (BPE) tokenizers used by OpenAI models. The default encoder is `o200k_base`, which is the tokenizer used by GPT-4o and other modern OpenAI models.

## Scoring Process

1. **Configuration Validation:** The scorer validates the configuration and sets defaults for `encoder`, `fields`, and `max_workers` if not specified.

2. **Field Extraction:** For each data sample, the specified fields (e.g., `instruction`, `input`, `output`) are extracted and filtered for non-empty values.

3. **Text Concatenation:** All extracted field values are converted to strings and concatenated using newline separators (`\n`).

4. **Tokenization:** The concatenated text is encoded using the specified tiktoken encoder with `disallowed_special=()` to handle special tokens properly.

5. **Token Counting:** The length of the encoded token list is computed as the final score.

6. **Parallel Processing:** The scorer uses `ProcessPoolExecutor` to process multiple samples in parallel, with a progress bar (tqdm) to track evaluation progress.

7. **Error Handling:** If any sample fails during processing, the scorer returns a score of 0 with an error marker.

## Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 256
}
```

- `id`: The unique identifier of the data sample, extracted from the `id` field in the input data. Returns `"unknown"` if the `id` field is missing or processing fails.
- `score`: The total token count of the concatenated text from specified fields. Returns `0` if tokenization fails or the sample is empty.

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
