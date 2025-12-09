# HddScorer

## Overview

The **HD-D (Hypergeometric Distribution D) Scorer** is a statistical evaluation tool designed to measure **lexical diversity** in text. Proposed by McCarthy & Jarvis (2010), HD-D provides a robust approach to quantifying vocabulary richness that is largely independent of text length, making it superior to traditional type-token ratio (TTR) measures. This scorer calculates the probability of encountering unique word types within sampled segments of text using the hypergeometric distribution function.

Unlike model-based scorers, HD-D is a **statistical metric** that does not require any pre-trained models. It is particularly useful for assessing the linguistic complexity and vocabulary variety of instruction-following datasets, providing insights into the lexical richness of both instructions and responses.

## Metric Definition:

* **Definition:** 

  HD-D is calculated by summing the contribution of each unique word type to the overall lexical diversity:

  ```
  HD-D = Σ [1 - P(X = 0)] / sample_size
  ```

  where `P(X = 0)` is the hypergeometric probability that a given word type does **not** appear in a random sample of tokens from the text.

* **Explanation:** This metric estimates the **lexical diversity** of text by measuring how likely each unique word type will appear in a randomly sampled segment:

  * A **higher HD-D score** indicates **greater lexical diversity**, suggesting rich vocabulary usage and varied word choices.
  * A **lower HD-D score** indicates **lower lexical diversity**, suggesting repetitive language or limited vocabulary.

The hypergeometric distribution accounts for sampling without replacement, making HD-D more mathematically sound than simple ratio-based measures. The metric is computed as:

```
P(X = k) = [C(K, k) × C(N-K, n-k)] / C(N, n)
```

where:
- `N` = total number of tokens in the text
- `K` = frequency of a specific word type
- `n` = sample size
- `k` = number of times the word type appears in the sample
- `C(n, r)` = binomial coefficient "n choose r"

## YAML Configuration

```yaml
name: HddScorer
sample_size: 42.0
max_workers: 8
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"HddScorer"` | Identifier for the scorer |
| `sample_size` | float | `42.0` | The sample size used in the hypergeometric distribution calculation. Controls the window size for calculating lexical diversity. If the text is shorter than the sample size, the actual text length will be used instead |
| `max_workers` | integer | CPU core count | Number of parallel worker processes for multiprocessing. Higher values can speed up processing for large datasets but consume more memory |

## Underlying Model

**Not applicable.** HD-D is a statistical metric based on the hypergeometric distribution and does not require any pre-trained language models. The calculation is purely mathematical and depends only on word frequency distributions in the text.

## Scoring Process

The HD-D scorer follows these steps to evaluate lexical diversity:

1. **Text Concatenation:** For each data sample, the `instruction`, `input`, and `output` fields are concatenated into a single text string:
   ```
   text = instruction + '\n' + input + '\n' + output
   ```
   If the `input` field is empty, it is omitted from concatenation.

2. **Tokenization:** The concatenated text is split into tokens (words) using whitespace separation.

3. **Preprocessing:** Each token is processed by:
   - Removing all punctuation marks
   - Converting to lowercase
   - Filtering out empty strings

4. **Type Counting:** A frequency dictionary is built where each unique word type (after preprocessing) is mapped to its occurrence count in the text.

5. **Hypergeometric Calculation:** For each unique word type:
   - Calculate the hypergeometric probability `P(X = 0)` that the word does **not** appear in a random sample
   - Compute the contribution: `[1 - P(X = 0)] / sample_size`
   - Handle any mathematical errors (overflow, division by zero) gracefully

6. **Score Aggregation:** Sum all individual contributions to obtain the final HD-D score.

7. **Parallel Processing:** Multiple samples are processed in parallel using `ProcessPoolExecutor` for improved performance on large datasets.

## Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 0.8523
}
```

- `id`: The unique identifier of the data sample, extracted from the input data's `id` field. If no `id` is present, defaults to `"unknown"`
- `score`: The HD-D lexical diversity score for the sample. Typically between 0 and the number of unique word types in the text. Higher values indicate greater lexical diversity. A value of 0.0 indicates empty text or processing error
- `error` (optional): Only present if an error occurred during processing. Contains the error message for debugging purposes

## Citation

```bibtex
@article{mccarthy2010mtld,
  title={MTLD, vocd-D, and HD-D: A validation study of sophisticated approaches to lexical diversity assessment},
  author={McCarthy, Philip M and Jarvis, Scott},
  journal={Behavior research methods},
  volume={42},
  number={2},
  pages={381--392},
  year={2010},
  publisher={Springer}
}
```

