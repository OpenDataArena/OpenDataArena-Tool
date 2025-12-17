# UniqueNgramScorer

## Overview

The **Unique N-gram Scorer** is a statistical evaluation tool designed to measure the lexical diversity of SFT (Supervised Fine-Tuning) data by calculating the ratio of unique n-grams to total n-grams in the combined text. This metric provides insight into the linguistic richness and repetitiveness of instruction-response pairs. Higher unique n-gram ratios indicate more diverse vocabulary and less repetitive text patterns, which is generally desirable for training data quality.

The scorer processes text by combining the instruction, input (if present), and output fields, then tokenizes the text and computes n-gram statistics. It supports parallel processing for efficient evaluation of large datasets.

## Metric Definition:

* **Definition:** 

  ```
  Unique_Ngram_Score = |unique n-grams| / |total n-grams|
  ```

  where n-grams are extracted from the lowercase tokenized text of combined instruction, input, and output fields.

* **Explanation:** This metric quantifies the lexical diversity of text by measuring the proportion of distinct n-gram patterns.
  
  * A **higher Unique N-gram Score** (closer to 1) indicates **greater lexical diversity** with minimal repetition, suggesting rich and varied language use.
  * A **lower Unique N-gram Score** (closer to 0) indicates **more repetitive patterns**, suggesting redundant or formulaic language.
  * A score of **0** is assigned when the text has fewer tokens than the specified n value.

## YAML Configuration

```yaml
name: UniqueNgramScorer
n: 2
max_workers: 8
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"UniqueNgramScorer"` | Identifier for the scorer |
| `n` | integer | `2` | The n-gram size to use for analysis. `n=1` (unigrams) measures unique word diversity, `n=2` (bigrams) measures unique word pair diversity, `n=3` (trigrams) measures unique three-word phrase diversity. Higher values capture longer linguistic patterns but require more tokens in the text |
| `max_workers` | integer | CPU count | Number of parallel workers for multiprocessing. Automatically defaults to the system's CPU core count if not specified or invalid. Higher values can speed up processing but may increase memory usage |

## Underlying Model

This scorer does **not require a deep learning model**. It uses **NLTK's punkt tokenizer** for word tokenization, which is a rule-based, language-independent tokenizer. The tokenizer is automatically downloaded if not already present in the system.

## Scoring Process

1. **Text Extraction:** For each data item, extract the `instruction`, `input` (optional), and `output` fields.

2. **Text Combination:** Concatenate the fields with newline separators:
   - If input exists: `instruction + '\n' + input + '\n' + output`
   - Otherwise: `instruction + '\n' + output`

3. **Text Normalization:** Convert the combined text to lowercase to ensure case-insensitive analysis.

4. **Tokenization:** Use NLTK's `word_tokenize` to split the text into tokens.

5. **N-gram Generation:** Generate all n-grams from the token sequence using the specified `n` value.

6. **Uniqueness Calculation:** 
   - Count the total number of n-grams
   - Count the number of unique n-grams (using a set)
   - Calculate the ratio: `unique_count / total_count`

7. **Edge Cases:**
   - If the token count is less than `n`, return a score of 0.0
   - If no n-grams can be generated, return a score of 0.0

8. **Parallel Processing:** The scorer uses `ProcessPoolExecutor` to process multiple data items in parallel, with progress tracking via `tqdm`.

## Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 0.87
}
```

- `id`: The unique identifier of the data item, extracted from the `id` field in the input data. Returns an empty string if the `id` field is not present
- `score`: The Unique N-gram Ratio, ranging from 0.0 to 1.0. **1.0** indicates every n-gram in the text is unique (maximum diversity), **0.5** indicates half of the n-grams are unique, **0.0** indicates all n-grams are identical (maximum repetition) or text has fewer than `n` tokens
- `error` (optional): Only present if an error occurred during processing. Contains the error message for debugging purposes

## Citation

```bibtex
@inproceedings{zhuang2025meta,
  title={Meta-rater: A multi-dimensional data selection method for pre-training language models},
  author={Zhuang, Xinlin and Peng, Jiahui and Ma, Ren and Wang, Yinfan and Bai, Tianyi and Wei, Xingjian and Jiantao, Qiu and Zhang, Chi and Qian, Ying and He, Conghui},
  booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={10856--10896},
  year={2025}
}
```

