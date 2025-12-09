# GramEntropyScorer

## Overview

The **Gram Entropy Scorer** is a statistical evaluation tool designed to measure the lexical diversity and linguistic richness of SFT (Supervised Fine-Tuning) data at the word level by computing 1-gram (unigram) entropy. This scorer analyzes the distribution of words in instruction-response pairs and quantifies their unpredictability using Shannon entropy. Higher entropy scores indicate more diverse vocabulary usage and potentially richer linguistic content, while lower scores suggest repetitive or limited word patterns.

Unlike token-based entropy scorers that operate on subword units, this scorer works at the natural word level using linguistic tokenization, making it more interpretable from a linguistic perspective and suitable for analyzing vocabulary diversity.

## Metric Definition:

* **Definition:** 

  1-Gram Entropy (Unigram Entropy) is computed using Shannon's entropy formula:
  
  ```
  H(X) = -Σ p(x) * log₂(p(x))
  ```
  
  where `p(x)` is the probability (frequency) of each unique word in the text.

* **Explanation:** This metric quantifies the unpredictability or diversity of word usage in the data:
  
  * A **higher 1-Gram Entropy score** indicates greater vocabulary diversity, with words distributed more evenly throughout the text. This suggests the data contains varied and linguistically rich content.
  * A **lower 1-Gram Entropy score** suggests repetitive word usage or limited vocabulary, potentially indicating redundant, template-like, or formulaic content.

## YAML Configuration

```yaml
name: GramEntropyScorer
max_workers: 8
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"GramEntropyScorer"` | Identifier for the scorer |
| `max_workers` | integer | Number of CPU cores | Number of parallel processes to use for scoring. Higher values speed up processing but consume more CPU resources |

## Underlying Model

This scorer does **not require a deep learning model**. It uses the **NLTK (Natural Language Toolkit)** library for word-level tokenization, specifically the `punkt_tab` tokenizer for sentence and word boundary detection. The tokenizer automatically downloads if not present in the system.

Unlike subword tokenizers (e.g., BPE or SentencePiece), NLTK's word tokenizer splits text into linguistically meaningful words, making it more suitable for analyzing natural vocabulary diversity.

## Scoring Process

1. **Text Concatenation**: For each data item, the instruction, optional input, and output (response) are concatenated: `text = instruction + '\n' + input + '\n' + output` (If no input field exists, it is omitted)

2. **Text Normalization**: The concatenated text is converted to lowercase to ensure case-insensitive word counting: `text = text.lower()`

3. **Word Tokenization**: The normalized text is tokenized into words using NLTK's `word_tokenize` function, which splits text into individual words, handling punctuation and special characters appropriately

4. **Word Frequency Analysis**: The frequency of each unique word is counted to build a probability distribution: `p(word_i) = count(word_i) / total_words`

5. **Entropy Calculation**: Shannon entropy is computed across all unique words: `H = -Σ p(word_i) * log₂(p(word_i))`

6. **Parallel Processing**: When evaluating datasets, the scorer uses `ProcessPoolExecutor` to distribute work across multiple CPU cores for efficient batch processing. Each worker process independently downloads NLTK data if needed

## Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 7.523
}
```

- `id`: The unique identifier of the data sample, extracted from the input data's `id` field
- `score`: The computed 1-gram entropy value. Higher values indicate greater vocabulary diversity. A score of 0.0 indicates either empty text or an error during processing
- `error` (optional): Present only if an error occurred during processing (e.g., tokenization failure). Contains the error message

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

