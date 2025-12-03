# ApJsScorer.py

## Overview

The **ApjsScorer** is a dataset-level diversity evaluation metric that measures the **Average Pairwise Jaccard Similarity (Apjs)** across all samples in an SFT dataset. Unlike sample-wise scoring methods, ApjsScorer computes a single aggregate score for the entire dataset by calculating the average Jaccard similarity between all possible pairs of samples based on their n-gram representations.

This metric is particularly useful for assessing dataset diversity, data deduplication analysis, and dataset quality evaluation. ApjsScorer supports flexible tokenization methods (word-level n-grams or token-level n-grams) and similarity computation strategies (exact or approximate), making it scalable for datasets of varying sizes.

## Metric Definition:

* **Definition:** 

  Given a dataset with N samples, the Apjs score is computed as:
  
  `Apjs = (1 / C(N,2)) × Σ Jaccard(S_i, S_j)`
  
  where `C(N,2) = N×(N-1)/2` is the total number of unique pairs, `S_i` and `S_j` are n-gram sets extracted from samples i and j, and `Jaccard(S_i, S_j) = |S_i ∩ S_j| / |S_i ∪ S_j|`.

* **Explanation:** This metric quantifies dataset-level diversity by measuring the average overlap between all sample pairs:
  
  * A **lower Apjs score** (closer to 0) indicates **higher diversity**, meaning samples share fewer common n-grams and the dataset contains more unique content.
  * A **higher Apjs score** (closer to 1) indicates **lower diversity**, suggesting many samples contain similar or redundant content.
  * A **score around 0.5** suggests **moderate diversity** with balanced content variation.

* **Key Advantages:**
  
  * **Dataset-level metric:** Provides a holistic view of diversity across the entire dataset
  * **Flexible tokenization:** Supports both word-level and token-level n-grams
  * **Scalable computation:** MinHash approximation enables efficient processing of large datasets

## YAML Configuration

```yaml
name: ApjsScorer
tokenization_method: gram
n: 3
similarity_method: direct
encoder: o200k_base
num_perm: 128
max_workers: 8
sample_pairs: null
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"ApjsScorer"` | Identifier for the scorer |
| `tokenization_method` | string | `"gram"` | Tokenization strategy: `"gram"` for word-level n-grams or `"token"` for token-level n-grams |
| `n` | int | `1` | Size of n-grams to extract from each sample |
| `similarity_method` | string | `"direct"` | Similarity computation: `"direct"` for exact calculation or `"minhash"` for approximation |
| `encoder` | string | `"o200k_base"` | Tiktoken encoder name (only for `tokenization_method="token"`) |
| `num_perm` | int | `128` | Number of hash permutations (only for `similarity_method="minhash"`) |
| `max_workers` | int | CPU count | Number of parallel processes for pairwise similarity computation |
| `sample_pairs` | int/null | `null` | Number of pairs to randomly sample (useful for very large datasets) |

## Scoring Process

1. **Text Extraction**: For each sample in the dataset, concatenate the instruction, input (if present), and output fields into a single text string

2. **N-gram Generation**: Generate n-gram sets based on the configured tokenization method:
   - `tokenization_method="gram"`: Tokenize text into words using NLTK's `word_tokenize`, convert to lowercase, and generate word-level n-grams
   - `tokenization_method="token"`: Encode text using tiktoken encoder and generate token-level n-grams from token IDs

3. **Pairwise Similarity Computation**: Compute Jaccard similarity for all unique pairs using the configured method:
   - `similarity_method="direct"`: Exact computation using `|S_i ∩ S_j| / |S_i ∪ S_j|`
   - `similarity_method="minhash"`: Approximate similarity using MinHash sketches

4. **Parallel Processing**: Leverages multi-processing with `ProcessPoolExecutor` to parallelize pairwise comparisons across CPU cores

5. **Aggregation**: Calculate the mean of all pairwise Jaccard similarities to obtain the final Apjs score

## Output Format

For each dataset evaluation, the scorer returns:

```json
{
  "score": 0.234,
  "num_samples": 1000,
  "num_pairs": 499500,
  "total_possible_pairs": 499500,
  "is_sampled": false,
  "tokenization_method": "gram",
  "n": 3,
  "similarity_method": "direct",
  "max_workers": 8
}
```

- `score`: The Average Pairwise Jaccard Similarity (Apjs) score for the dataset
- `num_samples`: Total number of samples in the dataset
- `num_pairs`: Number of pairs actually computed (equals `total_possible_pairs` if not sampled)
- `total_possible_pairs`: Total number of possible unique pairs: N×(N-1)/2
- `is_sampled`: Whether pairs were randomly sampled (true if `sample_pairs` was used)
- `tokenization_method`: Tokenization method used: "gram" or "token"
- `n`: N-gram size used for extraction
- `similarity_method`: Similarity computation method: "direct" or "minhash"
- `max_workers`: Number of parallel workers used

## Citation

```bibtex
@article{seed2025seed,
  title={Seed-coder: Let the code model curate data for itself},
  author={Seed, ByteDance and Zhang, Yuyu and Su, Jing and Sun, Yifan and Xi, Chenguang and Xiao, Xia and Zheng, Shen and Zhang, Anxiang and Liu, Kaibo and Zan, Daoguang and others},
  journal={arXiv preprint arXiv:2506.03524},
  year={2025}
}
```
