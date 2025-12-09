# ApsScorer

## Overview

The **Average Pairwise Similarity (APS) Scorer** is a dataset-level diversity evaluation metric that measures the average pairwise similarity across all samples in a dataset by computing similarity between their embedding representations. Unlike sample-wise scoring methods, ApsScorer computes a single aggregate score for the entire dataset by calculating the average similarity between all possible pairs of sample embeddings.

This metric is particularly useful for:
- **Assessing dataset diversity**: Lower APS scores indicate higher diversity (less redundancy) in the dataset
- **Data deduplication analysis**: Identifying semantically similar or duplicate samples

ApsScorer supports multiple similarity metrics (cosine, euclidean, manhattan, dot product, and Pearson correlation) and includes parallel processing capabilities, making it scalable for datasets of varying sizes.

## Metric Definition:

* **Definition:** 

```
APS = (1 / C(N,2)) × Σ Similarity(E_i, E_j)
```

where:
- `N` is the number of samples in the dataset
- `C(N,2) = N×(N-1)/2` is the total number of unique pairs
- `E_i` and `E_j` are embedding vectors for samples i and j
- `Similarity(E_i, E_j)` is the similarity score computed using the specified metric

* **Explanation:** The APS metric quantifies dataset-level diversity by measuring the average similarity between all sample pairs in embedding space:

  * A **lower APS score** indicates **higher diversity**, meaning samples have more distinct semantic representations and the dataset contains more unique content
  * A **higher APS score** indicates **lower diversity**, suggesting many samples contain similar or redundant semantic information
  * The interpretation depends on the similarity metric:
    - For **cosine/dot product/Pearson**: scores range from -1 to 1 (higher = more similar)
    - For **euclidean/manhattan**: lower scores indicate higher similarity

## YAML Configuration

```yaml
name: ApsScorer
embedding_path: /path/to/embeddings.npy
similarity_metric: cosine
max_workers: 8
sample_pairs: null
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"ApsScorer"` | Identifier for the scorer |
| `embedding_path` | string | *required* | Path to pre-computed embeddings file in NumPy `.npy` format with shape (N, D) where N is the number of samples and D is the embedding dimension. Must be computed in advance using an embedding model (e.g., Sentence-BERT, BGE, E5) |
| `similarity_metric` | string | `"cosine"` | Similarity metric to use: `"cosine"` (angular similarity, [-1,1]), `"euclidean"` (L2 distance), `"manhattan"` (L1 distance), `"dot_product"` (inner product), or `"pearson"` (correlation coefficient) |
| `max_workers` | int | CPU count | Number of parallel processes for pairwise similarity computation. Adjust based on system capabilities and memory constraints |
| `sample_pairs` | int/null | `null` | Number of pairs to randomly sample for estimation. Set to a positive integer for very large datasets to reduce computation time from O(N²). Provides an approximate APS score |

## Underlying Model

ApsScorer does **not require a specific language model** for inference. Instead, it operates on **pre-computed embeddings** that must be generated in advance using an embedding model of your choice. 

**Note**: The embeddings must be saved as a NumPy `.npy` file with shape (N, D) where N matches the number of samples in your dataset and D is the embedding dimension. The order of embeddings must correspond to the order of samples in your dataset file.

## Generating Embeddings

To generate the required embedding file for ApsScorer, you can use the provided `embed.py` script located at:

```bash
data_scorer/model_based/utils/embed.py
```

### Usage Example

```bash
python data_scorer/model_based/utils/embed.py \
    --embedder_model /path/to/embedding/model \
    --input_path /path/to/your/dataset.jsonl \
    --output_path /path/to/output/embeddings.npy \
    --fields instruction input \
    --max_tokens 32768 \
    --tokenize_batch_size 16384 \
    --embed_batch_size 16384
```

### Script Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--embedder_model` | string | `Qwen/Qwen3-Embedding-8B` | Path or name of the vLLM model for computing embeddings (task=embed) |
| `--input_path` | string | *required* | Path to the input JSONL file containing your dataset |
| `--output_path` | string | *required* | Path to save the output `.npy` embedding file |
| `--fields` | list | `["instruction", "input", "output"]` | Field names to extract from JSONL and concatenate with newlines. Specify multiple fields to combine |
| `--max_tokens` | int | `32768` | Maximum number of tokens allowed per text; texts exceeding this will be truncated |
| `--tokenize_batch_size` | int | `16384` | Batch size for tokenization (encode_batch). Adjust based on memory |
| `--embed_batch_size` | int | `16384` | Batch size for embedding computation. Adjust based on GPU/memory |
| `--truncate_report_path` | string | `""` | Optional: Write line numbers of truncated samples to this text file |

### Key Features

- **Batch Processing**: Processes large datasets efficiently using batched tokenization and embedding computation
- **Automatic Truncation**: Handles long texts by truncating to the specified `max_tokens` limit
- **vLLM Integration**: Uses vLLM for fast and memory-efficient embedding generation with GPU acceleration
- **Flexible Field Extraction**: Supports extracting and concatenating multiple fields from JSONL data
- **Progress Tracking**: Displays progress bars using tqdm for both tokenization and embedding stages

### Output Format

The script generates a NumPy `.npy` file containing embeddings in float64 format with shape (N, D), where:
- N = number of samples in your input dataset
- D = embedding dimension of the chosen model

This output file can be directly used as the `embedding_path` parameter in the ApsScorer configuration.

## Scoring Process

1. **Embedding Loading**: Load pre-computed embeddings from the specified `.npy` file and validate that the number of embeddings matches the dataset size

2. **Similarity Function Selection**: Select the appropriate similarity computation function based on the configured metric (cosine, euclidean, manhattan, dot product, or Pearson)

3. **Pair Generation**: Generate all possible unique pairs C(N,2) = N×(N-1)/2, or randomly sample `sample_pairs` pairs if specified

4. **Parallel Similarity Computation**: Distribute pair computations across multiple worker processes using `ProcessPoolExecutor` for efficient parallel processing

5. **Aggregation**: Calculate the mean of all pairwise similarities to obtain the final APS score

## Output Format

For the entire dataset, the scorer returns:

```json
{
  "score": 0.456,
  "num_samples": 1000,
  "num_pairs": 499500,
  "total_possible_pairs": 499500,
  "is_sampled": false,
  "similarity_metric": "cosine",
  "max_workers": 8
}
```

- `score`: The Average Pairwise Similarity (APS) score for the dataset
- `num_samples`: Total number of samples in the dataset
- `num_pairs`: Number of pairs actually computed (may be less than `total_possible_pairs` if sampled)
- `total_possible_pairs`: Total number of possible unique pairs: N×(N-1)/2
- `is_sampled`: Whether pairs were randomly sampled (true if `sample_pairs` was used)
- `similarity_metric`: Similarity metric used ("cosine", "euclidean", "manhattan", "dot_product", or "pearson")
- `max_workers`: Number of parallel workers used for computation
- `sample_pairs`: (Optional) Number of pairs sampled if `is_sampled=true`
- `warning`: (Optional) Warning message if dataset has insufficient samples (< 2) or mismatched embedding counts

## Citation

```bibtex
@article{yu2023metamath,
  title={Metamath: Bootstrap your own mathematical questions for large language models},
  author={Yu, Longhui and Jiang, Weisen and Shi, Han and Yu, Jincheng and Liu, Zhengying and Zhang, Yu and Kwok, James T and Li, Zhenguo and Weller, Adrian and Liu, Weiyang},
  journal={arXiv preprint arXiv:2309.12284},
  year={2023}
}
```
