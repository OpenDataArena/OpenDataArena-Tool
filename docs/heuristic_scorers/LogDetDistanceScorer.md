# LogDetDistanceScorer

## Overview

The **Log-Det Distance Scorer** is a diversity measurement tool for instruction tuning datasets based on the determinantal point process (DPP) framework. Proposed in [Wang et al., 2024](https://arxiv.org/abs/2402.02318), this method quantifies dataset diversity by computing the log-determinant of the cosine similarity matrix constructed from data embeddings. Unlike heuristic diversity measures (e.g., counting number of tasks), Log-Det Distance provides a principled, geometry-based metric that correlates with downstream instruction-following performance and can inform data selection strategies.

## Metric Definition:

* **Definition:** `Log-Det Distance = log(det(S))`

  where `S` is the `N × N` cosine similarity matrix computed from the embeddings of `N` data samples.

* **Explanation:** The log-determinant measures the "volume" spanned by the data embeddings in the feature space:
  
  * A **higher Log-Det value** indicates that the data samples are **more diverse** and span a larger volume in the embedding space, suggesting rich coverage of different instruction patterns.
  * A **lower Log-Det value** indicates **high similarity** among samples and **low diversity**, suggesting redundant or homogeneous data.

* **Mathematical Properties:**
  
  * The similarity matrix `S` should be positive semi-definite (all eigenvalues ≥ 0) to ensure `det(S) ≥ 0`, making `log(det(S))` mathematically valid.
  * Ridge regularization can be applied to improve numerical stability: `S' = S + α·I`, where `α` is a small positive constant.

## YAML Configuration

```yaml
name: LogDetDistanceScorer
embedding_path: path/to/embeddings.npy
max_workers: 8
ridge_alpha: 1e-10
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"LogDetDistanceScorer"` | Identifier for the scorer |
| `embedding_path` | string | **(required)** | Path to the pre-computed embedding file in `.npy` format. The embeddings should be a NumPy array of shape `[num_samples, embedding_dim]`, where each row corresponds to one data sample in the dataset |
| `max_workers` | integer | CPU cores | Number of parallel worker processes for computing the similarity matrix. For datasets with fewer than 5,000 samples, a vectorized method is automatically used instead |
| `ridge_alpha` | float | `1e-10` | Ridge regularization parameter added to the diagonal of the similarity matrix for numerical stability (`S' = S + α·I`). Can be specified in scientific notation |


## Underlying Model

LogDetDistanceScorer does **not require a specific language model** for inference. Instead, it operates on **pre-computed embeddings** that must be generated in advance using an embedding model of your choice. 

**Note**: The embeddings must be saved as a NumPy `.npy` file with shape (N, D) where N matches the number of samples in your dataset and D is the embedding dimension. The order of embeddings must correspond to the order of samples in your dataset file.

## Generating Embeddings

To generate the required embedding file for LogDetDistanceScorer, you can use the provided `embed.py` script located at:

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

This output file can be directly used as the `embedding_path` parameter in the LogDetDistanceScorer configuration.

## Scoring Process

The Log-Det Distance Scorer follows these steps:

1. **Load Pre-computed Embeddings:** The scorer loads the embedding file specified in `embedding_path`. It validates that the number of embeddings matches the number of samples in the dataset.

2. **Compute Cosine Similarity Matrix:** A cosine similarity matrix `S` of size `N × N` is computed, where each entry `S[i,j]` represents the cosine similarity between embeddings of sample `i` and sample `j`:

   ```
   S[i,j] = (emb[i] · emb[j]) / (||emb[i]|| × ||emb[j]||)
   ```

   * For datasets with < 5,000 samples, a vectorized computation method is used
   * For larger datasets, parallel processing with multiple workers is employed

3. **Apply Ridge Regularization:** To ensure numerical stability, a small regularization term is added to the diagonal:

   ```
   S' = S + α·I
   ```

4. **Check Matrix Properties:** The scorer computes eigenvalues to verify that the similarity matrix is positive semi-definite, which is a requirement for valid log-determinant computation.

5. **Compute Log-Determinant:** Using NumPy's `slogdet` function (numerically stable method), the log-determinant is computed:

   ```
   sign, logdet = slogdet(S')
   Log-Det Distance = logdet (if sign > 0)
   ```

6. **Return Diversity Score:** The final Log-Det value serves as the diversity metric for the entire dataset, along with detailed statistics about the similarity matrix and eigenvalues.

## Output Format

The scorer returns a **list containing a single dictionary** (since Log-Det is a dataset-level metric, not a per-sample metric):

```json
{
    "log_det": 1234.56,
    "sign": 1,
    "is_valid": true,
    "is_positive_definite": true,
    "is_positive_semidefinite": true,
    "num_samples": 10000,
    "embedding_dimension": 768,
    "similarity_metric": "cosine",
    "eigenvalue_stats": {
        "min": 0.000123,
        "max": 1.234567,
        "num_negative": 0
    },
    "similarity_matrix_stats": {
        "min": -0.12,
        "max": 1.00,
        "mean": 0.34,
        "std": 0.15,
        "diagonal_mean": 1.000001
    }
}
```

- `log_det`: The primary diversity metric. Higher values indicate greater diversity. If the determinant is zero or negative, this may be `None` or `-inf`
- `sign`: Indicates whether the determinant is positive (1), zero (0), or negative (-1). Valid similarity matrices should have `sign = 1`
- `is_valid`: Boolean flag indicating whether the log-det computation succeeded and the result is mathematically valid
- `is_positive_definite`: Whether all eigenvalues > 0
- `is_positive_semidefinite`: Whether all eigenvalues ≥ 0. A positive semi-definite matrix ensures the log-det is well-defined
- `num_samples`: Number of samples in the dataset
- `embedding_dimension`: Dimension of the embeddings
- `similarity_metric`: Similarity metric used (cosine)
- `eigenvalue_stats`: Statistics about the eigenvalues of the similarity matrix, useful for diagnosing numerical issues or understanding the geometry of the data
- `similarity_matrix_stats`: Statistics about the similarity matrix itself, helping to understand the distribution of pairwise similarities in the dataset
- `warning` (optional): If present, contains a warning message about issues during computation (e.g., singular matrix, negative determinant)
- `log_det_is_inf` (optional): If true, indicates that the log-det value is infinite

## Citation

```bibtex
@article{wang2024diversity,
  title={Diversity measurement and subset selection for instruction tuning datasets},
  author={Wang, Peiqi and Shen, Yikang and Guo, Zhen and Stallone, Matthew and Kim, Yoon and Golland, Polina and Panda, Rameswar},
  journal={arXiv preprint arXiv:2402.02318},
  year={2024}
}
```

