# KNN Scorer

## Overview

The **KNN Scorer** (K-Nearest Neighbors Scorer) is an embedding-based evaluation method that measures the **local density** and **uniqueness** of data points in the embedding space. This approach quantifies data diversity by computing the average distance to each sample's k-nearest neighbors. Originally proposed by Google Research as a high-fidelity data selection strategy, KNN scoring helps identify samples that are either centrally located (redundant) or peripherally positioned (unique/outlier) within the dataset's semantic space.

The core intuition is that samples with **larger K-nearest neighbor distances** are more unique or isolated in the embedding space, potentially representing diverse or valuable examples, while samples with **smaller distances** are surrounded by similar data and may be redundant.

## Metric Definition:

* **Definition:** 

  For a given data point \( x_i \) with embedding \( e_i \), the KNN distance is calculated as:

  \[
  \text{KNN\_Distance}(x_i) = \frac{1}{k} \sum_{j=1}^{k} d(e_i, e_{n_j})
  \]

  where \( e_{n_j} \) represents the embedding of the j-th nearest neighbor (excluding \( x_i \) itself), and \( d(\cdot, \cdot) \) is a distance metric (e.g., Euclidean, cosine, or Manhattan distance).

* **Explanation:** This metric quantifies how isolated or central a data point is within its local neighborhood:
  
  * A **higher KNN distance** indicates that the sample is **far from its neighbors**, suggesting it is **unique, diverse, or potentially an outlier** in the dataset.
  * A **lower KNN distance** indicates that the sample is **close to many similar samples**, suggesting it is **redundant or representative of a dense cluster**.

## YAML Configuration

```yaml
name: KNNScorer
embedding_path: /path/to/embeddings.npy
k: 5
distance_metric: euclidean
max_workers: 8
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"KNNScorer"` | Identifier for the scorer |
| `embedding_path` | string | **required** | Path to the pre-computed embeddings file in `.npy` format. This file should be a 2D NumPy array with shape `(num_samples, embedding_dim)`, where each row corresponds to the embedding of a data sample in the dataset. **The order of embeddings must match the order of samples in the dataset.** |
| `k` | integer | `5` | Number of nearest neighbors to consider for distance calculation. If `k` is greater than or equal to the dataset size, it will be automatically adjusted to `num_samples - 1` |
| `distance_metric` | string | `"euclidean"` | Distance metric for computing neighbor distances. Supported values: `"euclidean"` (Euclidean/L2 distance), `"cosine"` (cosine distance = 1 - cosine similarity), `"manhattan"` (Manhattan/L1 distance) |
| `max_workers` | integer | CPU cores | Number of parallel worker processes for scoring. Higher values speed up processing but require more memory |

## Underlying Model

KNNScorer does **not require a specific language model** for inference. Instead, it operates on **pre-computed embeddings** that must be generated in advance using an embedding model of your choice. 

**Note**: The embeddings must be saved as a NumPy `.npy` file with shape (N, D) where N matches the number of samples in your dataset and D is the embedding dimension. The order of embeddings must correspond to the order of samples in your dataset file.

## Generating Embeddings

To generate the required embedding file for KNNScorer, you can use the provided `embed.py` script located at:

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

This output file can be directly used as the `embedding_path` parameter in the KNNScorer configuration.

## Scoring Process

The KNN Scorer follows these steps to evaluate each data sample:

1. **Load Embeddings**: Load the pre-computed embedding matrix from the specified `.npy` file. The embeddings should be a 2D array with shape `(num_samples, embedding_dim)`.

2. **Build KNN Index**: Construct a K-Nearest Neighbors model using the specified distance metric (e.g., Euclidean, cosine, or Manhattan). The KNN index is built on all embeddings to enable efficient neighbor searches.

3. **Find K-Nearest Neighbors**: For each data point \( x_i \):
   - Query the KNN model to find the \( k+1 \) nearest neighbors (including the point itself)
   - Exclude the first neighbor (which is the point itself) to obtain the \( k \) nearest neighbors

4. **Calculate Average Distance**: Compute the mean distance to the \( k \) nearest neighbors:
   
   \[
   \text{score}_i = \frac{1}{k} \sum_{j=1}^{k} d(e_i, e_{n_j})
   \]

5. **Parallel Processing**: The scorer uses multiprocessing with configurable worker processes to efficiently handle large datasets.

## Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 0.524
}
```

- `id`: The unique identifier of the data sample, extracted from the `"id"` field in the input dataset
- `score`: The average K-nearest neighbor distance for this sample. Higher scores indicate greater uniqueness or isolation in the embedding space, while lower scores indicate redundancy or centrality within a dense cluster

**Interpretation:**

- **High scores** (e.g., > 0.8): Sample is semantically distant from neighbors → potentially unique, diverse, or outlier
- **Low scores** (e.g., < 0.3): Sample is semantically close to neighbors → potentially redundant or representative of common patterns

The specific threshold values depend on the embedding model, distance metric, and dataset characteristics.

## Citation

```bibtex
@misc{google2025highfidelity,
  title        = {Achieving 10000x Training Data Reduction with High-Fidelity Labels},
  author       = {{Google Research}},
  howpublished = {\url{https://research.google/blog/achieving-10000x-training-data-reduction-with-high-fidelity-labels/}},
  note         = {Accessed: 2025-02-xx}
}
```

