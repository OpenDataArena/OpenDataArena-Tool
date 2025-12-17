# ClusterInertiaScorer

## Overview

The **Cluster Inertia Scorer** is an embedding-based evaluation tool designed to measure the **diversity** and **dispersion** of datasets through cluster inertia analysis. Unlike sample-level scorers, this scorer evaluates the entire dataset holistically by calculating the sum of distances from all data points to their assigned cluster centroids. This metric provides insights into how tightly or loosely data samples are grouped, which serves as an indicator of dataset diversity and coverage.

Higher inertia values suggest greater data dispersion and diversity, while lower values indicate tighter clustering and more homogeneous data distribution.

## Metric Definition:

* **Definition:** 

  Given a dataset with embeddings and clustering results, the scorer computes:
  
  ```
  Cluster_Inertia = Σ(Σ(distance(x, c_i) for x in cluster_i) for all clusters i)
  ```
  
  Where:
  - `x` represents individual data points
  - `c_i` represents the centroid of cluster `i`
  - `distance(·, ·)` is a configurable distance function (e.g., cosine, Euclidean)

* **Explanation:** Cluster inertia quantifies the overall compactness of data clusters by summing the distances between all samples and their respective cluster centroids.
  
  * A **higher Cluster Inertia score** indicates that data points are **more dispersed** from their cluster centers, suggesting **greater diversity** and broader coverage across the feature space.
  * A **lower Cluster Inertia score** suggests that data points are **tightly grouped** around their centroids, indicating **lower diversity** and more concentrated data distribution.

## YAML Configuration

```yaml
name: ClusterInertiaScorer
embedding_path: /path/to/embeddings.npy
cluster_centroids_path: /path/to/centroids.npy
cluster_labels_path: /path/to/labels.npy
distance_metric: cosine
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"ClusterInertiaScorer"` | Identifier for the scorer |
| `embedding_path` | string | (required) | Path to the embeddings file in `.npy` format. The file should contain a 2D numpy array of shape `(num_samples, embedding_dim)`, where each row represents the embedding vector of a data sample |
| `cluster_centroids_path` | string | (required) | Path to the cluster centroids file in `.npy` format. The file should contain a 2D numpy array of shape `(num_clusters, embedding_dim)`, where each row represents the centroid vector of a cluster |
| `cluster_labels_path` | string | (required) | Path to the cluster labels file in `.npy` format. The file should contain a 1D numpy array of shape `(num_samples,)`, where each element indicates the cluster assignment (0 to num_clusters-1) for the corresponding data sample |
| `distance_metric` | string | `"cosine"` | Distance metric used to compute distances between embeddings and centroids. Supported metrics: `"cosine"` (cosine distance), `"euclidean"` (L2 norm), `"squared_euclidean"` (squared L2), `"manhattan"` (L1 norm) |
| `max_workers` | int | CPU count | Number of worker processes for parallel computation. Defaults to the system's CPU count if not specified or invalid |

## Underlying Model

The Cluster Inertia Scorer **does not require a language model** for evaluation. Instead, it operates on **pre-computed embeddings** and **clustering results**. Users need to:

1. Generate embeddings for their dataset using any embedding model (e.g., sentence transformers, text embeddings from LLMs)
2. Perform clustering analysis (e.g., K-means, DBSCAN, hierarchical clustering) to obtain cluster centroids and labels
3. Save these artifacts as `.npy` files for the scorer to load

This design allows flexibility in choosing embedding models and clustering algorithms based on specific use cases and data characteristics.

## Generating Embeddings

To generate the required embedding file for ClusterInertiaScorer, you can use the provided `embed.py` script located at:

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

This output file can be directly used as the `embedding_path` parameter in the ClusterInertiaScorer configuration.

## Scoring Process

1. **Validation Phase:**
   - Verify that all required files (embeddings, centroids, labels) exist and have correct formats (`.npy`)
   - Validate distance metric parameter, defaulting to `"cosine"` if invalid or unspecified

2. **Setup Phase:**
   - Load embeddings from `embedding_path` → shape: `(num_samples, embedding_dim)`
   - Load cluster centroids from `cluster_centroids_path` → shape: `(num_clusters, embedding_dim)`
   - Load cluster labels from `cluster_labels_path` → shape: `(num_samples,)`
   - Verify dimensional consistency between embeddings and centroids
   - Verify that the number of labels matches the number of embeddings
   - Initialize the distance function based on the specified metric

3. **Inertia Computation:**
   - For each cluster `i` from 0 to `num_clusters - 1`:
     - Identify all data points assigned to cluster `i`
     - Retrieve the corresponding centroid `c_i`
     - Compute the distance from each point to `c_i` using the specified distance metric
     - Sum all distances to get the cluster-specific inertia
   - Cluster inertias are computed in parallel using `ProcessPoolExecutor` with `max_workers` processes for improved performance
   - Aggregate all cluster inertias to compute the total dataset inertia
   - Calculate average inertia per sample by dividing total inertia by the number of samples

4. **Result Reporting:**
   - Return comprehensive statistics including total inertia, average inertia, cluster sizes, and per-cluster inertia values

**Note:** Unlike most scorers that operate on individual samples, the `ClusterInertiaScorer` evaluates the **entire dataset** as a single unit. The `score_item()` method is intentionally not implemented; users should call `evaluate(dataset)` instead.

## Output Format

The `evaluate()` method returns a dictionary containing the following keys:

```json
{
  "total_inertia": 1234.5678,
  "avg_inertia_per_sample": 0.1234,
  "num_samples": 10000,
  "num_clusters": 50,
  "distance_metric": "cosine",
  "max_workers": 8,
  "cluster_sizes": {"0": 150, "1": 230, "...": "..."},
  "cluster_inertias": {"0": 45.67, "1": 78.90, "...": "..."}
}
```

- `total_inertia`: The sum of distances from all samples to their assigned cluster centroids. Higher values indicate greater overall data dispersion
- `avg_inertia_per_sample`: Normalized inertia metric calculated as `total_inertia / num_samples`. Useful for comparing datasets of different sizes
- `num_samples`: Number of data samples included in the evaluation. May differ from the original dataset size if there are mismatches between data and embeddings
- `num_clusters`: Number of distinct clusters in the clustering solution, determined by the dimensionality of the centroids file
- `distance_metric`: The distance function used for computation (e.g., `"cosine"`, `"euclidean"`)
- `max_workers`: Number of worker processes used for parallel computation
- `cluster_sizes`: Dictionary showing the distribution of samples across clusters. Format: `{cluster_id: sample_count}`. Empty clusters will have a size of 0
- `cluster_inertias`: Dictionary showing the inertia contribution of each cluster. Format: `{cluster_id: inertia_value}`. Useful for identifying which clusters contribute most to overall data diversity

## Citation

```bibtex
@inproceedings{du2019boosting,
  title={Boosting dialog response generation},
  author={Du, Wenchao and Black, Alan W},
  booktitle={Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
  year={2019}
}
```

