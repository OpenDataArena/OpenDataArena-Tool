# FacilityLocationScorer

## Overview

The **Facility Location Scorer** is an embedding-based evaluation tool designed to assess the **coverage quality** of a selected data subset over the full dataset. Inspired by the classical Facility Location problem in operations research, this scorer treats the selected subset as "facilities" and the full dataset as "customers," measuring how well the facilities serve all customers in the embedding space.

This metric is particularly useful for evaluating data selection strategies, where the goal is to choose a representative subset that maximally covers the diversity of the entire dataset. A lower Facility Location score indicates better coverage, meaning the selected subset can adequately represent the full dataset's distribution.

## Metric Definition:

* **Definition:** 

  $$M_{FL}(X) = \sum_{x_j \in X_{all}} \min_{x_i \in X} d(x_j, x_i)$$

  Where:
  - \(X_{all}\) is the full dataset
  - \(X\) is the selected subset
  - \(d(x_j, x_i)\) is the distance between data points \(x_j\) and \(x_i\) in the embedding space

* **Explanation:**
  * A **lower score** indicates that the selected subset has **better coverage** of the full dataset, as every data point in the full dataset has at least one nearby representative in the subset.
  * A **higher score** suggests that the subset coverage is **inadequate**, with many data points in the full dataset being far from any point in the selected subset.

## YAML Configuration

```yaml
name: FacilityLocationScorer
embedding_path: path/to/full_dataset_embeddings.npy
subset_embeddings_path: path/to/subset_embeddings.npy
distance_metric: euclidean
max_workers: 8
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"FacilityLocationScorer"` | Identifier for the scorer |
| `embedding_path` | string | (required) | Path to the `.npy` file containing embeddings of the **full dataset**. This serves as the reference background for evaluating coverage quality. |
| `subset_embeddings_path` | string | (required) | Path to the `.npy` file containing embeddings of the **subset to be evaluated**. This subset corresponds to the data samples in `input_path`. |
| `distance_metric` | string | `"euclidean"` | The distance metric used for computing distances between embeddings. Available options: `"euclidean"` (Standard Euclidean distance), `"squared_euclidean"` (Squared Euclidean distance, faster), `"manhattan"` (Manhattan/L1 distance), `"cosine"` (Cosine distance: 1 - cosine similarity) |
| `max_workers` | integer | CPU count | Number of parallel workers for multiprocessing. Higher values can accelerate computation for large datasets. |

## Underlying Model

FacilityLocationScorer does **not require a specific language model** for inference. Instead, it operates on **pre-computed embeddings** that must be generated in advance using an embedding model of your choice. 

**Note**: The embeddings must be saved as a NumPy `.npy` file with shape (N, D) where N matches the number of samples in your dataset and D is the embedding dimension. The order of embeddings must correspond to the order of samples in your dataset file.

## Generating Embeddings

To generate the required embedding file for FacilityLocationScorer, you can use the provided `embed.py` script located at:

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

This output file can be directly used as the `embedding_path` parameter in the FacilityLocationScorer configuration.

## Scoring Process

The evaluation process consists of the following steps:

1. **Load Embeddings:**
   - Load the full dataset embeddings from `embedding_path` (shape: `[N, D]`, where `N` is the total number of samples and `D` is the embedding dimension).
   - Load the subset embeddings from `subset_embeddings_path` (shape: `[M, D]`, where `M` is the subset size).

2. **Validate Dimensions:**
   - Verify that both embeddings have the same dimension `D`.
   - Ensure the number of subset embeddings matches the number of lines in the input dataset.

3. **Parallel Distance Computation:**
   - For each point in the full dataset, compute the distance to all points in the subset using the specified distance metric.
   - Extract the minimum distance for each full dataset point.
   - Use `ProcessPoolExecutor` with `max_workers` processes to parallelize computation for efficiency.

4. **Aggregate Statistics:**
   - Sum all minimum distances to obtain the Facility Location score.
   - Compute additional statistics: average, maximum, median, and standard deviation of minimum distances.

## Output Format

For each evaluation, the scorer returns:

```json
{
  "facility_location_score": 1234.56,
  "avg_min_distance": 0.123,
  "max_min_distance": 2.345,
  "median_min_distance": 0.098,
  "std_min_distance": 0.234,
  "num_samples": 10000,
  "num_subset_samples": 1000,
  "distance_metric": "euclidean",
  "subset_ratio": 0.1
}
```

- `facility_location_score`: Sum of all minimum distances (primary metric). Lower values indicate better subset coverage.
- `avg_min_distance`: Average minimum distance. Provides an average sense of how close the subset is to the full dataset.
- `max_min_distance`: Maximum minimum distance (worst-case coverage). Identifies the worst-case scenario.
- `median_min_distance`: Median minimum distance. Offers a robust central tendency measure, less affected by outliers.
- `std_min_distance`: Standard deviation of minimum distances. Indicates the variability in coverage quality.
- `num_samples`: Number of samples in the full dataset.
- `num_subset_samples`: Number of samples in the subset.
- `distance_metric`: Distance metric used for computation.
- `subset_ratio`: Ratio of subset size to full dataset size.

## Citation

```bibtex
@book{farahani2009facility,
  title={Facility location: concepts, models, algorithms and case studies},
  author={Farahani, Reza Zanjirani and Hekmatfar, Masoud},
  year={2009},
  publisher={Springer Science \& Business Media}
}
```
