# NovelSumScorer

## Overview

The **NovelSum Scorer** is a dataset-level diversity metric designed to measure data diversity for instruction tuning. Proposed in [Yang et al., 2025](https://aclanthology.org/2025.acl-long.908/), this method addresses the fundamental problem of precisely defining and measuring data diversity in instruction-tuning datasets. Unlike per-sample scoring methods, NovelSum evaluates the entire dataset holistically by considering both inter-sample differences and information density in the sample space.

Through systematic analysis of existing diversity measurement methods, the authors found that a reliable diversity measure should properly account for sample-level "novelty." Experiments demonstrate that NovelSum achieves a **0.97 correlation** with instruction-tuned model performance, making it a valuable metric for guiding data engineering practices and diversity-oriented data selection strategies.

## Metric Definition:

* **Definition:** 

  NovelSum is computed as:
  
  \[ \text{NovelSum} = \frac{1}{N} \sum_{i=1}^{N} \text{WeightedAvg}(d_i \odot \rho) \]
  
  Where:
  - \( N \) is the number of samples in the dataset
  - \( d_i \) represents the distance vector from sample \( i \) to all other samples
  - \( \rho \) represents the local density of each sample
  - \( \odot \) denotes element-wise multiplication
  - WeightedAvg applies inverse-rank weighting to prioritize closer neighbors

* **Explanation:** NovelSum measures dataset diversity by considering two key factors:
  
  1. **Inter-sample Differences:** Captured through pairwise cosine distances between sample embeddings
  2. **Information Density:** Computed via local density estimation using k-nearest neighbors
  
  * A **higher NovelSum score** indicates **greater diversity**, suggesting the dataset contains more distinct and informative samples distributed across the feature space.
  * A **lower NovelSum score** suggests **lower diversity**, indicating samples are clustered or redundant.

## YAML Configuration

```yaml
name: NovelSumScorer
embedding_path: /path/to/embeddings.npy
dense_ref_path: /path/to/reference/embeddings/
max_workers: 8
density_powers: [0, 0.25, 0.5]
neighbors: [5, 10]
distance_powers: [0, 1, 2]
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"NovelSumScorer"` | Identifier for the scorer |
| `embedding_path` | string | *required* | Path to the embeddings file (`.npy` format) for the dataset to be evaluated. The embeddings should be a 2D numpy array with shape `(num_samples, embedding_dim)`, where each row corresponds to one data sample. |
| `dense_ref_path` | string | Directory containing `embedding_path` | Path to reference embeddings used for local density computation. Can be either a directory containing multiple `.npy` files (all will be loaded and concatenated) or a single `.npy` file. The reference data is used to build a Faiss index for computing local density in the embedding space. |
| `max_workers` | int | CPU count | Number of worker processes for parallel computation of the cosine distance matrix. If not specified or invalid, defaults to the number of CPU cores. Higher values can speed up computation but require more memory. |
| `density_powers` | list[float] | `[0, 0.25, 0.5]` | List of power values for density weighting. Controls how density affects the novelty calculation: `0` for no density weighting (uniform), `> 0` for higher density influence. |
| `neighbors` | list[int] | `[5, 10]` | List of k-nearest neighbor values for density estimation. Determines the neighborhood size used to compute local density. |
| `distance_powers` | list[float] | `[0, 1, 2]` | List of power values for distance weighting in the final aggregation. Controls the weighting scheme when averaging distances: `0` for uniform weighting, `1` for linear inverse-rank weighting, `2` for quadratic inverse-rank weighting. |

## Underlying Model

NovelSumScorer does **not require a specific language model** for inference. Instead, it operates on **pre-computed embeddings** that must be generated in advance using an embedding model of your choice. 

**Note**: The embeddings must be saved as a NumPy `.npy` file with shape (N, D) where N matches the number of samples in your dataset and D is the embedding dimension. The order of embeddings must correspond to the order of samples in your dataset file.

## Generating Embeddings

To generate the required embedding file for NovelSumScorer, you can use the provided `embed.py` script located at:

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

This output file can be directly used as the `embedding_path` parameter in the NovelSumScorer configuration.

## Scoring Process

1. **Load Embeddings:** The scorer loads the pre-computed embeddings from `embedding_path` (shape: `[num_samples, embedding_dim]`).

2. **Build Faiss Index:** A Faiss index is constructed using the reference embeddings from `dense_ref_path`. This index enables efficient k-nearest neighbor search for local density computation.

3. **Compute Distance Matrix:** A cosine distance matrix is computed between all pairs of samples in the dataset using the formula:
   \[ d_{ij} = 1 - \cos(\mathbf{e}_i, \mathbf{e}_j) \]
   where \( \mathbf{e}_i \) and \( \mathbf{e}_j \) are embedding vectors.

4. **Compute Local Density:** For each sample, local density is estimated using k-nearest neighbors:
   \[ \rho_i = \frac{1}{(\bar{d}_i + \epsilon)^p} \]
   where \( \bar{d}_i \) is the average distance to k-nearest neighbors, \( \epsilon \) is a small regularization term, and \( p \) is the density power.

5. **Calculate NovelSum:** For each hyperparameter combination (density power, neighbor count, distance power):
   - Weight the distance matrix by local densities: \( D' = D \odot \rho \)
   - Apply inverse-rank weighted averaging along each row
   - Compute the mean across all samples

6. **Return Results:** The scorer returns NovelSum scores for all hyperparameter combinations, along with the average cosine distance.

## Output Format

The `evaluate()` method returns a dictionary with the following structure:

```python
{
    'num_samples': 1000,
    'cos_distance': 0.654321,
    'neighbor_5_density_0_distance_0': 0.123456,
    'neighbor_5_density_0_distance_1': 0.234567,
    'neighbor_5_density_0_distance_2': 0.345678,
    'neighbor_5_density_0.25_distance_0': 0.456789,
    # ... (one entry per hyperparameter combination)
}
```

### Output Keys

* **`num_samples`**: Integer indicating the number of samples in the evaluated dataset.

* **`cos_distance`**: Float representing the average pairwise cosine distance across all samples. This serves as a baseline diversity measure.

* **`neighbor_{nb}_density_{dp}_distance_{distp}`**: Float representing the NovelSum score computed with:
  - `nb`: Number of neighbors for density estimation (from `neighbors` config)
  - `dp`: Density power (from `density_powers` config)
  - `distp`: Distance power for weighted averaging (from `distance_powers` config)
  
  Higher values indicate greater diversity under that specific configuration.

### Recommended Configuration

Based on the paper's findings, the configuration with **neighbor=10, density_power=0.5, distance_power=1** (i.e., `neighbor_10_density_0.5_distance_1`) typically achieves the strongest correlation with model performance and is recommended as the primary diversity metric.

## Citation

```bibtex
@inproceedings{yang2025measuring,
  title={Measuring data diversity for instruction tuning: A systematic analysis and a reliable metric},
  author={Yang, Yuming and Nan, Yang and Ye, Junjie and Dou, Shihan and Wang, Xiao and Li, Shuo and Lv, Huijie and Gui, Tao and Zhang, Qi and Huang, Xuan-Jing},
  booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={18530--18549},
  year={2025}
}
```
