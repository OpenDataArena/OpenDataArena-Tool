# RadiusScorer

## Overview

The **Radius Scorer** is a diversity metric designed to quantify the distributional spread of a dataset in its embedding space. Unlike per-sample scorers, Radius operates at the **dataset level**, computing a single aggregate score that reflects how widely distributed the data points are across all embedding dimensions, proposed by [Lai et al., 2020](https://arxiv.org/abs/2003.08529)

## Metric Definition:

* **Definition:**

  Radius is computed as the **geometric mean of standard deviations** across all embedding dimensions:
  
  \[ \text{Radius} = \left(\prod_{i=1}^{n} \sigma_i\right)^{1/n} = \exp\left(\frac{1}{n}\sum_{i=1}^{n} \log(\sigma_i)\right) \]
  
  where \( \sigma_i \) is the standard deviation of all data points along the \( i \)-th embedding dimension, and \( n \) is the total number of dimensions.

* **Explanation:**
  * A **higher Radius value** indicates that data points are **more widely distributed** across the embedding space, suggesting **higher diversity**.
  * A **lower Radius value** indicates that data points are **more concentrated**, suggesting **lower diversity** or higher homogeneity.
  * The geometric mean is preferred over the arithmetic mean because it is less sensitive to outlier dimensions and better captures the overall distributional balance.

## YAML Configuration

```yaml
name: RadiusScorer
embedding_path: path/to/embeddings.npy
max_workers: 8
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"RadiusScorer"` | Identifier for the scorer, used in logs and progress displays |
| `embedding_path` | string | required | Path to a pre-computed embedding file in `.npy` format. The file should contain a 2D NumPy array of shape `(num_samples, embedding_dim)`, where each row corresponds to the embedding vector of a data sample |
| `max_workers` | integer | Number of CPU cores | Number of worker processes for parallel computation of dimension-wise standard deviations. Parallel processing is automatically enabled when `max_workers > 1` and `embedding_dim > max_workers * 10`. Set to `1` to disable parallelization |

## Underlying Model

The Radius Scorer **does not require a specific language model** for scoring. Instead, it operates on **pre-computed embeddings** that must be generated beforehand using any embedding model of your choice, such as:

- **Sentence Transformers** (e.g., `sentence-transformers/all-MiniLM-L6-v2`)
- **OpenAI Embeddings** (e.g., `text-embedding-ada-002`)
- **Custom embedding models** fine-tuned for your domain

The embeddings should be saved as a `.npy` file with shape `(num_samples, embedding_dim)` where each row represents the embedding of a corresponding data sample.

## Scoring Process

The Radius Scorer computes dataset-level diversity through the following steps:

1. **Load Pre-computed Embeddings**: The scorer loads the embedding matrix from the specified `.npy` file.

2. **Validate Dataset-Embedding Alignment**: The number of lines in the dataset is compared with the number of rows in the embedding matrix. If they do not match, a warning is issued and the minimum count is used.

3. **Compute Dimension-wise Standard Deviations**:
   - For each embedding dimension \( i \), compute the standard deviation \( \sigma_i \) across all data samples.
   - If `max_workers > 1` and the embedding dimension is sufficiently large, this computation is parallelized across multiple processes using `ProcessPoolExecutor`.

4. **Handle Zero-Variance Dimensions**: If any dimension has zero standard deviation (all values identical), it is replaced with a small epsilon value (`1e-10`) to avoid numerical issues in the geometric mean.

5. **Compute Geometric Mean**: Using the log-transform trick to avoid numerical overflow, the Radius is computed as:
   \[
   \text{Radius} = \exp\left(\frac{1}{n}\sum_{i=1}^{n} \log(\sigma_i)\right)
   \]

6. **Return Comprehensive Statistics**: In addition to the Radius score, the scorer returns additional statistics including arithmetic mean, min/max/median standard deviations, and metadata about the dataset.

**Note**: Since Radius is a dataset-level metric, the `score_item()` method is **not implemented**. Always use the `evaluate()` method to compute the Radius for the entire dataset.

## Output Format

The scorer returns a dictionary containing the following fields:

```json
{
  "radius": 0.1234,
  "geometric_mean_std": 0.1234,
  "arithmetic_mean_std": 0.1456,
  "min_std": 0.0012,
  "max_std": 0.8765,
  "median_std": 0.0987,
  "num_samples": 10000,
  "embedding_dimension": 768,
  "zero_std_dimensions": 0
}
```

- `radius`: The primary diversity metric, computed as the geometric mean of all dimension-wise standard deviations. Higher values indicate greater diversity
- `geometric_mean_std`: Duplicate of `radius`, provided for clarity in output interpretation
- `arithmetic_mean_std`: The arithmetic mean of standard deviations, included for comparison with the geometric mean. Generally less robust than the geometric mean
- `min_std`: The smallest standard deviation among all dimensions, useful for identifying dimensions with extremely low variance
- `max_std`: The largest standard deviation among all dimensions, useful for identifying dimensions with extremely high variance
- `median_std`: The median standard deviation, providing a robust central tendency measure
- `num_samples`: Total number of data samples evaluated
- `embedding_dimension`: The dimensionality of the embedding vectors
- `zero_std_dimensions`: Number of dimensions where all values are identical (zero variance). High counts may indicate embedding quality issues

## Citation

```bibtex
@article{lai2020diversity,
  title={Diversity, density, and homogeneity: Quantitative characteristic metrics for text collections},
  author={Lai, Yi-An and Zhu, Xuan and Zhang, Yi and Diab, Mona},
  journal={arXiv preprint arXiv:2003.08529},
  year={2020}
}
```
