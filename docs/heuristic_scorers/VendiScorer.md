# VendiScorer

## Overview

The **Vendi Score** is a diversity evaluation metric for machine learning that measures the intrinsic diversity of a dataset without requiring any reference distribution or pre-trained classifier. Proposed by Friedman and Dieng (2023), the Vendi Score extends concepts from ecology and quantum statistical mechanics to evaluate the effective number of unique elements in a sample.

Unlike reference-based metrics (e.g., FID) or label-dependent metrics (e.g., Inception Score), the Vendi Score is a **reference-free, general-purpose diversity metric** that can be applied to any domain where similarity between samples can be defined. It takes as input a collection of embeddings and a user-specified similarity function, making it highly flexible for evaluating dataset diversity across different modalities (text, images, molecules, etc.).

## Metric Definition:

* **Definition:** 

  The Vendi Score is defined as the exponential of the Shannon entropy of the eigenvalues of a similarity matrix **K**, where **K**<sub>ij</sub> represents the similarity between samples *i* and *j*:

  ```
  VS = exp(H(λ))
  ```
  
  where H(λ) is the Shannon entropy of the normalized eigenvalues λ of the similarity matrix **K**.

* **Explanation:** The Vendi Score can be interpreted as the **effective number of unique elements** in the dataset:
  
  * A **higher Vendi Score** indicates **greater diversity** in the dataset, suggesting more distinct and varied samples.
  * A **lower Vendi Score** indicates **lower diversity**, suggesting samples are more similar to each other or repetitive.
  * The minimum value is 1 (all samples identical), and the maximum value approaches *n* (all samples completely dissimilar), where *n* is the number of samples.

* **Key Advantages:**
  
  * **Reference-free:** Does not require any reference dataset or distribution
  * **Label-independent:** Does not require discrete labels or categories
  * **Flexible:** Allows user-defined similarity functions to capture different notions of diversity
  * **Interpretable:** Can be understood as an effective sample count

## YAML Configuration

```yaml
name: VendiScorer
embedding_path: path/to/embeddings.npy
similarity_metric: cosine
max_workers: 8
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"VendiScorer"` | Identifier for the scorer |
| `embedding_path` | string | (required) | Path to a pre-computed embedding file in `.npy` format. The file should contain a 2D numpy array of shape `(num_samples, embedding_dim)`, where each row represents the embedding vector of a data sample |
| `similarity_metric` | string | `"cosine"` | The similarity function used to compute pairwise similarities between embeddings. Available options: `"cosine"`, `"euclidean"`, `"manhattan"`, `"dot_product"`, `"pearson"` |
| `max_workers` | integer | CPU count | Number of parallel processes to use for computation. This parameter controls the parallelization of similarity matrix computation |

## Underlying Model

VendiScorer does **not require a specific language model** for inference. Instead, it operates on **pre-computed embeddings** that must be generated in advance using an embedding model of your choice. 

**Note**: The embeddings must be saved as a NumPy `.npy` file with shape (N, D) where N matches the number of samples in your dataset and D is the embedding dimension. The order of embeddings must correspond to the order of samples in your dataset file.

## Generating Embeddings

To generate the required embedding file for VendiScorer, you can use the provided `embed.py` script located at:

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

This output file can be directly used as the `embedding_path` parameter in the VendiScorer configuration.


## Scoring Process

1. **Embedding Preparation**: Before running the scorer, embeddings must be pre-computed for all samples in the dataset and saved as a `.npy` file. Each embedding should capture the semantic or structural properties of the corresponding sample.

2. **Embedding Loading**: The scorer loads the embedding matrix from the specified path and verifies that the number of embeddings matches the dataset size.

3. **Similarity Matrix Construction**: Using the specified similarity metric, the scorer computes pairwise similarities between all embeddings to construct an *n* × *n* similarity matrix **K**, where *n* is the number of samples.

4. **Eigenvalue Computation**: The eigenvalues of the similarity matrix **K** are computed and normalized to form a probability distribution.

5. **Entropy Calculation**: The Shannon entropy H(λ) of the normalized eigenvalues is calculated:
   ```
   H(λ) = -Σ λ_i log(λ_i)
   ```

6. **Vendi Score Computation**: The final Vendi Score is computed as the exponential of the entropy:
   ```
   VS = exp(H(λ))
   ```

**Note**: The Vendi Score is a **global metric** computed for the entire dataset, not individual samples. The `score_item()` method is not implemented and will raise an error if called.

## Output Format

The `evaluate()` method returns a dictionary containing:

```json
{
  "vendi_score": 45.32,
  "num_samples": 1000,
  "similarity_metric": "cosine"
}
```

- `vendi_score`: The computed Vendi Score for the entire dataset. This represents the effective number of unique elements in the dataset. Higher values indicate greater diversity
- `num_samples`: The number of samples used in the computation. This should match the size of your dataset
- `similarity_metric`: The similarity metric used for computing the similarity matrix (e.g., `"cosine"`, `"euclidean"`)

## Citation

```bibtex
@article{friedman2023vendi,
  title={The Vendi Score: A Diversity Evaluation Metric for Machine Learning},
  author={Friedman, Dan and Dieng, Adji Bousso},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2023}
}
```
