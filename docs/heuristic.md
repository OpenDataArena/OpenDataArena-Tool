# Heuristic Scorers

Heuristic scorers are model-free evaluation metrics that assess data quality and diversity using statistical, linguistic, and mathematical methods. These scorers operate without requiring pre-trained language models, making them efficient, interpretable, and scalable for large-scale dataset analysis.

---

## ApjsScorer

### Overview

The **ApjsScorer** is a dataset-level diversity evaluation metric that measures the **Average Pairwise Jaccard Similarity (Apjs)** across all samples in an SFT dataset. Unlike sample-wise scoring methods, ApjsScorer computes a single aggregate score for the entire dataset by calculating the average Jaccard similarity between all possible pairs of samples based on their n-gram representations.

This metric is particularly useful for assessing dataset diversity, data deduplication analysis, and dataset quality evaluation. ApjsScorer supports flexible tokenization methods (word-level n-grams or token-level n-grams) and similarity computation strategies (exact or approximate), making it scalable for datasets of varying sizes.

### Metric Definition:

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

### YAML Configuration

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

#### Configuration Parameters

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

### Underlying Model

This scorer does **not require any model**. 

### Scoring Process

1. **Text Extraction**: For each sample in the dataset, concatenate the instruction, input (if present), and output fields into a single text string

2. **N-gram Generation**: Generate n-gram sets based on the configured tokenization method:
   - `tokenization_method="gram"`: Tokenize text into words using NLTK's `word_tokenize`, convert to lowercase, and generate word-level n-grams
   - `tokenization_method="token"`: Encode text using tiktoken encoder and generate token-level n-grams from token IDs

3. **Pairwise Similarity Computation**: Compute Jaccard similarity for all unique pairs using the configured method:
   - `similarity_method="direct"`: Exact computation using `|S_i ∩ S_j| / |S_i ∪ S_j|`
   - `similarity_method="minhash"`: Approximate similarity using MinHash sketches

4. **Parallel Processing**: Leverages multi-processing with `ProcessPoolExecutor` to parallelize pairwise comparisons across CPU cores

5. **Aggregation**: Calculate the mean of all pairwise Jaccard similarities to obtain the final Apjs score

### Output Format

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

### Citation

```bibtex
@article{seed2025seed,
  title={Seed-coder: Let the code model curate data for itself},
  author={Seed, ByteDance and Zhang, Yuyu and Su, Jing and Sun, Yifan and Xi, Chenguang and Xiao, Xia and Zheng, Shen and Zhang, Anxiang and Liu, Kaibo and Zan, Daoguang and others},
  journal={arXiv preprint arXiv:2506.03524},
  year={2025}
}
```


---

## ApsScorer

### Overview

The **Average Pairwise Similarity (APS) Scorer** is a dataset-level diversity evaluation metric that measures the average pairwise similarity across all samples in a dataset by computing similarity between their embedding representations. Unlike sample-wise scoring methods, ApsScorer computes a single aggregate score for the entire dataset by calculating the average similarity between all possible pairs of sample embeddings.

This metric is particularly useful for:
- **Assessing dataset diversity**: Lower APS scores indicate higher diversity (less redundancy) in the dataset
- **Data deduplication analysis**: Identifying semantically similar or duplicate samples

ApsScorer supports multiple similarity metrics (cosine, euclidean, manhattan, dot product, and Pearson correlation) and includes parallel processing capabilities, making it scalable for datasets of varying sizes.

### Metric Definition:

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

### YAML Configuration

```yaml
name: ApsScorer
embedding_path: /path/to/embeddings.npy
similarity_metric: cosine
max_workers: 8
sample_pairs: null
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"ApsScorer"` | Identifier for the scorer |
| `embedding_path` | string | *required* | Path to pre-computed embeddings file in NumPy `.npy` format with shape (N, D) where N is the number of samples and D is the embedding dimension. Must be computed in advance using an embedding model (e.g., Sentence-BERT, BGE, E5) |
| `similarity_metric` | string | `"cosine"` | Similarity metric to use: `"cosine"` (angular similarity, [-1,1]), `"euclidean"` (L2 distance), `"manhattan"` (L1 distance), `"dot_product"` (inner product), or `"pearson"` (correlation coefficient) |
| `max_workers` | int | CPU count | Number of parallel processes for pairwise similarity computation. Adjust based on system capabilities and memory constraints |
| `sample_pairs` | int/null | `null` | Number of pairs to randomly sample for estimation. Set to a positive integer for very large datasets to reduce computation time from O(N²). Provides an approximate APS score |

### Underlying Model

ApsScorer does **not require a specific language model** for inference. Instead, it operates on **pre-computed embeddings** that must be generated in advance using an embedding model of your choice. 

**Note**: The embeddings must be saved as a NumPy `.npy` file with shape (N, D) where N matches the number of samples in your dataset and D is the embedding dimension. The order of embeddings must correspond to the order of samples in your dataset file.

### Generating Embeddings

To generate the required embedding file for ApsScorer, you can use the provided `embed.py` script located at:

```bash
data_scorer/model_based/utils/embed.py
```

#### Usage Example

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

#### Script Parameters

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

#### Key Features

- **Batch Processing**: Processes large datasets efficiently using batched tokenization and embedding computation
- **Automatic Truncation**: Handles long texts by truncating to the specified `max_tokens` limit
- **vLLM Integration**: Uses vLLM for fast and memory-efficient embedding generation with GPU acceleration
- **Flexible Field Extraction**: Supports extracting and concatenating multiple fields from JSONL data
- **Progress Tracking**: Displays progress bars using tqdm for both tokenization and embedding stages

#### Output Format

The script generates a NumPy `.npy` file containing embeddings in float64 format with shape (N, D), where:
- N = number of samples in your input dataset
- D = embedding dimension of the chosen model

This output file can be directly used as the `embedding_path` parameter in the ApsScorer configuration.

### Scoring Process

1. **Embedding Loading**: Load pre-computed embeddings from the specified `.npy` file and validate that the number of embeddings matches the dataset size

2. **Similarity Function Selection**: Select the appropriate similarity computation function based on the configured metric (cosine, euclidean, manhattan, dot product, or Pearson)

3. **Pair Generation**: Generate all possible unique pairs C(N,2) = N×(N-1)/2, or randomly sample `sample_pairs` pairs if specified

4. **Parallel Similarity Computation**: Distribute pair computations across multiple worker processes using `ProcessPoolExecutor` for efficient parallel processing

5. **Aggregation**: Calculate the mean of all pairwise similarities to obtain the final APS score

### Output Format

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

### Citation

```bibtex
@article{yu2023metamath,
  title={Metamath: Bootstrap your own mathematical questions for large language models},
  author={Yu, Longhui and Jiang, Weisen and Shi, Han and Yu, Jincheng and Liu, Zhengying and Zhang, Yu and Kwok, James T and Li, Zhenguo and Weller, Adrian and Liu, Weiyang},
  journal={arXiv preprint arXiv:2309.12284},
  year={2023}
}
```


---

## ClusterInertiaScorer

### Overview

The **Cluster Inertia Scorer** is an embedding-based evaluation tool designed to measure the **diversity** and **dispersion** of datasets through cluster inertia analysis. Unlike sample-level scorers, this scorer evaluates the entire dataset holistically by calculating the sum of distances from all data points to their assigned cluster centroids. This metric provides insights into how tightly or loosely data samples are grouped, which serves as an indicator of dataset diversity and coverage.

Higher inertia values suggest greater data dispersion and diversity, while lower values indicate tighter clustering and more homogeneous data distribution.

### Metric Definition:

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

### YAML Configuration

```yaml
name: ClusterInertiaScorer
embedding_path: /path/to/embeddings.npy
cluster_centroids_path: /path/to/centroids.npy
cluster_labels_path: /path/to/labels.npy
distance_metric: cosine
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"ClusterInertiaScorer"` | Identifier for the scorer |
| `embedding_path` | string | (required) | Path to the embeddings file in `.npy` format. The file should contain a 2D numpy array of shape `(num_samples, embedding_dim)`, where each row represents the embedding vector of a data sample |
| `cluster_centroids_path` | string | (required) | Path to the cluster centroids file in `.npy` format. The file should contain a 2D numpy array of shape `(num_clusters, embedding_dim)`, where each row represents the centroid vector of a cluster |
| `cluster_labels_path` | string | (required) | Path to the cluster labels file in `.npy` format. The file should contain a 1D numpy array of shape `(num_samples,)`, where each element indicates the cluster assignment (0 to num_clusters-1) for the corresponding data sample |
| `distance_metric` | string | `"cosine"` | Distance metric used to compute distances between embeddings and centroids. Supported metrics: `"cosine"` (cosine distance), `"euclidean"` (L2 norm), `"squared_euclidean"` (squared L2), `"manhattan"` (L1 norm) |
| `max_workers` | int | CPU count | Number of worker processes for parallel computation. Defaults to the system's CPU count if not specified or invalid |

### Underlying Model

The Cluster Inertia Scorer **does not require a language model** for evaluation. Instead, it operates on **pre-computed embeddings** and **clustering results**. Users need to:

1. Generate embeddings for their dataset using any embedding model (e.g., sentence transformers, text embeddings from LLMs)
2. Perform clustering analysis (e.g., K-means, DBSCAN, hierarchical clustering) to obtain cluster centroids and labels
3. Save these artifacts as `.npy` files for the scorer to load

This design allows flexibility in choosing embedding models and clustering algorithms based on specific use cases and data characteristics.

### Generating Embeddings

To generate the required embedding file for ClusterInertiaScorer, you can use the provided `embed.py` script located at:

```bash
data_scorer/model_based/utils/embed.py
```

#### Usage Example

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

#### Script Parameters

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

#### Key Features

- **Batch Processing**: Processes large datasets efficiently using batched tokenization and embedding computation
- **Automatic Truncation**: Handles long texts by truncating to the specified `max_tokens` limit
- **vLLM Integration**: Uses vLLM for fast and memory-efficient embedding generation with GPU acceleration
- **Flexible Field Extraction**: Supports extracting and concatenating multiple fields from JSONL data
- **Progress Tracking**: Displays progress bars using tqdm for both tokenization and embedding stages

#### Output Format

The script generates a NumPy `.npy` file containing embeddings in float64 format with shape (N, D), where:
- N = number of samples in your input dataset
- D = embedding dimension of the chosen model

This output file can be directly used as the `embedding_path` parameter in the ClusterInertiaScorer configuration.

### Scoring Process

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

### Output Format

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

### Citation

```bibtex
@inproceedings{du2019boosting,
  title={Boosting dialog response generation},
  author={Du, Wenchao and Black, Alan W},
  booktitle={Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
  year={2019}
}
```



---

## FacilityLocationScorer

### Overview

The **Facility Location Scorer** is an embedding-based evaluation tool designed to assess the **coverage quality** of a selected data subset over the full dataset. Inspired by the classical Facility Location problem in operations research, this scorer treats the selected subset as "facilities" and the full dataset as "customers," measuring how well the facilities serve all customers in the embedding space.

This metric is particularly useful for evaluating data selection strategies, where the goal is to choose a representative subset that maximally covers the diversity of the entire dataset. A lower Facility Location score indicates better coverage, meaning the selected subset can adequately represent the full dataset's distribution.

### Metric Definition:

* **Definition:** 

  $$M_{FL}(X) = \sum_{x_j \in X_{all}} \min_{x_i \in X} d(x_j, x_i)$$

  Where:
  - \(X_{all}\) is the full dataset
  - \(X\) is the selected subset
  - \(d(x_j, x_i)\) is the distance between data points \(x_j\) and \(x_i\) in the embedding space

* **Explanation:**
  * A **lower score** indicates that the selected subset has **better coverage** of the full dataset, as every data point in the full dataset has at least one nearby representative in the subset.
  * A **higher score** suggests that the subset coverage is **inadequate**, with many data points in the full dataset being far from any point in the selected subset.

### YAML Configuration

```yaml
name: FacilityLocationScorer
embedding_path: path/to/full_dataset_embeddings.npy
subset_embeddings_path: path/to/subset_embeddings.npy
distance_metric: euclidean
max_workers: 8
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"FacilityLocationScorer"` | Identifier for the scorer |
| `embedding_path` | string | (required) | Path to the `.npy` file containing embeddings of the **full dataset**. This serves as the reference background for evaluating coverage quality. |
| `subset_embeddings_path` | string | (required) | Path to the `.npy` file containing embeddings of the **subset to be evaluated**. This subset corresponds to the data samples in `input_path`. |
| `distance_metric` | string | `"euclidean"` | The distance metric used for computing distances between embeddings. Available options: `"euclidean"` (Standard Euclidean distance), `"squared_euclidean"` (Squared Euclidean distance, faster), `"manhattan"` (Manhattan/L1 distance), `"cosine"` (Cosine distance: 1 - cosine similarity) |
| `max_workers` | integer | CPU count | Number of parallel workers for multiprocessing. Higher values can accelerate computation for large datasets. |

### Underlying Model

FacilityLocationScorer does **not require a specific language model** for inference. Instead, it operates on **pre-computed embeddings** that must be generated in advance using an embedding model of your choice. 

**Note**: The embeddings must be saved as a NumPy `.npy` file with shape (N, D) where N matches the number of samples in your dataset and D is the embedding dimension. The order of embeddings must correspond to the order of samples in your dataset file.

### Generating Embeddings

To generate the required embedding file for FacilityLocationScorer, you can use the provided `embed.py` script located at:

```bash
data_scorer/model_based/utils/embed.py
```

#### Usage Example

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

#### Script Parameters

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

#### Key Features

- **Batch Processing**: Processes large datasets efficiently using batched tokenization and embedding computation
- **Automatic Truncation**: Handles long texts by truncating to the specified `max_tokens` limit
- **vLLM Integration**: Uses vLLM for fast and memory-efficient embedding generation with GPU acceleration
- **Flexible Field Extraction**: Supports extracting and concatenating multiple fields from JSONL data
- **Progress Tracking**: Displays progress bars using tqdm for both tokenization and embedding stages

#### Output Format

The script generates a NumPy `.npy` file containing embeddings in float64 format with shape (N, D), where:
- N = number of samples in your input dataset
- D = embedding dimension of the chosen model

This output file can be directly used as the `embedding_path` parameter in the FacilityLocationScorer configuration.

### Scoring Process

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

### Output Format

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

### Citation

```bibtex
@book{farahani2009facility,
  title={Facility location: concepts, models, algorithms and case studies},
  author={Farahani, Reza Zanjirani and Hekmatfar, Masoud},
  year={2009},
  publisher={Springer Science \& Business Media}
}
```


---

## GramEntropyScorer

### Overview

The **Gram Entropy Scorer** is a statistical evaluation tool designed to measure the lexical diversity and linguistic richness of SFT (Supervised Fine-Tuning) data at the word level by computing 1-gram (unigram) entropy. This scorer analyzes the distribution of words in instruction-response pairs and quantifies their unpredictability using Shannon entropy. Higher entropy scores indicate more diverse vocabulary usage and potentially richer linguistic content, while lower scores suggest repetitive or limited word patterns.

Unlike token-based entropy scorers that operate on subword units, this scorer works at the natural word level using linguistic tokenization, making it more interpretable from a linguistic perspective and suitable for analyzing vocabulary diversity.

### Metric Definition:

* **Definition:** 

  1-Gram Entropy (Unigram Entropy) is computed using Shannon's entropy formula:
  
  ```
  H(X) = -Σ p(x) * log₂(p(x))
  ```
  
  where `p(x)` is the probability (frequency) of each unique word in the text.

* **Explanation:** This metric quantifies the unpredictability or diversity of word usage in the data:
  
  * A **higher 1-Gram Entropy score** indicates greater vocabulary diversity, with words distributed more evenly throughout the text. This suggests the data contains varied and linguistically rich content.
  * A **lower 1-Gram Entropy score** suggests repetitive word usage or limited vocabulary, potentially indicating redundant, template-like, or formulaic content.

### YAML Configuration

```yaml
name: GramEntropyScorer
max_workers: 8
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"GramEntropyScorer"` | Identifier for the scorer |
| `max_workers` | integer | Number of CPU cores | Number of parallel processes to use for scoring. Higher values speed up processing but consume more CPU resources |

### Underlying Model

This scorer does **not require a deep learning model**. It uses the **NLTK (Natural Language Toolkit)** library for word-level tokenization, specifically the `punkt_tab` tokenizer for sentence and word boundary detection. The tokenizer automatically downloads if not present in the system.

Unlike subword tokenizers (e.g., BPE or SentencePiece), NLTK's word tokenizer splits text into linguistically meaningful words, making it more suitable for analyzing natural vocabulary diversity.

### Scoring Process

1. **Text Concatenation**: For each data item, the instruction, optional input, and output (response) are concatenated: `text = instruction + '\n' + input + '\n' + output` (If no input field exists, it is omitted)

2. **Text Normalization**: The concatenated text is converted to lowercase to ensure case-insensitive word counting: `text = text.lower()`

3. **Word Tokenization**: The normalized text is tokenized into words using NLTK's `word_tokenize` function, which splits text into individual words, handling punctuation and special characters appropriately

4. **Word Frequency Analysis**: The frequency of each unique word is counted to build a probability distribution: `p(word_i) = count(word_i) / total_words`

5. **Entropy Calculation**: Shannon entropy is computed across all unique words: `H = -Σ p(word_i) * log₂(p(word_i))`

6. **Parallel Processing**: When evaluating datasets, the scorer uses `ProcessPoolExecutor` to distribute work across multiple CPU cores for efficient batch processing. Each worker process independently downloads NLTK data if needed

### Output Format

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

### Citation

```bibtex
@inproceedings{zhuang2025meta,
  title={Meta-rater: A multi-dimensional data selection method for pre-training language models},
  author={Zhuang, Xinlin and Peng, Jiahui and Ma, Ren and Wang, Yinfan and Bai, Tianyi and Wei, Xingjian and Jiantao, Qiu and Zhang, Chi and Qian, Ying and He, Conghui},
  booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={10856--10896},
  year={2025}
}
```



---

## HddScorer

### Overview

The **HD-D (Hypergeometric Distribution D) Scorer** is a statistical evaluation tool designed to measure **lexical diversity** in text. Proposed by McCarthy & Jarvis (2010), HD-D provides a robust approach to quantifying vocabulary richness that is largely independent of text length, making it superior to traditional type-token ratio (TTR) measures. This scorer calculates the probability of encountering unique word types within sampled segments of text using the hypergeometric distribution function.

Unlike model-based scorers, HD-D is a **statistical metric** that does not require any pre-trained models. It is particularly useful for assessing the linguistic complexity and vocabulary variety of instruction-following datasets, providing insights into the lexical richness of both instructions and responses.

### Metric Definition:

* **Definition:** 

  HD-D is calculated by summing the contribution of each unique word type to the overall lexical diversity:

  ```
  HD-D = Σ [1 - P(X = 0)] / sample_size
  ```

  where `P(X = 0)` is the hypergeometric probability that a given word type does **not** appear in a random sample of tokens from the text.

* **Explanation:** This metric estimates the **lexical diversity** of text by measuring how likely each unique word type will appear in a randomly sampled segment:

  * A **higher HD-D score** indicates **greater lexical diversity**, suggesting rich vocabulary usage and varied word choices.
  * A **lower HD-D score** indicates **lower lexical diversity**, suggesting repetitive language or limited vocabulary.

The hypergeometric distribution accounts for sampling without replacement, making HD-D more mathematically sound than simple ratio-based measures. The metric is computed as:

```
P(X = k) = [C(K, k) × C(N-K, n-k)] / C(N, n)
```

where:
- `N` = total number of tokens in the text
- `K` = frequency of a specific word type
- `n` = sample size
- `k` = number of times the word type appears in the sample
- `C(n, r)` = binomial coefficient "n choose r"

### YAML Configuration

```yaml
name: HddScorer
sample_size: 42.0
max_workers: 8
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"HddScorer"` | Identifier for the scorer |
| `sample_size` | float | `42.0` | The sample size used in the hypergeometric distribution calculation. Controls the window size for calculating lexical diversity. If the text is shorter than the sample size, the actual text length will be used instead |
| `max_workers` | integer | CPU core count | Number of parallel worker processes for multiprocessing. Higher values can speed up processing for large datasets but consume more memory |

### Underlying Model

**Not applicable.** HD-D is a statistical metric based on the hypergeometric distribution and does not require any pre-trained language models. The calculation is purely mathematical and depends only on word frequency distributions in the text.

### Scoring Process

The HD-D scorer follows these steps to evaluate lexical diversity:

1. **Text Concatenation:** For each data sample, the `instruction`, `input`, and `output` fields are concatenated into a single text string:
   ```
   text = instruction + '\n' + input + '\n' + output
   ```
   If the `input` field is empty, it is omitted from concatenation.

2. **Tokenization:** The concatenated text is split into tokens (words) using whitespace separation.

3. **Preprocessing:** Each token is processed by:
   - Removing all punctuation marks
   - Converting to lowercase
   - Filtering out empty strings

4. **Type Counting:** A frequency dictionary is built where each unique word type (after preprocessing) is mapped to its occurrence count in the text.

5. **Hypergeometric Calculation:** For each unique word type:
   - Calculate the hypergeometric probability `P(X = 0)` that the word does **not** appear in a random sample
   - Compute the contribution: `[1 - P(X = 0)] / sample_size`
   - Handle any mathematical errors (overflow, division by zero) gracefully

6. **Score Aggregation:** Sum all individual contributions to obtain the final HD-D score.

7. **Parallel Processing:** Multiple samples are processed in parallel using `ProcessPoolExecutor` for improved performance on large datasets.

### Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 0.8523
}
```

- `id`: The unique identifier of the data sample, extracted from the input data's `id` field. If no `id` is present, defaults to `"unknown"`
- `score`: The HD-D lexical diversity score for the sample. Typically between 0 and the number of unique word types in the text. Higher values indicate greater lexical diversity. A value of 0.0 indicates empty text or processing error
- `error` (optional): Only present if an error occurred during processing. Contains the error message for debugging purposes

### Citation

```bibtex
@article{mccarthy2010mtld,
  title={MTLD, vocd-D, and HD-D: A validation study of sophisticated approaches to lexical diversity assessment},
  author={McCarthy, Philip M and Jarvis, Scott},
  journal={Behavior research methods},
  volume={42},
  number={2},
  pages={381--392},
  year={2010},
  publisher={Springer}
}
```



---

## KNN Scorer

### Overview

The **KNN Scorer** (K-Nearest Neighbors Scorer) is an embedding-based evaluation method that measures the **local density** and **uniqueness** of data points in the embedding space. This approach quantifies data diversity by computing the average distance to each sample's k-nearest neighbors. Originally proposed by Google Research as a high-fidelity data selection strategy, KNN scoring helps identify samples that are either centrally located (redundant) or peripherally positioned (unique/outlier) within the dataset's semantic space.

The core intuition is that samples with **larger K-nearest neighbor distances** are more unique or isolated in the embedding space, potentially representing diverse or valuable examples, while samples with **smaller distances** are surrounded by similar data and may be redundant.

### Metric Definition:

* **Definition:** 

  For a given data point \( x_i \) with embedding \( e_i \), the KNN distance is calculated as:

  \[
  \text{KNN\_Distance}(x_i) = \frac{1}{k} \sum_{j=1}^{k} d(e_i, e_{n_j})
  \]

  where \( e_{n_j} \) represents the embedding of the j-th nearest neighbor (excluding \( x_i \) itself), and \( d(\cdot, \cdot) \) is a distance metric (e.g., Euclidean, cosine, or Manhattan distance).

* **Explanation:** This metric quantifies how isolated or central a data point is within its local neighborhood:
  
  * A **higher KNN distance** indicates that the sample is **far from its neighbors**, suggesting it is **unique, diverse, or potentially an outlier** in the dataset.
  * A **lower KNN distance** indicates that the sample is **close to many similar samples**, suggesting it is **redundant or representative of a dense cluster**.

### YAML Configuration

```yaml
name: KNNScorer
embedding_path: /path/to/embeddings.npy
k: 5
distance_metric: euclidean
max_workers: 8
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"KNNScorer"` | Identifier for the scorer |
| `embedding_path` | string | **required** | Path to the pre-computed embeddings file in `.npy` format. This file should be a 2D NumPy array with shape `(num_samples, embedding_dim)`, where each row corresponds to the embedding of a data sample in the dataset. **The order of embeddings must match the order of samples in the dataset.** |
| `k` | integer | `5` | Number of nearest neighbors to consider for distance calculation. If `k` is greater than or equal to the dataset size, it will be automatically adjusted to `num_samples - 1` |
| `distance_metric` | string | `"euclidean"` | Distance metric for computing neighbor distances. Supported values: `"euclidean"` (Euclidean/L2 distance), `"cosine"` (cosine distance = 1 - cosine similarity), `"manhattan"` (Manhattan/L1 distance) |
| `max_workers` | integer | CPU cores | Number of parallel worker processes for scoring. Higher values speed up processing but require more memory |

### Underlying Model

KNNScorer does **not require a specific language model** for inference. Instead, it operates on **pre-computed embeddings** that must be generated in advance using an embedding model of your choice. 

**Note**: The embeddings must be saved as a NumPy `.npy` file with shape (N, D) where N matches the number of samples in your dataset and D is the embedding dimension. The order of embeddings must correspond to the order of samples in your dataset file.

### Generating Embeddings

To generate the required embedding file for KNNScorer, you can use the provided `embed.py` script located at:

```bash
data_scorer/model_based/utils/embed.py
```

#### Usage Example

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

#### Script Parameters

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

#### Key Features

- **Batch Processing**: Processes large datasets efficiently using batched tokenization and embedding computation
- **Automatic Truncation**: Handles long texts by truncating to the specified `max_tokens` limit
- **vLLM Integration**: Uses vLLM for fast and memory-efficient embedding generation with GPU acceleration
- **Flexible Field Extraction**: Supports extracting and concatenating multiple fields from JSONL data
- **Progress Tracking**: Displays progress bars using tqdm for both tokenization and embedding stages

#### Output Format

The script generates a NumPy `.npy` file containing embeddings in float64 format with shape (N, D), where:
- N = number of samples in your input dataset
- D = embedding dimension of the chosen model

This output file can be directly used as the `embedding_path` parameter in the KNNScorer configuration.

### Scoring Process

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

### Output Format

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

### Citation

```bibtex
@misc{google2025highfidelity,
  title        = {Achieving 10000x Training Data Reduction with High-Fidelity Labels},
  author       = {{Google Research}},
  howpublished = {\url{https://research.google/blog/achieving-10000x-training-data-reduction-with-high-fidelity-labels/}},
  note         = {Accessed: 2025-02-xx}
}
```



---

## LogDetDistanceScorer

### Overview

The **Log-Det Distance Scorer** is a diversity measurement tool for instruction tuning datasets based on the determinantal point process (DPP) framework. Proposed in [Wang et al., 2024](https://arxiv.org/abs/2402.02318), this method quantifies dataset diversity by computing the log-determinant of the cosine similarity matrix constructed from data embeddings. Unlike heuristic diversity measures (e.g., counting number of tasks), Log-Det Distance provides a principled, geometry-based metric that correlates with downstream instruction-following performance and can inform data selection strategies.

### Metric Definition:

* **Definition:** `Log-Det Distance = log(det(S))`

  where `S` is the `N × N` cosine similarity matrix computed from the embeddings of `N` data samples.

* **Explanation:** The log-determinant measures the "volume" spanned by the data embeddings in the feature space:
  
  * A **higher Log-Det value** indicates that the data samples are **more diverse** and span a larger volume in the embedding space, suggesting rich coverage of different instruction patterns.
  * A **lower Log-Det value** indicates **high similarity** among samples and **low diversity**, suggesting redundant or homogeneous data.

* **Mathematical Properties:**
  
  * The similarity matrix `S` should be positive semi-definite (all eigenvalues ≥ 0) to ensure `det(S) ≥ 0`, making `log(det(S))` mathematically valid.
  * Ridge regularization can be applied to improve numerical stability: `S' = S + α·I`, where `α` is a small positive constant.

### YAML Configuration

```yaml
name: LogDetDistanceScorer
embedding_path: path/to/embeddings.npy
max_workers: 8
ridge_alpha: 1e-10
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"LogDetDistanceScorer"` | Identifier for the scorer |
| `embedding_path` | string | **(required)** | Path to the pre-computed embedding file in `.npy` format. The embeddings should be a NumPy array of shape `[num_samples, embedding_dim]`, where each row corresponds to one data sample in the dataset |
| `max_workers` | integer | CPU cores | Number of parallel worker processes for computing the similarity matrix. For datasets with fewer than 5,000 samples, a vectorized method is automatically used instead |
| `ridge_alpha` | float | `1e-10` | Ridge regularization parameter added to the diagonal of the similarity matrix for numerical stability (`S' = S + α·I`). Can be specified in scientific notation |


### Underlying Model

LogDetDistanceScorer does **not require a specific language model** for inference. Instead, it operates on **pre-computed embeddings** that must be generated in advance using an embedding model of your choice. 

**Note**: The embeddings must be saved as a NumPy `.npy` file with shape (N, D) where N matches the number of samples in your dataset and D is the embedding dimension. The order of embeddings must correspond to the order of samples in your dataset file.

### Generating Embeddings

To generate the required embedding file for LogDetDistanceScorer, you can use the provided `embed.py` script located at:

```bash
data_scorer/model_based/utils/embed.py
```

#### Usage Example

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

#### Script Parameters

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

#### Key Features

- **Batch Processing**: Processes large datasets efficiently using batched tokenization and embedding computation
- **Automatic Truncation**: Handles long texts by truncating to the specified `max_tokens` limit
- **vLLM Integration**: Uses vLLM for fast and memory-efficient embedding generation with GPU acceleration
- **Flexible Field Extraction**: Supports extracting and concatenating multiple fields from JSONL data
- **Progress Tracking**: Displays progress bars using tqdm for both tokenization and embedding stages

#### Output Format

The script generates a NumPy `.npy` file containing embeddings in float64 format with shape (N, D), where:
- N = number of samples in your input dataset
- D = embedding dimension of the chosen model

This output file can be directly used as the `embedding_path` parameter in the LogDetDistanceScorer configuration.

### Scoring Process

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

### Output Format

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

### Citation

```bibtex
@article{wang2024diversity,
  title={Diversity measurement and subset selection for instruction tuning datasets},
  author={Wang, Peiqi and Shen, Yikang and Guo, Zhen and Stallone, Matthew and Kim, Yoon and Golland, Polina and Panda, Rameswar},
  journal={arXiv preprint arXiv:2402.02318},
  year={2024}
}
```



---

## MtldScorer

### Overview

The **MTLD (Measure of Textual Lexical Diversity) Scorer** is a statistical evaluation tool designed to measure **lexical diversity** in text through a novel sequential analysis approach. Proposed by McCarthy & Jarvis (2010), MTLD calculates lexical diversity as the mean length of sequential word strings that maintain a specified Type-Token Ratio (TTR) threshold. This approach addresses the well-known sensitivity of traditional TTR measures to text length, providing a more robust and length-independent assessment of vocabulary richness.

Unlike model-based scorers, MTLD is a **statistical metric** that does not require any pre-trained models. It is particularly valuable for evaluating the linguistic complexity and vocabulary variety of instruction-following datasets, offering complementary insights to other lexical diversity measures like HD-D.

### Metric Definition:

* **Definition:** 

  MTLD is calculated as the average of forward and backward traversals through the text:

  ```
  MTLD = (MTLD_forward + MTLD_backward) / 2
  ```

  where each directional MTLD is computed as:

  ```
  MTLD_directional = total_tokens / factor_count
  ```

  A **factor** is defined as a contiguous sequence of tokens where the TTR (unique types / total tokens) remains above a specified threshold. When TTR drops to or below the threshold, a new factor begins.

* **Explanation:** This metric estimates **lexical diversity** by measuring how long the text can maintain vocabulary variety before repeating words:
  
  * A **higher MTLD score** indicates **greater lexical diversity**, meaning the text maintains vocabulary richness over longer sequences, suggesting sophisticated and varied language use.
  * A **lower MTLD score** indicates **lower lexical diversity**, meaning the text quickly exhausts its vocabulary and begins repeating words, suggesting simpler or more repetitive language.

* **Key Advantages:**
  
  * **Length-invariant:** Unlike traditional TTR measures, MTLD is not biased by text length
  * **Bidirectional calculation:** Forward and backward traversals help mitigate artifacts from word ordering
  * **Partial factor inclusion:** Provides more accurate measures for texts that don't end on factor boundaries

### YAML Configuration

```yaml
name: MtldScorer
ttr_threshold: 0.72
max_workers: 8
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"MtldScorer"` | Identifier for the scorer |
| `ttr_threshold` | float | `0.72` | The Type-Token Ratio threshold used to determine factor boundaries. When TTR drops to or below this value, it marks the end of a factor. Range: (0, 1). The default 0.72 is established in the original research as providing optimal discrimination |
| `max_workers` | integer | CPU core count | Number of parallel worker processes for multiprocessing. Higher values speed up processing but consume more memory |

### Underlying Model

**Not applicable.** MTLD is a statistical metric based on sequential Type-Token Ratio analysis and does not require any pre-trained language models. The calculation is purely algorithmic and depends only on the sequential ordering and frequency of words in the text.

### Scoring Process

1. **Input Processing**: For each data sample, the scorer concatenates the `instruction`, `input`, and `output` fields into a single text string. If the `input` field is empty, it is omitted from concatenation

2. **Tokenization**: The concatenated text is split into tokens (words) using whitespace separation

3. **Preprocessing**: Each token is processed by removing all punctuation marks and converting to lowercase

4. **Forward MTLD Calculation**: Sequentially process tokens from start to end, tracking TTR. Each time TTR drops to or below the threshold, count a complete factor. Calculate `MTLD_forward = total_tokens / (factor_count + partial_factor)`

5. **Backward MTLD Calculation**: Repeat the forward calculation with the token sequence reversed to get `MTLD_backward`

6. **Score Computation**: Calculate final score as `(MTLD_forward + MTLD_backward) / 2`

### Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 67.34
}
```

- `id`: The unique identifier of the data sample, extracted from the input data's `id` field. If no `id` is present, defaults to `"unknown"`
- `score`: The MTLD lexical diversity score for the sample. Typically ranges from a few (for highly repetitive text) to hundreds or more (for highly diverse text). Higher values indicate greater lexical diversity. Value of 0.0 indicates empty text or processing error
- `error` (optional): Only present if an error occurred during processing. Contains the error message for debugging purposes

### Citation

```bibtex
@article{mccarthy2010mtld,
  title={MTLD, vocd-D, and HD-D: A validation study of sophisticated approaches to lexical diversity assessment},
  author={McCarthy, Philip M and Jarvis, Scott},
  journal={Behavior research methods},
  volume={42},
  number={2},
  pages={381--392},
  year={2010},
  publisher={Springer}
}
```



---

## NovelSumScorer

### Overview

The **NovelSum Scorer** is a dataset-level diversity metric designed to measure data diversity for instruction tuning. Proposed in [Yang et al., 2025](https://aclanthology.org/2025.acl-long.908/), this method addresses the fundamental problem of precisely defining and measuring data diversity in instruction-tuning datasets. Unlike per-sample scoring methods, NovelSum evaluates the entire dataset holistically by considering both inter-sample differences and information density in the sample space.

Through systematic analysis of existing diversity measurement methods, the authors found that a reliable diversity measure should properly account for sample-level "novelty." Experiments demonstrate that NovelSum achieves a **0.97 correlation** with instruction-tuned model performance, making it a valuable metric for guiding data engineering practices and diversity-oriented data selection strategies.

### Metric Definition:

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

### YAML Configuration

```yaml
name: NovelSumScorer
embedding_path: /path/to/embeddings.npy
dense_ref_path: /path/to/reference/embeddings/
max_workers: 8
density_powers: [0, 0.25, 0.5]
neighbors: [5, 10]
distance_powers: [0, 1, 2]
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"NovelSumScorer"` | Identifier for the scorer |
| `embedding_path` | string | *required* | Path to the embeddings file (`.npy` format) for the dataset to be evaluated. The embeddings should be a 2D numpy array with shape `(num_samples, embedding_dim)`, where each row corresponds to one data sample. |
| `dense_ref_path` | string | Directory containing `embedding_path` | Path to reference embeddings used for local density computation. Can be either a directory containing multiple `.npy` files (all will be loaded and concatenated) or a single `.npy` file. The reference data is used to build a Faiss index for computing local density in the embedding space. |
| `max_workers` | int | CPU count | Number of worker processes for parallel computation of the cosine distance matrix. If not specified or invalid, defaults to the number of CPU cores. Higher values can speed up computation but require more memory. |
| `density_powers` | list[float] | `[0, 0.25, 0.5]` | List of power values for density weighting. Controls how density affects the novelty calculation: `0` for no density weighting (uniform), `> 0` for higher density influence. |
| `neighbors` | list[int] | `[5, 10]` | List of k-nearest neighbor values for density estimation. Determines the neighborhood size used to compute local density. |
| `distance_powers` | list[float] | `[0, 1, 2]` | List of power values for distance weighting in the final aggregation. Controls the weighting scheme when averaging distances: `0` for uniform weighting, `1` for linear inverse-rank weighting, `2` for quadratic inverse-rank weighting. |

### Underlying Model

NovelSumScorer does **not require a specific language model** for inference. Instead, it operates on **pre-computed embeddings** that must be generated in advance using an embedding model of your choice. 

**Note**: The embeddings must be saved as a NumPy `.npy` file with shape (N, D) where N matches the number of samples in your dataset and D is the embedding dimension. The order of embeddings must correspond to the order of samples in your dataset file.

### Generating Embeddings

To generate the required embedding file for NovelSumScorer, you can use the provided `embed.py` script located at:

```bash
data_scorer/model_based/utils/embed.py
```

#### Usage Example

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

#### Script Parameters

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

#### Key Features

- **Batch Processing**: Processes large datasets efficiently using batched tokenization and embedding computation
- **Automatic Truncation**: Handles long texts by truncating to the specified `max_tokens` limit
- **vLLM Integration**: Uses vLLM for fast and memory-efficient embedding generation with GPU acceleration
- **Flexible Field Extraction**: Supports extracting and concatenating multiple fields from JSONL data
- **Progress Tracking**: Displays progress bars using tqdm for both tokenization and embedding stages

#### Output Format

The script generates a NumPy `.npy` file containing embeddings in float64 format with shape (N, D), where:
- N = number of samples in your input dataset
- D = embedding dimension of the chosen model

This output file can be directly used as the `embedding_path` parameter in the NovelSumScorer configuration.

### Scoring Process

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

### Output Format

The `evaluate()` method returns a dictionary with the following structure:

```python
{
    'num_samples': 1000,
    'cos_distance': 0.654321,
    'neighbor_5_density_0_distance_0': 0.123456,
    'neighbor_5_density_0_distance_1': 0.234567,
    'neighbor_5_density_0_distance_2': 0.345678,
    'neighbor_5_density_0.25_distance_0': 0.456789,
#    # ... (one entry per hyperparameter combination)
}
```

#### Output Keys

* **`num_samples`**: Integer indicating the number of samples in the evaluated dataset.

* **`cos_distance`**: Float representing the average pairwise cosine distance across all samples. This serves as a baseline diversity measure.

* **`neighbor_{nb}_density_{dp}_distance_{distp}`**: Float representing the NovelSum score computed with:
  - `nb`: Number of neighbors for density estimation (from `neighbors` config)
  - `dp`: Density power (from `density_powers` config)
  - `distp`: Distance power for weighted averaging (from `distance_powers` config)
  
  Higher values indicate greater diversity under that specific configuration.

#### Recommended Configuration

Based on the paper's findings, the configuration with **neighbor=10, density_power=0.5, distance_power=1** (i.e., `neighbor_10_density_0.5_distance_1`) typically achieves the strongest correlation with model performance and is recommended as the primary diversity metric.

### Citation

```bibtex
@inproceedings{yang2025measuring,
  title={Measuring data diversity for instruction tuning: A systematic analysis and a reliable metric},
  author={Yang, Yuming and Nan, Yang and Ye, Junjie and Dou, Shihan and Wang, Xiao and Li, Shuo and Lv, Huijie and Gui, Tao and Zhang, Qi and Huang, Xuan-Jing},
  booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={18530--18549},
  year={2025}
}
```


---

## PartitionEntropyScorer

### Overview

The **Partition Entropy Scorer** is a statistical evaluation tool designed to measure the *diversity* of a data subset by analyzing its distribution across global clusters. This metric quantifies how uniformly the subset data is distributed among predefined clusters, providing insights into data heterogeneity without requiring any language model.

Unlike model-based scorers, this is a **dataset-level metric** that computes a single score for the entire subset rather than individual samples. It is particularly useful for evaluating whether a selected subset maintains diverse coverage across different data clusters.

### Metric Definition:

* **Definition:** 
  
  The Partition Entropy is calculated using the standard entropy formula:
  
  \[ H = -\sum_{i=1}^{k} p_i \log(p_i) \]
  
  where \( p_i \) is the proportion of samples in the subset belonging to cluster \( i \), and \( k \) is the number of clusters represented in the subset.

* **Interpretation:**
  
  * A **higher entropy** indicates that the subset is **more uniformly distributed** across clusters, suggesting **greater diversity**.
  * A **lower entropy** indicates that the subset is **concentrated** in fewer clusters, suggesting **lower diversity**.
  * The **normalized entropy** (entropy divided by \(\log(N)\), where \(N\) is the total number of global clusters) provides a scale-invariant measure between 0 and 1.

### YAML Configuration

```yaml
name: PartitionEntropyScorer
num_clusters: 100
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"PartitionEntropyScorer"` | Identifier for the scorer |
| `num_clusters` | integer | - | The total number of global clusters used for partitioning the entire dataset. This must be a positive integer and should match the clustering performed on your full dataset. This parameter is crucial for computing the normalized entropy, which measures subset diversity relative to the global cluster space. |

### Underlying Model

This scorer does **not require any language model**. It is a purely statistical method based on entropy calculation from cluster distributions.

### Scoring Process

1. **Input Validation**: Each data sample in the subset must contain a `cluster_id` field indicating its cluster assignment from a global clustering.

2. **Cluster Distribution Counting**: The scorer reads through the entire subset and counts how many samples belong to each cluster.

3. **Probability Calculation**: For each cluster represented in the subset, the probability \( p_i \) is computed as:
   \[ p_i = \frac{\text{count}_i}{\text{total samples in subset}} \]

4. **Entropy Computation**: The partition entropy is calculated using the formula:
   \[ H = -\sum_{i \in \text{subset clusters}} p_i \log(p_i) \]

5. **Normalization**: The normalized entropy is computed as:
   \[ H_{\text{normalized}} = \frac{H}{\log(N)} \]
   where \( N \) is the `num_clusters` parameter (total global clusters).

### Output Format

The scorer returns a dictionary containing the following metrics:

```json
{
  "entropy": 4.2341,
  "normalized_entropy": 0.9182,
  "max_entropy": 4.6052,
  "num_samples": 1000,
  "num_clusters_global": 100,
  "num_clusters_in_subset": 68,
  "cluster_counts": {
    "0": 15,
    "1": 12,
    "2": 18,
    "...": "..."
  },
  "cluster_probabilities": {
    "0": 0.015,
    "1": 0.012,
    "2": 0.018,
    "...": "..."
  }
}
```

- `entropy`: The raw partition entropy value \( H = -\sum p_i \log(p_i) \). Higher values indicate more uniform distribution across clusters.
- `normalized_entropy`: The entropy normalized by the maximum possible entropy \( \log(N) \), where \( N \) is the total number of global clusters. This value ranges from 0 to 1, making it easier to compare across different clustering configurations.
- `max_entropy`: The maximum possible entropy based on the global number of clusters, calculated as \( \log(\text{num\_clusters\_global}) \).
- `num_samples`: The total number of valid samples in the subset that have a `cluster_id` field.
- `num_clusters_global`: The total number of global clusters (from configuration), representing the full cluster space.
- `num_clusters_in_subset`: The number of unique clusters actually represented in the subset. This can be less than or equal to `num_clusters_global`.
- `cluster_counts`: A dictionary mapping each cluster ID to the number of samples from that cluster in the subset.
- `cluster_probabilities`: A dictionary mapping each cluster ID to its probability \( p_i \) in the subset distribution.

### Citation

```bibtex
@inproceedings{yang2025measuring,
  title={Measuring data diversity for instruction tuning: A systematic analysis and a reliable metric},
  author={Yang, Yuming and Nan, Yang and Ye, Junjie and Dou, Shihan and Wang, Xiao and Li, Shuo and Lv, Huijie and Gui, Tao and Zhang, Qi and Huang, Xuan-Jing},
  booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={18530--18549},
  year={2025}
}
```



---

## PureThinkScorer

### Overview

The **PureThink Scorer** is a rule-based evaluation tool designed to assess the quality of reasoning-based code generation data, particularly for datasets containing explicit thinking processes. Inspired by the methodology in [OpenCodeReasoning (Ahmad et al., 2025)](https://arxiv.org/pdf/2504.01943), this scorer evaluates whether the reasoning traces (enclosed in `<think>...</think>` or `<redacted_reasoning>...</redacted_reasoning>` tags) are **pure thinking** - containing only logical reasoning without executable code - while the final solution code exists separately outside the thinking tags.

This scorer is particularly useful for filtering and evaluating distilled reasoning datasets where models generate chain-of-thought reasoning before producing code solutions, ensuring that the thinking process remains conceptual and the implementation details are properly separated.

### Metric Definition:

* **Definition:** 

  The PureThink Score is a categorical metric that evaluates the structure and quality of reasoning-augmented code generation data:
  
  * **Score = 1**: **Ideal** - Contains thinking tags with pure reasoning (no code blocks inside thinking section), and has executable code blocks outside the thinking section.
  * **Score = 0**: **Mixed** - Contains thinking tags, has code blocks outside thinking section, but also contains code blocks within the thinking section (reasoning is contaminated with implementation details).
  * **Score = -1**: **Incomplete** - Contains thinking tags but lacks executable code blocks after removing the thinking section (missing final implementation).
  * **Score = -2**: **No Reasoning** - No thinking tags detected in the response.

* **Explanation:** 

  This metric ensures the separation of concerns in reasoning-based code generation:
  
  * A **score of 1** indicates high-quality data where reasoning and implementation are properly separated, which is ideal for training models to think before coding.
  * A **score of 0** suggests that code appears prematurely in the reasoning phase, potentially reducing the quality of the thinking process.
  * A **score of -1** indicates incomplete responses where reasoning exists but implementation is missing.
  * A **score of -2** indicates responses without explicit reasoning traces.

### YAML Configuration

```yaml
name: PureThinkScorer
field: output  # The field name in the dataset to examine (default: "output")
max_workers: 8  # Number of parallel worker processes for scoring (default: CPU count)
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"PureThinkScorer"` | Identifier for the scorer |
| `field` | string | `"output"` | The key in each data item's JSON structure that contains the text to be evaluated. This should point to the field containing model-generated responses with potential thinking tags |
| `max_workers` | integer | CPU count | Number of parallel processes for multi-core processing. Higher values increase processing speed for large datasets |


### Underlying Model

This scorer **does not require any language model**. It operates purely through rule-based pattern matching and regex-based text analysis, making it extremely fast and resource-efficient. The scorer:

* Uses compiled regular expressions to detect thinking tags: `<think>`, `</think>`, `<redacted_reasoning>`, `</redacted_reasoning>`
* Employs pattern matching to identify markdown code blocks (` ```language...``` `)
* Performs deterministic text processing without any neural network inference

### Scoring Process

1. **Extract Text**: Read the specified field (e.g., `output`) from each data item

2. **Detect Thinking Tags**: Check if the text contains `<think>...</think>` or `<redacted_reasoning>...</redacted_reasoning>` tags
   - If **no thinking tags found** → Return score **-2**

3. **Extract Components**: If thinking tags exist:
   - Extract content **within** thinking tags (thinking content)
   - Extract content **outside** thinking tags (remaining text)

4. **Check Code in Remaining Text**: Examine if remaining text contains markdown code blocks (` ```...``` `)
   - If **no code blocks in remaining text** → Return score **-1**

5. **Check Code in Thinking Content**: If code blocks exist in remaining text, check if thinking content also contains code blocks
   - If **thinking content contains code blocks** → Return score **0**
   - If **thinking content has NO code blocks** → Return score **1** (ideal)

6. **Code Block Detection**: The scorer identifies code blocks using markdown syntax with pattern ` ```language\ncode\n``` ` or ` ```\ncode\n``` `, supporting common languages like `python`, `java`, `cpp`, `javascript`, etc.

7. **Parallel Processing**: The scorer leverages Python's `ProcessPoolExecutor` to distribute data items across multiple worker processes for efficient processing, with progress tracked using `tqdm` progress bars

### Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 1.0
}
```

- `id`: The unique identifier of the data item (extracted from the original data's `"id"` field). If no `id` field exists in the input, defaults to `"unknown"`
- `score`: The PureThink score (float value)
  - `1.0`: Pure thinking with proper code separation (ideal)
  - `0.0`: Thinking contaminated with code blocks
  - `-1.0`: Thinking exists but no final code implementation
  - `-2.0`: No thinking tags detected

### Citation

```bibtex
@article{ahmad2025opencodereasoning,
  title={Opencodereasoning: Advancing data distillation for competitive coding},
  author={Ahmad, Wasi Uddin and Narenthiran, Sean and Majumdar, Somshubra and Ficek, Aleksander and Jain, Siddhartha and Huang, Jocelyn and Noroozi, Vahid and Ginsburg, Boris},
  journal={arXiv preprint arXiv:2504.01943},
  year={2025}
}
```



---

## RadiusScorer

### Overview

The **Radius Scorer** is a diversity metric designed to quantify the distributional spread of a dataset in its embedding space. Unlike per-sample scorers, Radius operates at the **dataset level**, computing a single aggregate score that reflects how widely distributed the data points are across all embedding dimensions, proposed by [Lai et al., 2020](https://arxiv.org/abs/2003.08529)

### Metric Definition:

* **Definition:**

  Radius is computed as the **geometric mean of standard deviations** across all embedding dimensions:
  
  \[ \text{Radius} = \left(\prod_{i=1}^{n} \sigma_i\right)^{1/n} = \exp\left(\frac{1}{n}\sum_{i=1}^{n} \log(\sigma_i)\right) \]
  
  where \( \sigma_i \) is the standard deviation of all data points along the \( i \)-th embedding dimension, and \( n \) is the total number of dimensions.

* **Explanation:**
  * A **higher Radius value** indicates that data points are **more widely distributed** across the embedding space, suggesting **higher diversity**.
  * A **lower Radius value** indicates that data points are **more concentrated**, suggesting **lower diversity** or higher homogeneity.
  * The geometric mean is preferred over the arithmetic mean because it is less sensitive to outlier dimensions and better captures the overall distributional balance.

### YAML Configuration

```yaml
name: RadiusScorer
embedding_path: path/to/embeddings.npy
max_workers: 8
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"RadiusScorer"` | Identifier for the scorer, used in logs and progress displays |
| `embedding_path` | string | required | Path to a pre-computed embedding file in `.npy` format. The file should contain a 2D NumPy array of shape `(num_samples, embedding_dim)`, where each row corresponds to the embedding vector of a data sample |
| `max_workers` | integer | Number of CPU cores | Number of worker processes for parallel computation of dimension-wise standard deviations. Parallel processing is automatically enabled when `max_workers > 1` and `embedding_dim > max_workers * 10`. Set to `1` to disable parallelization |

### Underlying Model

RadiusScorer does **not require a specific language model** for inference. Instead, it operates on **pre-computed embeddings** that must be generated in advance using an embedding model of your choice. 

**Note**: The embeddings must be saved as a NumPy `.npy` file with shape (N, D) where N matches the number of samples in your dataset and D is the embedding dimension. The order of embeddings must correspond to the order of samples in your dataset file.

### Generating Embeddings

To generate the required embedding file for RadiusScorer, you can use the provided `embed.py` script located at:

```bash
data_scorer/model_based/utils/embed.py
```

#### Usage Example

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

#### Script Parameters

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

#### Key Features

- **Batch Processing**: Processes large datasets efficiently using batched tokenization and embedding computation
- **Automatic Truncation**: Handles long texts by truncating to the specified `max_tokens` limit
- **vLLM Integration**: Uses vLLM for fast and memory-efficient embedding generation with GPU acceleration
- **Flexible Field Extraction**: Supports extracting and concatenating multiple fields from JSONL data
- **Progress Tracking**: Displays progress bars using tqdm for both tokenization and embedding stages

#### Output Format

The script generates a NumPy `.npy` file containing embeddings in float64 format with shape (N, D), where:
- N = number of samples in your input dataset
- D = embedding dimension of the chosen model

This output file can be directly used as the `embedding_path` parameter in the RadiusScorer configuration.

### Scoring Process

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

### Output Format

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

### Citation

```bibtex
@article{lai2020diversity,
  title={Diversity, density, and homogeneity: Quantitative characteristic metrics for text collections},
  author={Lai, Yi-An and Zhu, Xuan and Zhang, Yi and Diab, Mona},
  journal={arXiv preprint arXiv:2003.08529},
  year={2020}
}
```


---

## StrLengthScorer

### Overview

The **String Length Scorer** is a heuristic-based evaluation tool designed to measure the total character length of specified fields in SFT (Supervised Fine-Tuning) data samples. This scorer provides a simple, efficient, and model-free metric to assess the verbosity or content volume of training data. It is particularly useful for filtering data based on length requirements, identifying overly brief or excessively verbose samples, or analyzing dataset characteristics.

Unlike model-based approaches, the String Length Scorer requires no GPU resources and operates purely through string operations, making it extremely fast and scalable for large datasets. It supports multiprocessing for efficient parallel computation across multiple CPU cores.

### Metric Definition:

* **Definition:** 

  `Str_Length = len(concatenated_text)`
  
  Where `concatenated_text` is formed by joining specified fields (e.g., `instruction`, `input`, `output`) with newline separators.

* **Explanation:** This metric quantifies the total number of characters in the combined text of selected fields:
  
  * A **higher Str_Length value** indicates more verbose or content-rich data samples, which may contain more detailed instructions, longer responses, or comprehensive information.
  * A **lower Str_Length value** suggests concise or minimal content, which may indicate simple instructions or brief responses.

### YAML Configuration

```yaml
name: StrLengthScorer
fields:
  - instruction
  - input
  - output
max_workers: 8
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"StrLengthScorer"` | Identifier for the scorer |
| `fields` | list of strings | `["instruction", "input", "output"]` | Specifies which data fields should be included in the length calculation. The fields are concatenated with newline separators (`\n`) before measuring length |
| `max_workers` | integer | CPU core count | The number of parallel worker processes to use for processing the dataset. Higher values can significantly speed up processing on multi-core systems |

### Underlying Model

This scorer does **not require any language model**. 

### Scoring Process

1. **Field Extraction**: For each data item, the scorer extracts the values of specified fields (e.g., `instruction`, `input`, `output`) from the JSON record.

2. **Text Concatenation**: All extracted field values are converted to strings and joined together using newline characters (`\n`) as separators. Empty or missing fields are automatically excluded.

3. **Length Calculation**: The total character count of the concatenated text is computed using Python's built-in `len()` function. This includes all characters: letters, numbers, spaces, punctuation, and special characters.

4. **Parallel Processing**: To handle large datasets efficiently, the scorer uses Python's `ProcessPoolExecutor` to distribute the workload across multiple CPU cores. Each worker process independently computes scores for a subset of data items.

5. **Progress Tracking**: A progress bar (via `tqdm`) displays real-time processing status, showing the number of items processed and estimated time remaining.

6. **Error Handling**: If an error occurs during processing of any individual item (e.g., malformed JSON), the scorer returns a score of `0` for that item and logs the error, allowing the pipeline to continue.

### Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 1247
}
```

- `id`: The unique identifier of the data item, extracted from the `"id"` field in the input JSON
- `score`: The total character length of the concatenated text from specified fields (non-negative integer)

### Citation

```bibtex
@misc{opendataarena_tool_2025,
  author       = {OpenDataArena},
  title        = {{OpenDataArena-Tool}},
  year         = {2025},
  url          = {https://github.com/OpenDataArena/OpenDataArena-Tool},
  note         = {GitHub repository},
  howpublished = {\url{https://github.com/OpenDataArena/OpenDataArena-Tool}},
}
```


---

## ThinkOrNotScorer

### Overview

The **ThinkOrNot Scorer** is a lightweight, rule-based evaluation tool designed to identify whether a response contains explicit reasoning traces in reasoning-augmented datasets. Motivated by the methodology in [OpenCodeReasoning (Ahmad et al., 2025)](https://arxiv.org/pdf/2504.01943), this scorer performs binary classification to detect the presence of thinking tags (`<think>...</think>` or `<redacted_reasoning>...</redacted_reasoning>`) in model-generated responses.

Unlike more sophisticated scorers that evaluate reasoning quality, ThinkOrNotScorer focuses solely on **presence detection**, making it extremely fast and suitable for large-scale dataset preprocessing, data filtering, and quality control in reasoning distillation pipelines.

### Metric Definition:

* **Definition:** 

  The ThinkOrNot Score is a binary indicator:
  
  * **Score = 1**: The response **contains** thinking tags (`<think>...</think>` or `<redacted_reasoning>...</redacted_reasoning>`)
  * **Score = 0**: The response **does not contain** any thinking tags

* **Explanation:** This metric provides a simple presence/absence classification:
  
  * A **score of 1** indicates the response includes explicit reasoning traces, which is typical for distilled reasoning models or chain-of-thought augmented data.
  * A **score of 0** indicates the response is a standard format without explicit reasoning sections, either because the model directly provided an answer or the thinking tags were not generated.

### YAML Configuration

```yaml
name: ThinkOrNotScorer
field: output
max_workers: 8
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"ThinkOrNotScorer"` | Identifier for the scorer |
| `field` | string | `"output"` | The field name in the dataset to examine (model-generated responses) |
| `max_workers` | integer | CPU count | Number of parallel worker processes for scoring |


### Underlying Model

This scorer **does not require any language model**. It operates entirely through rule-based pattern matching using compiled regular expressions. The scorer detects the following tag patterns (case-insensitive):
* `<think>` and `</think>`
* `<think >` and `</think >` (with optional trailing spaces)
* `<redacted_reasoning>` and `</redacted_reasoning>`
* `<redacted_reasoning >` and `</redacted_reasoning >` (with optional trailing spaces)

### Scoring Process

1. **Extract Text**: Read the specified field (e.g., `output`) from each data item in the dataset

2. **Validate Field**: Check if the field exists and contains valid string content (missing or empty field returns score 0)

3. **Pattern Matching**: Search for thinking tag patterns using pre-compiled regular expressions with case-insensitive matching

4. **Binary Classification**: If any thinking tag is found, return score 1; otherwise return score 0

5. **Parallel Processing**: Distribute data items across multiple worker processes using Python's ProcessPoolExecutor for efficient large-scale processing

### Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 1.0
}
```

- `id`: The unique identifier of the data item (defaults to "unknown" if not present)
- `score`: The binary ThinkOrNot score (1.0 for reasoning present, 0.0 for standard response)

### Citation

```bibtex
@misc{opendataarena_tool_2025,
  author       = {OpenDataArena},
  title        = {{OpenDataArena-Tool}},
  year         = {2025},
  url          = {https://github.com/OpenDataArena/OpenDataArena-Tool},
  note         = {GitHub repository},
  howpublished = {\url{https://github.com/OpenDataArena/OpenDataArena-Tool}},
}
```


---

## TokenEntropyScorer

### Overview

The **Token Entropy Scorer** is a statistical evaluation tool designed to measure the lexical diversity and information richness of SFT (Supervised Fine-Tuning) data by computing token-level entropy. This scorer analyzes the distribution of tokens in instruction-response pairs and quantifies their unpredictability using Shannon entropy. Higher entropy scores indicate more diverse token usage and potentially richer information content, while lower scores suggest repetitive or predictable token patterns.

Unlike model-based approaches, this scorer relies purely on statistical analysis of token distributions using the `tiktoken` tokenizer, making it computationally efficient and model-agnostic.

### Metric Definition:

* **Definition:** 

  Token Entropy is computed using Shannon's entropy formula:
  
  ```
  H(X) = -Σ p(x) * log₂(p(x))
  ```
  
  where `p(x)` is the probability (frequency) of each unique token in the text.

* **Explanation:** This metric quantifies the unpredictability or diversity of token usage in the data:
  
  * A **higher Token Entropy score** indicates greater lexical diversity, with tokens distributed more evenly across the vocabulary. This suggests the data contains varied and information-rich content.
  * A **lower Token Entropy score** suggests repetitive token usage or limited vocabulary diversity, potentially indicating redundant or template-like content.

### YAML Configuration

```yaml
name: TokenEntropyScorer
encoder: o200k_base
max_workers: 8
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"TokenEntropyScorer"` | Identifier for the scorer instance. Used for logging and progress tracking |
| `encoder` | string | `"o200k_base"` | The tiktoken encoder to use for tokenization. Common options: `o200k_base` (GPT-4o and newer), `cl100k_base` (GPT-4, GPT-3.5-turbo), `p50k_base` (older GPT-3 models) |
| `max_workers` | integer | CPU cores | Number of parallel processes to use for scoring. Higher values speed up processing but consume more CPU resources |

### Underlying Model

This scorer does **not require a deep learning model**. It uses the `tiktoken` library for tokenization, which provides efficient byte-pair encoding (BPE) tokenizers compatible with OpenAI models. The tokenizer is used purely for splitting text into tokens; no neural network inference is involved.

### Scoring Process

The Token Entropy Scorer evaluates each data sample through the following steps:

1. **Text Concatenation:** For each data item, the instruction, optional input, and output (response) are concatenated:
   ```
   text = instruction + '\n' + input + '\n' + output
   ```
   (If no input field exists, it is omitted)

2. **Tokenization:** The concatenated text is tokenized using the specified tiktoken encoder, producing a sequence of token IDs:
   ```python
   tokens = encoder.encode(text, disallowed_special=())
   ```

3. **Token Frequency Analysis:** The frequency of each unique token is counted to build a probability distribution:
   ```
   p(token_i) = count(token_i) / total_tokens
   ```

4. **Entropy Calculation:** Shannon entropy is computed across all unique tokens:
   ```
   H = -Σ p(token_i) * log₂(p(token_i))
   ```

5. **Parallel Processing:** When evaluating datasets, the scorer uses `ProcessPoolExecutor` to distribute work across multiple CPU cores for efficient batch processing.

### Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 8.234
}
```

- `id`: The unique identifier of the data sample, extracted from the input data's `id` field
- `score`: The computed token entropy value. Higher values indicate greater lexical diversity. A score of 0.0 indicates either empty text or an error during processing

### Citation

```bibtex
@inproceedings{zhuang2025meta,
  title        = {Meta-rater: A multi-dimensional data selection method for pre-training language models},
  author       = {Zhuang, Xinlin and Peng, Jiahui and Ma, Ren and Wang, Yinfan and Bai, Tianyi and Wei, Xingjian and Jiantao, Qiu and Zhang, Chi and Qian, Ying and He, Conghui},
  booktitle    = {Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages        = {10856--10896},
  year         = {2025}
}
```


---

## TokenLengthScorer

### Overview

The **Token Length Scorer** is a lightweight, efficient evaluation tool designed to measure the token length of SFT (Supervised Fine-Tuning) data samples. Unlike model-based scorers, this metric provides a deterministic, tokenization-based measurement that quantifies the raw token count of specified fields in each data sample. Token length serves as a fundamental characteristic for data selection, filtering, and quality control in instruction-tuning pipelines, helping practitioners identify samples that are too short (potentially low-information) or too long (potentially noisy or verbose).

This scorer leverages OpenAI's **tiktoken** library to efficiently tokenize text and supports parallel processing for large-scale datasets.

### Metric Definition:

* **Definition:** 
  
  Token_Length = number of tokens after encoding the concatenated text of specified fields using a tiktoken encoder.

* **Explanation:** This metric counts the total number of tokens in a data sample after concatenating the specified fields (e.g., instruction, input, output) with newline separators.
  
  * A **higher Token Length** indicates a longer sample, which may contain more comprehensive information but could also be verbose or redundant.
  * A **lower Token Length** suggests a more concise sample, which could be efficient but might lack necessary detail.

### YAML Configuration

```yaml
name: TokenLengthScorer
encoder: o200k_base
fields:
  - instruction
  - input
  - output
max_workers: 8
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"TokenLengthScorer"` | Identifier for the scorer |
| `encoder` | string | `"o200k_base"` | The tiktoken encoder to use for tokenization. Common options: `o200k_base` (GPT-4o), `cl100k_base` (GPT-4, GPT-3.5-turbo), `p50k_base` (Codex models). Automatically falls back to `o200k_base` if the specified encoder fails to load. |
| `fields` | list | `["instruction", "input", "output"]` | The fields to extract from each data sample for token counting. Fields are concatenated with newline separators (`\n`) before tokenization. Only non-empty fields present in the data item are included. |
| `max_workers` | integer | Number of CPU cores | The number of parallel worker processes for data processing. Higher values can significantly speed up processing for large datasets. Recommended range: 4-16 depending on CPU availability and dataset size. |

### Underlying Model

This scorer **does not use a language model**. Instead, it relies on the **tiktoken** tokenization library, which provides fast and accurate byte-pair encoding (BPE) tokenizers used by OpenAI models. The default encoder is `o200k_base`, which is the tokenizer used by GPT-4o and other modern OpenAI models.

### Scoring Process

1. **Configuration Validation:** The scorer validates the configuration and sets defaults for `encoder`, `fields`, and `max_workers` if not specified.

2. **Field Extraction:** For each data sample, the specified fields (e.g., `instruction`, `input`, `output`) are extracted and filtered for non-empty values.

3. **Text Concatenation:** All extracted field values are converted to strings and concatenated using newline separators (`\n`).

4. **Tokenization:** The concatenated text is encoded using the specified tiktoken encoder with `disallowed_special=()` to handle special tokens properly.

5. **Token Counting:** The length of the encoded token list is computed as the final score.

6. **Parallel Processing:** The scorer uses `ProcessPoolExecutor` to process multiple samples in parallel, with a progress bar (tqdm) to track evaluation progress.

7. **Error Handling:** If any sample fails during processing, the scorer returns a score of 0 with an error marker.

### Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 256
}
```

- `id`: The unique identifier of the data sample, extracted from the `id` field in the input data. Returns `"unknown"` if the `id` field is missing or processing fails.
- `score`: The total token count of the concatenated text from specified fields. Returns `0` if tokenization fails or the sample is empty.

### Citation

```bibtex
@misc{opendataarena_tool_2025,
  author       = {OpenDataArena},
  title        = {{OpenDataArena-Tool}},
  year         = {2025},
  url          = {https://github.com/OpenDataArena/OpenDataArena-Tool},
  note         = {GitHub repository},
  howpublished = {\url{https://github.com/OpenDataArena/OpenDataArena-Tool}},
}
```


---

## TreeInstructScorer

### Overview

The **TreeInstruct Complexity Scorer** is a syntax-tree-based evaluation tool designed to measure instruction complexity through structural analysis of text. Inspired by the paper [Zhao et al., 2023](https://arxiv.org/abs/2308.05696), this scorer quantifies complexity by analyzing the syntactic dependency tree structure rather than relying on surface-level features. The core insight is that more complex instructions naturally exhibit deeper and more elaborate syntactic trees with more nodes and hierarchical levels.

Unlike token-count-based metrics, TreeInstruct captures the **structural complexity** of instructions by examining how words relate to each other grammatically, providing a linguistically grounded measure of instruction difficulty.

### Metric Definition:

* **Definition:**
  
  1. Parse the text using dependency parsing to construct a syntactic tree
  2. Extract two structural metrics:
     - **TreeInstruct_Nodes**: Total number of nodes in the syntactic tree
     - **TreeInstruct_Depth**: Maximum depth (number of hierarchical levels) of the syntactic tree

* **Explanation:** The complexity is measured through syntactic tree structure:
  
  * **Higher node counts** indicate more words and relationships, suggesting greater information density and complexity
  * **Greater tree depth** reflects deeper nested grammatical structures, indicating more sophisticated linguistic constructions
  
  These metrics align with the Tree-Instruct methodology, which systematically enhances instruction complexity by adding nodes to semantic trees.

### YAML Configuration

```yaml
name: TreeInstructScorer
model: en_core_web_sm
max_workers: 8
text_fields:
  - instruction
  - input
  - output
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"TreeInstructScorer"` | Identifier for the scorer |
| `model` | string | `"en_core_web_sm"` | spaCy language model for dependency parsing (Options: `en_core_web_sm`, `en_core_web_md`, `en_core_web_lg` for English; for other languages, use corresponding spaCy models like `zh_core_web_sm` for Chinese) |
| `max_workers` | integer | CPU count | Number of parallel processes for scoring (higher values speed up processing but consume more memory; recommended to set to number of CPU cores available) |
| `text_fields` | list | `["instruction", "input", "output"]` | List of fields to concatenate for analysis (concatenates specified fields with newlines before parsing; customize based on your dataset schema) |

### Underlying Model

The scorer uses **spaCy** NLP models for syntactic dependency parsing. spaCy is an industrial-strength natural language processing library that provides accurate and efficient linguistic analysis.

**Default Model:** `en_core_web_sm` (English, small model)

**Alternative Models:**
- `en_core_web_md`: Medium English model (more accurate)
- `en_core_web_lg`: Large English model (most accurate)
- Language-specific models for non-English text

**Installation:**
```bash
python -m spacy download en_core_web_sm
```

For other languages or model sizes, visit [spaCy's model documentation](https://spacy.io/models).

### Scoring Process

1. **Text Preparation:** Concatenate text from specified fields (`text_fields`) with newline separators

2. **Dependency Parsing:** Process the text through the spaCy model to construct a syntactic dependency tree, where:
   - Each word becomes a node
   - Edges represent grammatical relationships
   - The sentence root serves as the tree root

3. **Node Counting:** Recursively traverse the tree starting from the root(s) to count all nodes (words)

4. **Depth Calculation:** Recursively compute the maximum depth by finding the longest path from root to leaf
   - A single word has depth 1
   - Depth increases by 1 for each level of nested dependencies

5. **Parallel Processing:** For large datasets, the scorer distributes work across multiple processes using `ProcessPoolExecutor` for efficient computation

### Output Format

For each input sample, the scorer returns:

```json
{
  "id": "sample_001",
  "TreeInstruct_Nodes": 45,
  "TreeInstruct_Depth": 8
}
```

- `id`: Unique identifier from the original data item (preserved from input)
- `TreeInstruct_Nodes`: Total number of nodes (tokens) in the syntactic tree (Range: [0, ∞); higher values indicate more complex syntactic structures with more words and relationships)
- `TreeInstruct_Depth`: Maximum depth of the syntactic tree (Range: [0, ∞); higher values indicate deeper nested grammatical constructions)

**Note:** If parsing fails or text is empty, both metrics return 0, with an optional `error` field containing the error message.

### Citation

```bibtex
@article{zhao2023preliminary,
  title={A preliminary study of the intrinsic relationship between complexity and alignment},
  author={Zhao, Yingxiu and Yu, Bowen and Hui, Binyuan and Yu, Haiyang and Huang, Fei and Li, Yongbin and Zhang, Nevin L},
  journal={arXiv preprint arXiv:2308.05696},
  year={2023}
}
```


---

## TsPythonScorer

### Overview

The **TsPythonScorer** is a syntax-based code quality scorer that validates Python code correctness using static analysis. This scorer leverages the Tree-sitter parsing library to check whether Python code snippets are syntactically valid. It is designed to filter out code samples with syntax errors, which are typically low-quality or incomplete code fragments unsuitable for model training.

The scorer supports both plain Python code and code embedded within Markdown code blocks (triple backticks format), making it versatile for processing various data formats commonly found in code-related datasets. It employs multiprocessing to efficiently handle large-scale datasets.

This scorer aligns with the data quality control principles discussed in the Seed-Coder paper, where ensuring syntactic correctness is a fundamental prerequisite for code pretraining data.

### Metric Definition:

* **Definition:** 

  The scorer outputs a **binary score (0.0 or 1.0)** indicating whether the code is syntactically correct:
  
  - **1.0**: All Python code in the sample is syntactically valid
  - **0.0**: The sample contains syntax errors, is empty, or cannot be parsed

* **Explanation:** 
  
  For samples containing multiple Markdown code blocks, **all blocks must be syntactically correct** for the sample to receive a score of 1.0. If any code block fails parsing, the entire sample is scored as 0.0.

* **Key Features:**
  
  - **Format-agnostic:** Handles both plain Python code and Markdown-embedded code blocks
  - **Strict validation:** Any syntax error results in a zero score
  - **Efficient processing:** Leverages multiprocessing for scalable batch evaluation

### YAML Configuration

```yaml
scorers:
  - name: ts_python_syntax
    type: TsPythonScorer
    config:
      field: "output"
      max_workers: 16
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `field` | string | `"output"` | Specifies the JSON field name containing the code to be evaluated |
| `max_workers` | integer | CPU count | Number of parallel worker processes for batch evaluation. Higher values increase processing speed but consume more system resources |

### Underlying Model

This scorer does not use any language model. Instead, it relies on the **Tree-sitter** parsing library with the `tree-sitter-python` grammar for syntactic analysis. Tree-sitter is a parser generator tool that builds concrete syntax trees and is widely used in code editors and static analysis tools for syntax validation.

### Scoring Process

1. **Text Extraction**: Extract the text content from the specified field (default: `"output"`) in the JSON data item

2. **Markdown Code Block Detection**: 
   - Attempt to identify Markdown code blocks using the pattern: ` ```[language]\n...\n``` `
   - If code blocks are found, extract all of them for individual validation
   - If no code blocks are detected, treat the entire text as Python code

3. **Syntax Parsing**:
   - For each code snippet (either extracted blocks or the full text), use Tree-sitter to parse the Python code
   - Check if the resulting Abstract Syntax Tree (AST) contains any error nodes

4. **Score Determination**:
   - If all code snippets parse successfully without errors: **score = 1.0**
   - If any snippet contains syntax errors or is empty: **score = 0.0**
   - If the field is missing or invalid: **score = 0.0**

5. **Parallel Processing**: When evaluating datasets in batch mode, the scorer distributes samples across multiple worker processes for efficient parallel processing

### Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 1.0
}
```

- `id`: The unique identifier from the input data item. If missing, defaults to `"unknown"`
- `score`: Binary score indicating syntax validity (1.0 = syntactically correct, 0.0 = contains syntax errors)

### Citation

```bibtex
@misc{tree_sitter,
  title        = {tree-sitter},
  author       = {Max Brunsfeld and contributors},
  year         = {2018},
  howpublished = {\url{https://github.com/tree-sitter/tree-sitter}},
  note         = {GitHub repository},
}
```


---

## UniqueNgramScorer

### Overview

The **Unique N-gram Scorer** is a statistical evaluation tool designed to measure the lexical diversity of SFT (Supervised Fine-Tuning) data by calculating the ratio of unique n-grams to total n-grams in the combined text. This metric provides insight into the linguistic richness and repetitiveness of instruction-response pairs. Higher unique n-gram ratios indicate more diverse vocabulary and less repetitive text patterns, which is generally desirable for training data quality.

The scorer processes text by combining the instruction, input (if present), and output fields, then tokenizes the text and computes n-gram statistics. It supports parallel processing for efficient evaluation of large datasets.

### Metric Definition:

* **Definition:** 

  ```
  Unique_Ngram_Score = |unique n-grams| / |total n-grams|
  ```

  where n-grams are extracted from the lowercase tokenized text of combined instruction, input, and output fields.

* **Explanation:** This metric quantifies the lexical diversity of text by measuring the proportion of distinct n-gram patterns.
  
  * A **higher Unique N-gram Score** (closer to 1) indicates **greater lexical diversity** with minimal repetition, suggesting rich and varied language use.
  * A **lower Unique N-gram Score** (closer to 0) indicates **more repetitive patterns**, suggesting redundant or formulaic language.
  * A score of **0** is assigned when the text has fewer tokens than the specified n value.

### YAML Configuration

```yaml
name: UniqueNgramScorer
n: 2
max_workers: 8
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"UniqueNgramScorer"` | Identifier for the scorer |
| `n` | integer | `2` | The n-gram size to use for analysis. `n=1` (unigrams) measures unique word diversity, `n=2` (bigrams) measures unique word pair diversity, `n=3` (trigrams) measures unique three-word phrase diversity. Higher values capture longer linguistic patterns but require more tokens in the text |
| `max_workers` | integer | CPU count | Number of parallel workers for multiprocessing. Automatically defaults to the system's CPU core count if not specified or invalid. Higher values can speed up processing but may increase memory usage |

### Underlying Model

This scorer does **not require a deep learning model**. It uses **NLTK's punkt tokenizer** for word tokenization, which is a rule-based, language-independent tokenizer. The tokenizer is automatically downloaded if not already present in the system.

### Scoring Process

1. **Text Extraction:** For each data item, extract the `instruction`, `input` (optional), and `output` fields.

2. **Text Combination:** Concatenate the fields with newline separators:
   - If input exists: `instruction + '\n' + input + '\n' + output`
   - Otherwise: `instruction + '\n' + output`

3. **Text Normalization:** Convert the combined text to lowercase to ensure case-insensitive analysis.

4. **Tokenization:** Use NLTK's `word_tokenize` to split the text into tokens.

5. **N-gram Generation:** Generate all n-grams from the token sequence using the specified `n` value.

6. **Uniqueness Calculation:** 
   - Count the total number of n-grams
   - Count the number of unique n-grams (using a set)
   - Calculate the ratio: `unique_count / total_count`

7. **Edge Cases:**
   - If the token count is less than `n`, return a score of 0.0
   - If no n-grams can be generated, return a score of 0.0

8. **Parallel Processing:** The scorer uses `ProcessPoolExecutor` to process multiple data items in parallel, with progress tracking via `tqdm`.

### Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 0.87
}
```

- `id`: The unique identifier of the data item, extracted from the `id` field in the input data. Returns an empty string if the `id` field is not present
- `score`: The Unique N-gram Ratio, ranging from 0.0 to 1.0. **1.0** indicates every n-gram in the text is unique (maximum diversity), **0.5** indicates half of the n-grams are unique, **0.0** indicates all n-grams are identical (maximum repetition) or text has fewer than `n` tokens
- `error` (optional): Only present if an error occurred during processing. Contains the error message for debugging purposes

### Citation

```bibtex
@inproceedings{zhuang2025meta,
  title={Meta-rater: A multi-dimensional data selection method for pre-training language models},
  author={Zhuang, Xinlin and Peng, Jiahui and Ma, Ren and Wang, Yinfan and Bai, Tianyi and Wei, Xingjian and Jiantao, Qiu and Zhang, Chi and Qian, Ying and He, Conghui},
  booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={10856--10896},
  year={2025}
}
```



---

## UniqueNtokenScorer

### Overview

The **Unique N-token Scorer** is a statistical evaluation tool designed to measure the lexical diversity of SFT (Supervised Fine-Tuning) data by calculating the ratio of unique token-level n-grams to total token-level n-grams in the combined text. Unlike word-based n-gram analysis, this scorer operates at the **token level** using tiktoken encoders, which provides a more granular and tokenizer-aware assessment of text diversity that aligns with how modern language models process text.

This metric is particularly valuable for evaluating training data quality from a model-centric perspective, as it captures repetitive patterns at the subword level. Higher unique token n-gram ratios indicate more diverse token sequences and less repetitive patterns, which is generally desirable for training data quality. The scorer supports parallel processing for efficient evaluation of large datasets.

### Metric Definition:

* **Definition:** 

  ```
  Unique_Ntoken_Score = |unique token n-grams| / |total token n-grams|
  ```

  where token n-grams are extracted from the tokenized text (using tiktoken encoder) of combined instruction, input, and output fields.

* **Explanation:** This metric quantifies the lexical diversity of text at the token level by measuring the proportion of distinct token n-gram patterns.

  * A **higher Unique N-token Score** (closer to 1) indicates **greater token-level diversity** with minimal repetition, suggesting rich and varied token sequences.
  * A **lower Unique N-token Score** (closer to 0) indicates **more repetitive token patterns**, suggesting redundant or formulaic sequences at the subword level.
  * A score of **0** is assigned when the text has fewer tokens than the specified n value.

### YAML Configuration

```yaml
name: UniqueNtokenScorer
encoder: o200k_base
n: 2
max_workers: 8
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"UniqueNtokenScorer"` | Identifier for the scorer. Used for logging and result tracking. |
| `encoder` | string | `"o200k_base"` | The tiktoken encoder to use for tokenization. Options: `"o200k_base"` (GPT-4o and newer models, 200k vocabulary), `"cl100k_base"` (GPT-4 and GPT-3.5-turbo, 100k vocabulary), `"p50k_base"` (Codex and older GPT-3 models, 50k vocabulary), `"r50k_base"` (Older GPT-3 models like davinci, 50k vocabulary). The choice of encoder should align with the tokenizer of the target model for training. |
| `n` | integer | `2` | The n-gram size to use for token-level analysis. `n=1` measures unique token diversity (unigrams), `n=2` measures unique token pair diversity (bigrams), `n=3` measures unique three-token sequence diversity (trigrams). Higher values of `n` capture longer token patterns but require more tokens in the text. |
| `max_workers` | integer | CPU count | The number of parallel workers for multiprocessing. Automatically defaults to the system's CPU core count if not specified or invalid. Higher values can speed up processing but may increase memory usage. Recommended to set based on available CPU cores and memory. |

### Underlying Model

This scorer does **not require a deep learning model**. It uses **tiktoken**, OpenAI's fast byte pair encoding (BPE) tokenizer library, for token-level text processing. The default encoder is `o200k_base`, which is used by GPT-4o and newer models. Users can specify alternative tiktoken encoders based on their target model's tokenization scheme.

Tiktoken is a highly efficient tokenization library that provides:
- Fast encoding and decoding
- Multiple pre-trained BPE vocabularies
- Consistency with OpenAI's language models

### Scoring Process

1. **Text Extraction:** For each data item, extract the `instruction`, `input` (optional), and `output` fields.

2. **Text Combination:** Concatenate the fields with newline separators:
   - If input exists: `instruction + '\n' + input + '\n' + output`
   - Otherwise: `instruction + '\n' + output`

3. **Token Encoding:** Use tiktoken's encoder to convert the text into a sequence of token IDs based on the specified encoder (e.g., `o200k_base`).

4. **Token N-gram Generation:** Generate all token-level n-grams from the token ID sequence:
   - For each position `i` in the token list, extract a tuple of `n` consecutive token IDs: `(token[i], token[i+1], ..., token[i+n-1])`
   - Continue until reaching the end of the token sequence

5. **Uniqueness Calculation:** 
   - Count the total number of token n-grams
   - Count the number of unique token n-grams (using a set of tuples)
   - Calculate the ratio: `unique_count / total_count`

6. **Edge Cases:**
   - If the token count is less than `n`, return a score of 0.0
   - If encoding fails, return a score of 0.0 and log the error

7. **Parallel Processing:** The scorer uses `ProcessPoolExecutor` to process multiple data items in parallel, with progress tracking via `tqdm`.

### Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 0.89
}
```

- `id`: The unique identifier of the data item, extracted from the `id` field in the input data. Returns an empty string if the `id` field is not present.
- `score`: The Unique N-token Ratio, ranging from 0.0 to 1.0. A value of **1.0** indicates every token n-gram in the text is unique (maximum diversity), **0.5** indicates half of the token n-grams are unique, and **0.0** indicates all token n-grams are identical (maximum repetition) or text has fewer than `n` tokens.
- `error` (optional): Only present if an error occurred during processing. Contains the error message for debugging purposes.

### Citation

```bibtex
@misc{opendataarena_tool_2025,
  author       = {OpenDataArena},
  title        = {{OpenDataArena-Tool}},
  year         = {2025},
  url          = {https://github.com/OpenDataArena/OpenDataArena-Tool},
  note         = {GitHub repository},
  howpublished = {\url{https://github.com/OpenDataArena/OpenDataArena-Tool}},
}
```


---

## VendiScorer

### Overview

The **Vendi Score** is a diversity evaluation metric for machine learning that measures the intrinsic diversity of a dataset without requiring any reference distribution or pre-trained classifier. Proposed by Friedman and Dieng (2023), the Vendi Score extends concepts from ecology and quantum statistical mechanics to evaluate the effective number of unique elements in a sample.

Unlike reference-based metrics (e.g., FID) or label-dependent metrics (e.g., Inception Score), the Vendi Score is a **reference-free, general-purpose diversity metric** that can be applied to any domain where similarity between samples can be defined. It takes as input a collection of embeddings and a user-specified similarity function, making it highly flexible for evaluating dataset diversity across different modalities (text, images, molecules, etc.).

### Metric Definition:

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

### YAML Configuration

```yaml
name: VendiScorer
embedding_path: path/to/embeddings.npy
similarity_metric: cosine
max_workers: 8
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"VendiScorer"` | Identifier for the scorer |
| `embedding_path` | string | (required) | Path to a pre-computed embedding file in `.npy` format. The file should contain a 2D numpy array of shape `(num_samples, embedding_dim)`, where each row represents the embedding vector of a data sample |
| `similarity_metric` | string | `"cosine"` | The similarity function used to compute pairwise similarities between embeddings. Available options: `"cosine"`, `"euclidean"`, `"manhattan"`, `"dot_product"`, `"pearson"` |
| `max_workers` | integer | CPU count | Number of parallel processes to use for computation. This parameter controls the parallelization of similarity matrix computation |

### Underlying Model

VendiScorer does **not require a specific language model** for inference. Instead, it operates on **pre-computed embeddings** that must be generated in advance using an embedding model of your choice. 

**Note**: The embeddings must be saved as a NumPy `.npy` file with shape (N, D) where N matches the number of samples in your dataset and D is the embedding dimension. The order of embeddings must correspond to the order of samples in your dataset file.

### Generating Embeddings

To generate the required embedding file for VendiScorer, you can use the provided `embed.py` script located at:

```bash
data_scorer/model_based/utils/embed.py
```

#### Usage Example

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

#### Script Parameters

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

#### Key Features

- **Batch Processing**: Processes large datasets efficiently using batched tokenization and embedding computation
- **Automatic Truncation**: Handles long texts by truncating to the specified `max_tokens` limit
- **vLLM Integration**: Uses vLLM for fast and memory-efficient embedding generation with GPU acceleration
- **Flexible Field Extraction**: Supports extracting and concatenating multiple fields from JSONL data
- **Progress Tracking**: Displays progress bars using tqdm for both tokenization and embedding stages

#### Output Format

The script generates a NumPy `.npy` file containing embeddings in float64 format with shape (N, D), where:
- N = number of samples in your input dataset
- D = embedding dimension of the chosen model

This output file can be directly used as the `embedding_path` parameter in the VendiScorer configuration.


### Scoring Process

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

### Output Format

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

### Citation

```bibtex
@article{friedman2023vendi,
  title={The Vendi Score: A Diversity Evaluation Metric for Machine Learning},
  author={Friedman, Dan and Dieng, Adji Bousso},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2023}
}
```


---

## VocdDScorer

### Overview

The **VOCD-D (Vocabulary Diversity-D) Scorer** is a statistical evaluation tool designed to measure **lexical diversity** in text through a sophisticated mathematical modeling approach. Proposed by Malvern, Richards, Chipere, and Durán (2004), VOCD-D addresses the fundamental limitation of traditional Type-Token Ratio (TTR) measures: their sensitivity to text length.

The measure is based on a mathematical model that describes the relationship between types and tokens, and uses curve-fitting procedures to estimate a single parameter `D` that quantifies lexical diversity. This approach provides a robust, length-independent assessment of vocabulary richness that remains stable across varying text lengths, making it particularly valuable for comparing texts of different sizes.

Unlike model-based scorers, VOCD-D is a **statistical metric** that does not require any pre-trained models. It is particularly valuable for evaluating the linguistic complexity and vocabulary variety of instruction-following datasets, offering insights into the lexical sophistication of both instructions and responses.

### Metric Definition:

* **Definition:** VOCD-D is calculated through a curve-fitting procedure that models the relationship between types (unique words) and tokens (total words) in randomly sampled text segments of varying lengths. The algorithm:

  1. Takes random samples of tokens from the text at multiple sample sizes (typically from 35 to `ntokens`)
  2. For each sample size, calculates the average Type-Token Ratio across multiple trials (`within_sample`)
  3. Fits these observed TTR values to a mathematical model curve
  4. Extracts the parameter `D` that best describes the curve, representing the text's lexical diversity

* **Explanation:** The D parameter represents the **inherent lexical diversity** of the text, independent of its length:

  * A **higher D score** (typically ranging from 50-100+) indicates **greater lexical diversity**, meaning the text demonstrates rich and varied vocabulary usage with less repetition, suggesting sophisticated and diverse language use.
  * A **lower D score** (typically below 50) indicates **lower lexical diversity**, meaning the text relies on a more limited vocabulary with higher word repetition, suggesting simpler or more constrained language.
  * A **score of 0.0** indicates insufficient text for analysis (fewer tokens than the minimum required for sampling).

The mathematical modeling approach makes VOCD-D more robust than simple TTR measures, as it accounts for the stochastic nature of word occurrence and provides consistent results regardless of text length, making it suitable for comparing diverse text samples.

### YAML Configuration

```yaml
name: VocdDScorer
ntokens: 50
within_sample: 100
seed: 42
max_workers: 128
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"VocdDScorer"` | Identifier for the scorer |
| `ntokens` | integer | `50` | Maximum number of tokens used for sampling in the curve-fitting procedure. This determines the upper bound of sample sizes used when modeling the type-token relationship. Texts must contain at least this many words to receive a non-zero score. Typical values range from 35-100. |
| `within_sample` | integer | `100` | Number of random samples taken at each sample size during the curve-fitting procedure. Higher values (e.g., 100-200) provide more reliable estimates but take longer to compute. Lower values (e.g., 20-50) are faster but may be less stable. |
| `seed` | integer | `42` | Random seed for reproducible sampling. Ensures that repeated calculations on the same text yield identical results. Setting a fixed seed is recommended for reproducible research and consistent evaluation results. |
| `max_workers` | integer | `128` | Number of parallel worker processes for multiprocessing. Higher values can significantly speed up processing for large datasets but consume more CPU resources and memory. |

### Underlying Model

VOCD-D is a statistical metric based on mathematical modeling of the type-token relationship and does not require any pre-trained language models. The calculation is purely algorithmic, using curve-fitting techniques to estimate the diversity parameter D from observed type-token patterns in randomly sampled text segments. The implementation uses the [lexicalrichness](https://github.com/LSYS/lexicalrichness) Python library, which provides an efficient implementation of the VOCD algorithm originally proposed by Malvern et al.

### Scoring Process

1. **Text Concatenation**: For each data sample, the `instruction`, `input`, and `output` fields are concatenated into a single text string. If the `input` field is empty, it is omitted from concatenation.

2. **Text Validation**: Check if the text contains sufficient content. Empty or whitespace-only texts receive a score of 0.0. Texts with fewer tokens than the configured `ntokens` value receive a score of 0.0.

3. **Tokenization**: The text is split into tokens (words)

4. **Random Sampling**: For sample sizes ranging from 35 to `ntokens`, take `within_sample` random samples of each size and calculate the average Type-Token Ratio

5. **Curve Fitting**: Fit the observed TTR values to the theoretical VOCD model curve

6. **Parameter Estimation**: Extract the D parameter that best describes the curve, representing lexical diversity

7. **Error Handling**: If the calculation fails for any reason, the scorer returns 0.0 and logs the error

### Output Format

For each input sample, the scorer returns:

```json
{
  "id": "1",
  "score": 67.45
}
```

- `id`: The unique identifier of the data sample, extracted from the input data's `id` field. If no `id` is present, defaults to `"unknown"`.
- `score`: The VOCD-D lexical diversity score (D parameter) for the sample. Typically ranges from 0 to 100+, with most texts falling between 20-100. Higher values (typically 70-100+) indicate very rich and diverse vocabulary. Lower values (typically below 40) indicate limited vocabulary diversity. A value of 0.0 indicates empty text, insufficient text length (fewer than `ntokens` words), or processing error.
- `error` (optional): Only present if an error occurred during processing. Contains the error message for debugging purposes.

### Citation

```bibtex
@article{mccarthy2010mtld,
  title={MTLD, vocd-D, and HD-D: A validation study of sophisticated approaches to lexical diversity assessment},
  author={McCarthy, Philip M and Jarvis, Scott},
  journal={Behavior research methods},
  volume={42},
  number={2},
  pages={381--392},
  year={2010},
  publisher={Springer}
}
```


---
