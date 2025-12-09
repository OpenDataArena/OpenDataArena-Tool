# Task2VecScorer

## Overview

The **Task2Vec Diversity Coefficient** is a data quality metric designed to quantify the variability and diversity of natural language datasets. Proposed in [Miranda et al., 2023](https://arxiv.org/abs/2306.13840), this method moves beyond simple dataset scale considerations to assess the structural and semantic diversity of pre-training data. The diversity coefficient measures the expected distance between Task2Vec embeddings of data samples, providing a formal and interpretable measure of how varied the content is within a dataset.

Unlike scale-focused approaches, the diversity coefficient captures the richness and variety of data—characteristics that are crucial for training models with strong general capabilities and in-context learning abilities. Higher diversity scores indicate greater variability in the dataset, which has been shown to correlate with improved downstream model performance.

## Metric Definition:

* **Definition:** 

  The diversity coefficient is computed as the expected pairwise cosine distance between Task2Vec embeddings of randomly sampled batches from the dataset.

  ```
  Diversity = E[cosine_distance(embedding_i, embedding_j)]
  ```

  where each embedding is the diagonal of the Fisher Information Matrix (FIM) computed from a fixed probe network (GPT-2) fine-tuned on the target text.

* **Explanation:** 

  The diversity coefficient quantifies the level of structural and semantic diversity in natural language data:
  
  * A **higher diversity score** indicates greater variability in the dataset, suggesting more diverse concepts, richer vocabulary, and varied semantic content. This typically leads to better downstream performance on diverse evaluation tasks.
  * A **lower diversity score** suggests that the dataset contains more homogeneous or repetitive content with limited semantic variability.

## YAML Configuration

```yaml
name: Task2VecScorer
model: openai-community/gpt2
last_layer_only: false
max_length: 512
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"Task2VecScorer"` | Identifier for the scorer |
| `model` | string | `"openai-community/gpt2"` | Path to the GPT-2 model used as the probe network for computing Task2Vec embeddings. Can be a local path or a HuggingFace model identifier |
| `last_layer_only` | boolean | `false` | Whether to compute FIM using only the last layer parameters (`true`: faster computation, lower dimensionality) or all model parameters (`false`: more comprehensive representation) |
| `max_length` | integer | `512` | Maximum sequence length for tokenization. Texts longer than this value will be truncated |

## Underlying Model

The scorer uses **GPT-2** as the fixed probe network for computing Task2Vec embeddings. By default, it uses [openai-community/gpt2](https://huggingface.co/openai-community/gpt2), but users can specify other GPT-2 variants or local model paths through the `model` configuration parameter.

The probe network remains fixed across all samples to ensure embeddings are comparable. The model's final layer is fine-tuned with a next-token prediction objective to compute the Fisher Information Matrix for each text sample.

## Scoring Process

The Task2Vec scoring process consists of three main stages:

### 1. Fisher Information Matrix (FIM) Computation

For each text sample in the dataset:
  * The text is tokenized using the GPT-2 tokenizer (with truncation to `max_length`)
  * The model performs a forward pass with gradient computation enabled
  * For each token position, the gradient of the log probability with respect to model parameters is computed
  * The squared gradients are accumulated and averaged over the sequence length
  * The diagonal of the FIM serves as the Task2Vec embedding for that sample

### 2. Embedding Distance Calculation

After computing embeddings for all samples:
  * A pairwise cosine distance matrix is computed between all Task2Vec embeddings
  * For each sample, the average distance to all other samples is calculated (excluding self-distance)

### 3. Diversity Score Aggregation

The final diversity coefficient is computed as the mean of all pairwise average distances, representing the overall variability of the dataset.

**Note:** Unlike other scorers that assign individual scores to each sample, Task2VecScorer computes a single aggregate score for the entire dataset, making it suitable for dataset-level quality assessment rather than sample-level filtering.

## Output Format

The scorer returns a dictionary containing the following metrics:

```json
{
  "score": 0.8234,
  "num_samples": 1000,
  "num_anomalous": 5,
  "num_truncated": 120,
  "truncation_rate": 0.12,
  "last_layer_only": false,
  "embedding_dim": 124439808
}
```

- `score`: The diversity coefficient computed as the expected pairwise cosine distance between Task2Vec embeddings. Higher values indicate greater dataset diversity
- `num_samples`: Total number of valid samples successfully processed from the dataset
- `num_anomalous`: Count of records that could not be processed due to JSON parsing errors or missing required fields
- `num_truncated`: Number of samples that exceeded `max_length` and were truncated during tokenization
- `truncation_rate`: Proportion of truncated samples (num_truncated / num_samples)
- `last_layer_only`: Boolean indicating whether the FIM was computed using only the last layer parameters
- `embedding_dim`: The dimensionality of the Task2Vec embeddings, which depends on the number of parameters considered

## Citation

```bibtex
@article{miranda2023beyond,
  title={Beyond scale: The diversity coefficient as a data quality metric for variability in natural language data},
  author={Miranda, Brando and Lee, Alycia and Sundar, Sudharsan and Casasola, Allison and Koyejo, Sanmi},
  journal={arXiv preprint arXiv:2306.13840},
  year={2023}
}
```
