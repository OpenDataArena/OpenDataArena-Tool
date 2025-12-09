# PartitionEntropyScorer

## Overview

The **Partition Entropy Scorer** is a statistical evaluation tool designed to measure the *diversity* of a data subset by analyzing its distribution across global clusters. This metric quantifies how uniformly the subset data is distributed among predefined clusters, providing insights into data heterogeneity without requiring any language model.

Unlike model-based scorers, this is a **dataset-level metric** that computes a single score for the entire subset rather than individual samples. It is particularly useful for evaluating whether a selected subset maintains diverse coverage across different data clusters.

## Metric Definition:

* **Definition:** 
  
  The Partition Entropy is calculated using the standard entropy formula:
  
  \[ H = -\sum_{i=1}^{k} p_i \log(p_i) \]
  
  where \( p_i \) is the proportion of samples in the subset belonging to cluster \( i \), and \( k \) is the number of clusters represented in the subset.

* **Interpretation:**
  
  * A **higher entropy** indicates that the subset is **more uniformly distributed** across clusters, suggesting **greater diversity**.
  * A **lower entropy** indicates that the subset is **concentrated** in fewer clusters, suggesting **lower diversity**.
  * The **normalized entropy** (entropy divided by \(\log(N)\), where \(N\) is the total number of global clusters) provides a scale-invariant measure between 0 and 1.

## YAML Configuration

```yaml
name: PartitionEntropyScorer
num_clusters: 100
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"PartitionEntropyScorer"` | Identifier for the scorer |
| `num_clusters` | integer | - | The total number of global clusters used for partitioning the entire dataset. This must be a positive integer and should match the clustering performed on your full dataset. This parameter is crucial for computing the normalized entropy, which measures subset diversity relative to the global cluster space. |

## Underlying Model

This scorer does **not require any language model**. It is a purely statistical method based on entropy calculation from cluster distributions.

## Scoring Process

1. **Input Validation**: Each data sample in the subset must contain a `cluster_id` field indicating its cluster assignment from a global clustering.

2. **Cluster Distribution Counting**: The scorer reads through the entire subset and counts how many samples belong to each cluster.

3. **Probability Calculation**: For each cluster represented in the subset, the probability \( p_i \) is computed as:
   \[ p_i = \frac{\text{count}_i}{\text{total samples in subset}} \]

4. **Entropy Computation**: The partition entropy is calculated using the formula:
   \[ H = -\sum_{i \in \text{subset clusters}} p_i \log(p_i) \]

5. **Normalization**: The normalized entropy is computed as:
   \[ H_{\text{normalized}} = \frac{H}{\log(N)} \]
   where \( N \) is the `num_clusters` parameter (total global clusters).

## Output Format

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

