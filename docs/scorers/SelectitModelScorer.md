# SelectitModelScorer

## Overview

The **SelectIT Model Scorer** is an ensemble-based evaluation tool that leverages **model-level uncertainty** to assess the quality of instruction-tuning (SFT) data. Inspired by the SelectIT framework from [Liu et al., 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/b130a5691815f550977e331f8bec08ae-Paper-Conference.pdf), this scorer focuses on combining predictions from **multiple language models** to produce robust quality assessments.

By employing multiple models with different capacities or architectures, this approach captures diverse perspectives on data quality. The key insight is that **disagreement between models** can reveal ambiguous or low-quality samples, while **consensus across models** indicates high-quality, well-formed instruction-response pairs. This model-level ensemble strategy is particularly effective at filtering out edge cases that might fool a single model.

## Metric Definition:

* **Definition:**
  
  This scorer employs a **multi-model ensemble approach** to evaluate instruction-tuning data quality through three hierarchical levels:
  
  1. **Token-level Scoring**: Each model computes a probability distribution over rating tokens (1-5) to generate an expected score for each prompt.
  2. **Sentence-level Aggregation**: For each model, k different rating prompts are applied to the same sample, and scores are aggregated with a standard deviation penalty to ensure prompt-level consistency.
  3. **Model-level Ensemble** (Core Innovation): Multiple language models independently evaluate the same data, and their scores are combined using weighted averaging to leverage model diversity and reduce individual model biases.

* **Explanation:** 
  
  The final SelectIT Model score reflects the **consensus across multiple models**, weighted by their respective importance. This ensemble strategy provides several benefits:
  
  * A **higher score** (closer to 5) indicates **strong agreement among models** that the sample is high-quality, well-formed, and suitable for instruction tuning.
  * A **lower score** (closer to 1) suggests **model disagreement or consistently low ratings**, indicating the sample may be ambiguous, poorly written, or problematic.
  * The multi-model design ensures that scores are **robust to individual model biases** and capture diverse quality perspectives.

* **Key Advantages:**
  
  * **Model-level ensemble**: Primary strength lies in combining multiple models (e.g., LLaMA-2-7B + LLaMA-2-13B) to capture complementary quality signals and reduce evaluation variance
  * **Weighted aggregation**: Flexible weighting scheme allows prioritizing certain models based on their reliability or domain expertise
  * **Bias mitigation**: Model diversity helps filter out false positives/negatives that might occur with single-model evaluation
  * **Probabilistic foundation**: Token-level probability distributions provide fine-grained quality estimates rather than binary judgments

## YAML Configuration

```yaml
name: SelectitModelScorer
models:
  - meta-llama/Llama-2-7b-hf
  - meta-llama/Llama-2-13b-hf
model_weights: [0.5, 0.5]
rp_file: scorers/SelectIT_rating_prompt.txt
k: 5
alpha: 0.2
max_length: 512
batch_size: 16
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"SelectitModelScorer"` | Identifier for the scorer |
| `models` | list | - | List of model paths for ensemble scoring. Multiple models enable model-level uncertainty assessment |
| `model_weights` | list | Equal weights | Weights for each model in the ensemble. Must match the length of `models` |
| `rp_file` | string | `"scorers/SelectIT_rating_prompt.txt"` | Path to the rating prompt template file containing k different prompts for sentence-level reflection |
| `k` | integer | `5` | Number of different rating prompt templates to use per sample for sentence-level uncertainty assessment |
| `alpha` | float | `0.2` | Standard deviation penalty coefficient. Higher values penalize inconsistent ratings more strongly |
| `max_length` | integer | `512` | Maximum token length for input sequences. Should be between 1 and 2048 |
| `batch_size` | integer | `16` | Number of samples to process in each batch |

## Underlying Model

The SelectIT Scorer is model-agnostic and can work with **any autoregressive language model** that supports causal language modeling. The original paper uses foundation models such as **LLaMA-2** series (7B, 13B, etc.) and **BLOOM** series.

Users can specify any compatible models from Hugging Face Hub or local paths. The scorer automatically handles tokenization and uses the model's native tokenizer. For ensemble evaluation, it is recommended to use 2-3 models of different sizes or architectures to capture diverse perspectives on data quality.

## Scoring Process

The SelectIT scoring process follows a three-level uncertainty framework:

### 1. Token-level Self-Reflection

For each instruction-response pair and each rating prompt:
- The model generates logits for the next token after the prompt
- Probabilities for rating tokens ("1", "2", "3", "4", "5") are extracted
- These probabilities are normalized to form a rating distribution
- Expected score is calculated: `score = Σ(rating × probability)`

### 2. Sentence-level Self-Reflection

For each sample:
- **k** different rating prompts are applied (typically k=5)
- Each prompt produces a token-level score
- Mean (μ) and standard deviation (σ) of k scores are computed
- Sentence-level score is adjusted: `sentence_score = μ / (1 + alpha × σ)`
- This penalizes samples with inconsistent ratings across different prompts

### 3. Model-level Self-Reflection

When multiple models are used:
- Each model independently produces sentence-level scores
- Scores are combined using weighted averaging: `final_score = Σ(model_weight_i × score_i)`
- This leverages uncertainty between different models for robust evaluation

### Complete Pipeline

```
Input: instruction-response pair
  ↓
Generate k rating prompts
  ↓
For each model:
  ↓
  For each prompt:
    → Compute token probabilities
    → Calculate expected score
  ↓
  Aggregate with std penalty (sentence-level)
  ↓
Weighted average across models (model-level)
  ↓
Output: Final SelectIT score (1-5)
```

## Output Format

For each input sample, the scorer returns:

```json
{
  "id": "sample_001",
  "score": 4.23
}
```

- `id`: The unique identifier of the data sample (from the original dataset)
- `score`: The final SelectIT score, ranging from 1.0 to 5.0
  - **1.0-2.0**: Low quality data, likely inconsistent or problematic
  - **2.0-3.0**: Below average quality
  - **3.0-4.0**: Good quality data
  - **4.0-5.0**: High quality, consistent instruction-tuning data

Scores are continuous values, not discrete integers, reflecting the probabilistic nature of the scoring process.

## Citation

```bibtex
@article{liu2024selectit,
  title={SelectIT: Selective instruction tuning for LLMs via uncertainty-aware self-reflection},
  author={Liu, Liangxin and Liu, Xuebo and Wong, Derek F and Li, Dongfang and Wang, Ziyi and Hu, Baotian and Zhang, Min},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={97800--97825},
  year={2024}
}
```

