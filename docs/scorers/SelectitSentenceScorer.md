# SelectitSentenceScorer

## Overview

The **SelectIT Sentence Scorer** is a model-based evaluation tool that implements the sentence-level uncertainty assessment from the SelectIT framework proposed in [Liu et al., 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/b130a5691815f550977e331f8bec08ae-Paper-Conference.pdf). Unlike the full SelectIT scorer which uses multiple models (model-level uncertainty), this scorer focuses on leveraging **token-level** and **sentence-level** uncertainty within a single foundation model to assess instruction-tuning data quality.

This approach evaluates the same instruction-response pair using multiple different rating prompts and measures the consistency of ratings across prompts. High-quality data samples receive consistent ratings across different prompt formulations, while low-quality or ambiguous samples exhibit high variance. The scorer is lightweight, requiring only a single model while still providing robust quality estimates through prompt-based uncertainty assessment.

## Metric Definition:

* **Definition:** 

  Given an instruction-response pair, the scorer computes:
  
  1. **Token-level Self-Reflection**: For each rating prompt j, compute the expected rating as `score_j = Σ(rating_i × P(rating_i))` where P(rating_i) is the probability the model assigns to rating i ∈ {1,2,3,4,5}
  2. **Sentence-level Self-Reflection**: Apply k different rating prompts to the same sample, compute mean μ and standard deviation σ of the k token-level scores
  3. **Normalized Score**: `final_score = μ / (1 + alpha × σ)` where alpha controls the penalty for inconsistency

* **Explanation:** This metric measures data quality through **rating consistency across multiple prompt formulations**:
  
  * A **higher score** (closer to 5) indicates **high-quality data** with consistent ratings across multiple prompts, suggesting the model confidently evaluates the sample positively.
  * A **lower score** (closer to 1) suggests **low-quality data** with inconsistent ratings, indicating ambiguity or problems in the instruction-response pair.
  * A **score close to 3** represents neutral or average quality, often assigned to samples with high variance.

* **Key Advantages:**
  
  * **Prompt-based uncertainty**: Measures quality through consistency rather than a single rating
  * **Lightweight**: Requires only one model (unlike model-level ensemble methods)
  * **Interpretable**: Scores range from 1-5 with clear quality implications
  * **Robust**: Inconsistency penalty prevents high scores for ambiguous samples

## YAML Configuration

```yaml
name: SelectitSentenceScorer
model: meta-llama/Llama-2-7b-hf
rp_file: scorers/SelectIT_rating_prompt.txt
k: 5
alpha: 0.2
max_length: 512
batch_size: 16
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"SelectitSentenceScorer"` | Identifier for the scorer |
| `model` | string | `"princeton-nlp/QuRater-1.3B"` | HuggingFace model path or local path to the causal language model used for rating |
| `rp_file` | string | `"scorers/SelectIT_rating_prompt.txt"` | Path to the rating prompt template file containing k different prompts for sentence-level reflection |
| `k` | integer | `5` | Number of different rating prompt templates to use per sample for uncertainty assessment |
| `alpha` | float | `0.2` | Standard deviation penalty coefficient; higher values apply stronger penalties for inconsistency |
| `max_length` | integer | `512` | Maximum sequence length for tokenization (must be between 1 and 2048) |
| `batch_size` | integer | `16` | Number of samples to process in parallel per forward pass |

## Underlying Model

The SelectIT Sentence Scorer is **model-agnostic** and can work with any autoregressive causal language model. The implementation provides flexibility in model selection:

- **Default fallback**: `princeton-nlp/QuRater-1.3B` - A specialized model for rating instruction-tuning data
- **Recommended models**: 
  - LLaMA-2 series (7B, 13B)
  - BLOOM series
  - Mistral/Mixtral models
  - Any instruction-tuned or base language models supporting causal LM

The scorer automatically handles different tokenizers and dynamically identifies the token IDs corresponding to ratings "1" through "5" for each model. Users can specify any compatible model from Hugging Face Hub or provide a local model path.

## Scoring Process

1. **Input Processing**: For each data sample, extract instruction and response fields

2. **Prompt Generation**: Create k different rating prompts by combining each rating template with the instruction-response pair:
   ```
   [Rating Prompt Template]
   Instruction: [instruction text]
   Response: [response text]
   The answer is:
   ```

3. **Token-level Probability Extraction**: For each of the k prompts:
   - Feed the prompt to the model
   - Extract logits for the next token position after "The answer is:"
   - Compute softmax probabilities over rating tokens {"1", "2", "3", "4", "5"}
   - Calculate expected score: `score_j = 1×P("1") + 2×P("2") + 3×P("3") + 4×P("4") + 5×P("5")`

4. **Sentence-level Aggregation**: Collect k token-level scores and compute:
   - Mean: `μ = (1/k) × Σ score_j`
   - Standard deviation: `σ = sqrt((1/k) × Σ(score_j - μ)²)`

5. **Score Computation**: Apply consistency penalty to obtain final score:
   ```
   final_score = μ / (1 + alpha × σ)
   ```
   Low σ (consistent ratings) → minimal penalty; High σ (inconsistent ratings) → substantial penalty

6. **Batch Processing**: All k×n prompts (for n samples) are processed together in batches for efficiency

## Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 4.15
}
```

- `id`: Unique identifier of the data sample from the original dataset
- `score`: Final SelectIT Sentence score, a continuous value ranging from 1.0 to 5.0
  - **1.0-2.0**: Low quality - inconsistent ratings or poor content
  - **2.0-3.0**: Below average quality - moderate inconsistency
  - **3.0-4.0**: Good quality - consistent ratings with minor variations
  - **4.0-5.0**: High quality - highly consistent ratings across all prompts

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

