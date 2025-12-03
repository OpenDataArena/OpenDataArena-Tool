# SelectitTokenScorer

## Overview

The **SelectIT Token Scorer** is a model-based evaluation tool that implements the token-level and sentence-level uncertainty assessment from the SelectIT framework proposed in [Liu et al., 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/b130a5691815f550977e331f8bec08ae-Paper-Conference.pdf). This scorer provides a lightweight yet effective approach to assess instruction-tuning data quality using a single foundation model.

**By default (k=1)**, this scorer **focuses on token-level uncertainty** by computing the probability distribution over rating tokens (1-5) for a single rating prompt. This provides an efficient quality estimate based on the model's confidence in its ratings.

**When k>1**, the scorer additionally introduces **prompt-level uncertainty** assessment by evaluating the same sample with multiple different rating prompts and measuring consistency across prompts (sentence-level self-reflection). This provides more robust quality estimates by penalizing samples where the model provides inconsistent ratings across different prompt formulations, but increases computational cost.


## Metric Definition:

* **Definition:**
  
  **When k=1 (default - token-level uncertainty only):**
  1. **Token-level Self-Reflection**: The foundation model rates the instruction-response pair from 1 to 5 by computing the probability distribution over rating tokens: `score = Σ(rating_i × P(rating_i))`
  2. **Final Score**: `final_score = score` (no consistency penalty applied)
  
  **When k>1 (token-level + prompt-level uncertainty):**
  1. **Token-level Self-Reflection**: For each of the k rating prompts, compute the token-level score: `score_j = Σ(rating_i × P(rating_i))`
  2. **Sentence-level Self-Reflection**: Measure uncertainty across different prompts via standard deviation to penalize inconsistent ratings
  3. **Final Score Calculation**: `final_score = μ / (1 + alpha × σ)`
  
  Where:
  - `μ` = mean of k token-level scores
  - `σ` = standard deviation of k token-level scores
  - `alpha` = penalty coefficient for inconsistency (default: 0.2)

* **Explanation:** 
  
  **With k=1**: The score purely reflects token-level uncertainty - how confident the model is in its rating based on the probability distribution over rating tokens (1-5). Higher scores indicate the model assigns high probability to high ratings.
  
  **With k>1**: The score additionally considers prompt-level consistency. 
  * A **higher score** (closer to 5) indicates **high-quality data** where the model consistently assigns high ratings across all prompts.
  * A **lower score** (closer to 1) suggests **low-quality data** with either low ratings or high inconsistency across prompts.
  * The standard deviation penalty (controlled by `alpha`) ensures that samples with high rating variance receive lower scores, promoting data consistency.

## YAML Configuration

```yaml
name: SelectitTokenScorer
model: meta-llama/Llama-3.1-8B
rp_file: scorers/SelectIT/rating_prompt.txt
k: 1
alpha: 0.2
max_length: 2048
batch_size: 8
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"SelectitTokenScorer"` | Identifier for the scorer |
| `model` | string | `"meta-llama/Llama-3.1-8B"` | Path to the foundation model (local or Hugging Face Hub). Can be any autoregressive causal language model |
| `rp_file` | string | `"scorers/SelectIT/rating_prompt.txt"` | Path to the rating prompt template file containing k different prompts for evaluation. Each line should contain one rating prompt template |
| `k` | integer | `1` | Number of different rating prompt templates to use per sample. **Default k=1 focuses on token-level uncertainty only**. Set k>1 to introduce prompt-level uncertainty assessment. Higher values provide more robust consistency estimates but increase computation |
| `alpha` | float | `0.2` | Standard deviation penalty coefficient. Controls how strongly inconsistency across prompts penalizes the final score when k>1. Higher values apply stronger penalties. **This parameter has no effect when k=1** |
| `max_length` | integer | `2048` | Maximum token length for input sequences. Longer sequences will be truncated |
| `batch_size` | integer | `8` | Number of samples to process in each batch. Larger batches improve throughput but require more GPU memory |

## Underlying Model

The SelectIT Token Scorer is **model-agnostic** and can work with any autoregressive causal language model. The implementation provides flexibility in model selection:

- **Default fallback**: `meta-llama/Llama-3.1-8B` - A capable foundation model optimized for instruction understanding
- **Recommended models**: 
  - LLaMA-3/3.1 series (8B, 70B)
  - LLaMA-2 series (7B, 13B, 70B)
  - BLOOM series
  - Mistral/Mixtral models
  - Any instruction-tuned or base language models supporting causal LM

The scorer automatically handles different tokenizers and dynamically identifies the token IDs corresponding to ratings "1" through "5" for each model. Users can specify any compatible model from Hugging Face Hub or provide a local model path. The scorer will automatically fall back to the default model if the specified model fails to load.

## Scoring Process

The SelectIT Token scoring process operates differently depending on the k parameter:

**When k=1 (default)**: Focuses solely on **token-level uncertainty** by computing probability distribution over rating tokens for a single prompt.

**When k>1**: Integrates both **token-level** and **sentence-level (prompt-level) uncertainty** assessment.

### 1. Token-level Self-Reflection

For each instruction-response pair and each of the k rating prompts (when k=1, only one prompt is used):

1. **Prompt Construction**: Combine rating prompt template with instruction and response
   ```
   [Rating Prompt Template]
   Instruction: [instruction text]
   Response: [response text]
   The answer is:
   ```

2. **Probability Extraction**: 
   - Tokenize and feed the prompt to the model
   - Extract logits for the next token position after the prompt
   - Compute softmax probabilities for rating tokens ("1", "2", "3", "4", "5")
   - Normalize to ensure probabilities sum to 1.0

3. **Expected Score Calculation**:
   ```
   token_score = Σ(rating × P(rating))
   = 1×P("1") + 2×P("2") + 3×P("3") + 4×P("4") + 5×P("5")
   ```

### 2. Sentence-level Consistency Assessment (Only when k>1)

**When k=1**: This step is skipped. The final score equals the token-level score from the single prompt.

**When k>1**: After obtaining k token-level scores for each sample:

1. **Compute Statistics**:
   - Mean (μ): `μ = (1/k) × Σ token_score_j`
   - Standard deviation (σ): `σ = sqrt((1/k) × Σ(token_score_j - μ)²)`

2. **Apply Consistency Penalty**:
   ```
   final_score = μ / (1 + alpha × σ)
   ```
   - Consistent ratings (low σ) → minimal penalty → score ≈ μ
   - Inconsistent ratings (high σ) → substantial penalty → score < μ

### Complete Pipeline

**When k=1 (token-level uncertainty only):**
```
Input: instruction-response pair
  ↓
Validate required fields (instruction, output)
  ↓
Generate single rating prompt
  ↓
Tokenize prompt
  ↓
Forward pass through model
  ↓
Extract token probabilities for ratings 1-5
  ↓
Normalize probability distribution
  ↓
Calculate expected score: final_score = Σ(rating × P(rating))
  ↓
Output: SelectIT Token score (1-5)
```

**When k>1 (token-level + prompt-level uncertainty):**
```
Input: instruction-response pair
  ↓
Validate required fields (instruction, output)
  ↓
Generate k different rating prompts
  ↓
Batch tokenize all prompts
  ↓
Single forward pass through model
  ↓
For each prompt:
  → Extract token probabilities for ratings 1-5
  → Normalize probability distribution
  → Calculate expected score
  ↓
Collect k token-level scores
  ↓
Compute mean (μ) and std (σ)
  ↓
Apply penalty: final_score = μ / (1 + alpha × σ)
  ↓
Output: SelectIT Token score (1-5)
```

### Batch Processing

The scorer implements efficient batch processing:
- All k×n prompts (for n samples in a batch) are tokenized together using `padding="longest"`
- Single forward pass through the model for all prompts
- Truncation warnings are issued if prompts exceed `max_length`
- Results are grouped back by sample for final score computation
- Invalid samples (missing required fields) receive a default score of 3.0

## Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 4.28
}
```

- `id`: The unique identifier of the data sample (extracted from the original dataset's `"id"` field)
- `score`: The final SelectIT Token score, a continuous value ranging from 1.0 to 5.0
  - **1.0-2.0**: Low quality - low ratings or highly inconsistent across prompts
  - **2.0-3.0**: Below average quality - moderate ratings with some inconsistency
  - **3.0**: Default score assigned to invalid or unparseable data
  - **3.0-4.0**: Good quality - decent ratings with acceptable consistency
  - **4.0-5.0**: High quality - high ratings with strong consistency across all prompts

**Score Interpretation:**
- **When k=1**: Score magnitude purely reflects token-level uncertainty - the expected rating based on probability distribution over rating tokens
- **When k>1**: 
  - **Score magnitude** is determined by the average rating from the model across k prompts
  - **Consistency penalty**: High variance across prompts reduces the score, even if the average is high
  - **Robustness**: Using multiple prompts provides more stable estimates than single-prompt evaluation
- **Default handling**: Samples with missing required fields (`instruction` or `output`) receive a neutral score of 3.0

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

