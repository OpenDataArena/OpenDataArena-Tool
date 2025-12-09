# AnswerProbScorer

## Overview

The **Answer Probability Scorer** is a model-based evaluation tool designed to assess the quality and difficulty of instruction-answer pairs by computing the conditional probability of answers given instructions. This scorer leverages causal language models to measure how well an answer aligns with its corresponding instruction through probabilistic analysis.

**Important Design Choice:** This scorer **focuses exclusively on the final answer tokens** (e.g., extracted from `\boxed{...}` notation or the `answer` field), rather than evaluating the entire output including reasoning steps and explanations. This design choice ensures:
- **Precision**: Direct measurement of how well the instruction guides to the correct answer
- **Fairness**: Avoids bias from varying output lengths and writing styles
- **Clarity**: Scores directly reflect instruction-answer alignment, not intermediate reasoning quality

Unlike simple perplexity-based metrics, the Answer Probability Scorer implements a **normalized scoring mechanism** that accounts for the intrinsic probability of the answer itself. By comparing the conditional probability P(Answer|Instruction) against the baseline probability P(Answer), this method provides a more robust measure of instruction-answer alignment that is less biased by answer length or common phrase frequencies.

## Metric Definition:

* **Definition:** 

  Given an instruction Q and answer A (note: **only the final answer is evaluated, not the entire output**), the scorer computes:
  
  1. **Conditional Probability Score (P_A):** The average log probability of **answer tokens only** given the full instruction-answer context
  2. **Baseline Probability Score (P_B):** The average log probability of **answer tokens only** without any instruction context
  3. **Normalized Score:** `score = log(P_A) - log(P_B) = log(P_A / P_B)`

* **Why Focus Only on Answer Tokens?**
  
  This scorer deliberately excludes intermediate reasoning steps and explanations from the output, focusing solely on the final answer because:
  - The goal is to measure **how effectively the instruction guides to the correct answer**
  - Including reasoning steps would introduce noise from varying writing styles and output lengths
  - Answer-focused evaluation provides fairer comparison across samples with different output structures
  - This aligns with practical scenarios where the correctness of the final answer is the primary concern

* **Explanation:** This metric measures the **relative probability gain** when the instruction is provided:
  
  * A **higher normalized score** (positive value) indicates that the instruction **significantly increases** the likelihood of the answer, suggesting strong instruction-answer alignment and higher quality.
  * A **lower normalized score** (negative value) indicates that the instruction provides **little guidance** or even **contradicts** the natural answer generation, suggesting poor alignment or lower quality.
  * A **score close to zero** suggests that the instruction provides **marginal information** beyond what the model already knows.

* **Key Advantages:**
  
  * **Length-invariant:** By using average log probabilities, the metric is not biased by answer length
  * **Baseline normalization:** Subtracting the unconditional answer probability removes bias toward common phrases
  * **Log-space computation:** Prevents numerical underflow and provides interpretable probability ratios

## YAML Configuration

```yaml
name: AnswerProbScorer
model: Qwen/Qwen2.5-7B
case_sensitive: true
batch_size: 16
max_length: 2048
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"AnswerProbScorer"` | Identifier for the scorer |
| `model` | string | `"Qwen/Qwen3-8B"` | HuggingFace model path for the causal language model used to compute probabilities |
| `case_sensitive` | boolean | `true` | Whether to perform case-sensitive answer extraction and matching |
| `batch_size` | integer | `1` | Number of samples to process in parallel per forward pass |
| `max_length` | integer | `2048` | Maximum sequence length for tokenization |


## Underlying Model

The scorer uses causal language models from the HuggingFace ecosystem to compute token-level probabilities. By default, it uses **Qwen/Qwen3-8B**, but can be configured to use any autoregressive language model.

## Scoring Process

1. **Input Processing**: For each data sample, the scorer extracts:
   - Instruction (from `instruction` and optional `input` fields)
   - Answer (from `answer` field if present, otherwise extracted from `output` using `\boxed{...}` notation)

2. **Tokenization**: The concatenated instruction-answer text is tokenized with offset mapping to track character-to-token alignment

3. **Forward Pass A (Conditional)**: Compute log probabilities for all tokens in the instruction+answer sequence

4. **Answer Token Identification**: Use offset mapping to identify which tokens correspond to the answer segment

5. **Forward Pass B (Baseline)**: Compute log probabilities for the answer-only sequence without instruction

6. **Score Computation**: Calculate normalized score as `log(P_A / P_B)` where:
   - P_A = average log probability of answer tokens in full context
   - P_B = average log probability of answer tokens without instruction

## Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "mean_prob": -2.3456,
  "token_count": 15,
  "answers": ["42"],
  "answer_str": "42",
  "mean_prob_answer_only": -3.1234,
  "score": 0.7778,
  "answer_only_token_count": 15
}
```

- `mean_prob`: Average log probability of answer tokens given instruction (P_A)
- `token_count`: Number of answer tokens used in conditional probability calculation
- `answers`: Extracted answer(s) from the output
- `answer_str`: Comma-separated string of all answers
- `mean_prob_answer_only`: Average log probability of answer tokens without instruction (P_B)
- `score`: Normalized score = log(P_A / P_B)
- `answer_only_token_count`: Number of tokens in answer-only sequence

## Citation

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

