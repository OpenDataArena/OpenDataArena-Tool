# UPDScorer

## Overview

The **UPD Scorer** (Unpredictable Diversity Scorer) is a model-based evaluation tool designed to quantify the diversity and difficulty of supervised fine-tuning (SFT) data by measuring the **unpredictability** of model responses. Proposed in the paper [Zhang et al., 2025](https://arxiv.org/abs/2503.11441) as part of the D3 (Diversity, Difficulty, and Dependability) framework, this metric evaluates how unexpected each token in the output is, given the instruction context and preceding tokens.

The UPD metric combines two key aspects of token generation to identify samples that challenge the model with unpredictable yet coherent responses: (1) **Predictability** measured by cross-entropy loss, and (2) **Distribution Concentration** measured by Shannon entropy.

## Metric Definition:

* **Definition:** 

  Given an instruction and output pair, the UPD scorer computes:
  
  1. **Token-Level UPD Score:** For each token position *t* in the output:
     ```
     UPD_t = σ(L_t) × max(0, 1 - H_t / log(V))
     ```
     Where:
     - `L_t = -log(P(y_t | x, y_<t))` is the **cross-entropy loss** for predicting token *y_t*
     - `H_t = -Σ P(y) × log(P(y))` is the **Shannon entropy** of the probability distribution
     - `V` is the **vocabulary size**
     - `σ(·)` is the **sigmoid function**
  
  2. **Sample-Level UPD Score:** The final score is the **average UPD** across all output tokens:
     ```
     UPD = (1/N) × Σ UPD_t
     ```
     Where *N* is the number of tokens in the output.

* **Explanation:** This metric measures the **unpredictability and diversity** of model outputs:
  
  * A **higher UPD score** (0.5 - 1.0) indicates that the output is **unpredictable and diverse**, with tokens that are hard to predict but generated from relatively focused distributions. These samples exhibit high diversity and difficulty.
  * A **medium UPD score** (0.2 - 0.5) suggests moderate unpredictability, representing typical instruction-following behavior.
  * A **lower UPD score** (0.0 - 0.2) indicates the output is **predictable** or has high-entropy (near-uniform) distributions, suggesting either memorized patterns or completely uncertain predictions.

* **Key Advantages:**
  
  * **Combines predictability and concentration:** Jointly considers cross-entropy loss and Shannon entropy for robust diversity assessment
  * **Filters random outputs:** The entropy normalization term `max(0, 1 - H_t / log(V))` filters out tokens with extremely high entropy (near-uniform distributions)
  * **Identifies valuable training samples:** Focuses on cases where the model has a relatively concentrated distribution but still struggles to predict the correct token

## YAML Configuration

```yaml
name: UPDScorer
model: meta-llama/Llama-2-7b-hf
max_length: 2048
batch_size: 8
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"UPDScorer"` | Identifier for the scorer |
| `model` | string | `"Qwen/Qwen3-8B"` | HuggingFace model path for the causal language model used to compute probabilities. Can be any autoregressive LM |
| `max_length` | integer | `2048` | Maximum sequence length for tokenization |
| `batch_size` | integer | `8` | Number of samples to process in parallel per forward pass |


## Underlying Model

The scorer uses causal language models from the HuggingFace ecosystem to compute token-level probabilities and probability distributions. By default, it uses **Qwen/Qwen3-8B**, but can be configured to use any autoregressive language model. The model computes the probability distribution over the vocabulary at each token position, enabling the calculation of cross-entropy loss and Shannon entropy for UPD scoring.

## Scoring Process

1. **Data Preparation**: For each sample, construct the full text as `instruction + input (if exists) + output`, tokenize the instruction part separately to determine where the output begins, and tokenize the full text with padding and truncation to `max_length`

2. **Batch Forward Pass**: Process multiple samples in parallel and run the causal language model to obtain logits for each token position

3. **Token-Level UPD Computation**: For each token *t* in the output portion, compute:
   - Cross-entropy loss: `L_t = -log(P(y_t | x, y_<t))`
   - Shannon entropy: `H_t = -Σ P(y) × log(P(y))`
   - Token UPD score: `UPD_t = σ(L_t) × max(0, 1 - H_t / log(V))`

4. **Sample-Level Aggregation**: Collect UPD scores for all tokens in the output and compute the average UPD as the final sample score

5. **Edge Case Handling**: Samples exceeding `max_length` are truncated with a warning; samples with no valid output tokens receive a score of `0.0`; padding tokens are automatically skipped during computation

## Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 0.4523
}
```

- `id`: Unique identifier from the input data (empty string if not provided)
- `score`: Average UPD score across all output tokens. Range: [0, 1]. Higher values indicate more unpredictable and diverse outputs.

## Citation

```bibtex
@article{zhang2025d3,
  title={D3: Diversity, Difficulty, and Dependability-Aware Data Selection for Sample-Efficient LLM Instruction Tuning},
  author={Zhang, Jia and Zhang, Chen-Xi and Liu, Yao and Jin, Yi-Xuan and Yang, Xiao-Wen and Zheng, Bo and Liu, Yi and Guo, Lan-Zhe},
  journal={arXiv preprint arXiv:2503.11441},
  year={2025}
}
```
