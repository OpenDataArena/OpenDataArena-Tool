# NormLossScorer

## Overview

The **NormLoss Scorer** is a model-based evaluation tool that assesses SFT data quality through the lens of **compression efficiency**. Inspired by the finding that "compression represents intelligence linearly" ([Huang et al., 2024](https://arxiv.org/abs/2404.09937)), this scorer leverages language models as data compressors to measure the complexity and quality of instruction-response pairs. The underlying principle is that better data exhibits predictable patterns that can be efficiently compressed by language models, while lower-quality or overly complex data results in higher compression costs.

By computing the **normalized cross-entropy loss** (in bits per token), NormLoss Scorer provides an unsupervised, model-based metric that correlates with data quality and model learning efficiency. This approach is particularly useful for identifying high-quality training samples that align well with the model's learned representations.

## Metric Definition:

* **Definition:** 

```
NormLoss = (1 / N) × Σ -log₂ P(token_i | context)
```

where `N` is the number of tokens in the sequence, and `P(token_i | context)` is the model's predicted probability for each token given its context.

* **Explanation:** This metric measures the **average number of bits** required to encode each token in the text using the language model as a compressor. It reflects how well the model can predict and compress the given text:
  
  * A **lower NormLoss score** indicates the text is **more compressible** and aligns better with the model's learned patterns, suggesting **higher quality** or **better fit** with the model's knowledge distribution.
  * A **higher NormLoss score** suggests the text is **less compressible**, containing more surprising or unpredictable content relative to the model's expectations, which may indicate **lower quality** or **higher complexity**.

* **Key Advantages:**
  
  * **Information-theoretic foundation:** Rooted in the principle that compression and prediction are equivalent - better compression implies better understanding
  * **Unsupervised metric:** No need for labeled data or human annotations
  * **Model-aligned:** Measures data quality from the perspective of the specific model's learned distribution

## YAML Configuration

```yaml
name: NormLossScorer
model: meta-llama/Llama-3.1-8B
max_length: 2048
batch_size: 8
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"NormLossScorer"` | Identifier for the scorer |
| `model` | string | `"meta-llama/Llama-3.1-8B"` | HuggingFace model path or local path for the causal language model used to compute cross-entropy loss |
| `max_length` | integer | `2048` | Maximum sequence length for tokenization. Sequences longer than this will be truncated |
| `batch_size` | integer | `8` | Number of samples to process in parallel per forward pass |

## Underlying Model

The scorer uses causal language models from the HuggingFace ecosystem to compute cross-entropy loss. By default, it uses **meta-llama/Llama-3.1-8B**, but can be configured to use any autoregressive language model.

## Scoring Process

1. **Input Processing**: For each data sample, the scorer concatenates:
   - Instruction (from `instruction` field)
   - Optional input text (from `input` field if present)
   - Response (from `output` field)
   
   Format: `text = instruction + '\n' + [input + '\n'] + output`

2. **Tokenization**: The concatenated text is tokenized with padding and truncation at `max_length`

3. **Forward Pass**: Compute token-level log probabilities through the causal language model

4. **Cross-Entropy Computation**: Calculate loss for each valid token as:
   `loss_i = -log P(token_i | token_1, ..., token_{i-1})`

5. **Normalization**: Average the loss over valid tokens and convert to bits per token:
   `NormLoss = (Σ loss_i / N) / ln(2)`

## Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 2.456
}
```

- `id`: Unique identifier for the data sample (from the `id` field in input data, or empty string if not present)
- `score`: The computed NormLoss value representing normalized cross-entropy in bits per token
  - **Lower values** indicate better compression and potentially higher quality
  - **Higher values** indicate poorer compression and potentially lower quality or higher complexity

## Citation

```bibtex
@article{shum2025predictive,
  title={Predictive data selection: The data that predicts is the data that teaches},
  author={Shum, Kashun and Huang, Yuzhen and Zou, Hongjian and Ding, Qi and Liao, Yixuan and Chen, Xiaoxin and Liu, Qian and He, Junxian},
  journal={arXiv preprint arXiv:2503.00808},
  year={2025}
}
```
