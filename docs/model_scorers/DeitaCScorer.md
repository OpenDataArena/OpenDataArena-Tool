# DeitaCScorer

## Overview

The **Deita Complexity Scorer** is a model-based evaluation tool designed to estimate the *instruction complexity* of SFT data. Proposed in the paper [Liu et al., 2024](https://arxiv.org/abs/2312.15685), this method aims to measure how cognitively demanding an instruction is for a model to execute. Rather than relying on shallow heuristics, the Deita Complexity Scorer provides a learning-based, instruction-only metric that correlates with downstream performance and instruction-following capabilities.

## Metric Definition:

* **Definition:**
  
    1. First generate variations of each instruction with increasing difficulty using the In-Depth Evolving Prompt method.
    2. Collect these data-score pairs to train a LLM as a complexity scorer.
    3. The trained scorer is used to predict complexity scores (1-6) for new instructions.

* **Explanation:** Intuitively, the complexity score estimates how *unexpected or difficult* an instruction is to follow for the SFT model.

  * A **higher Deita Complexity score** imply that the SFT model struggles with the instruction relative to the reference model, indicating **greater complexity**.
  * A **lower Deita Complexity score** suggest that the instruction is easy to complete and consistent with the SFT model's learned behaviors.

## YAML Configuration
```yaml
name: DeitaCScorer
model: hkust-nlp/deita-complexity-scorer
max_length: 2048
batch_size: 32
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"DeitaCScorer"` | Identifier for the scorer |
| `model` | string | `"hkust-nlp/deita-complexity-scorer"` | HuggingFace model path for the Deita complexity scorer |
| `max_length` | integer | `2048` | Maximum sequence length for tokenization |
| `batch_size` | integer | `32` | Number of samples to process in parallel per forward pass |


## Underlying Model

The scorer uses [hkust-nlp/deita-complexity-scorer](https://huggingface.co/hkust-nlp/deita-complexity-scorer), which is introduced in [Liu et al., 2024](https://arxiv.org/abs/2312.15685).

## Scoring Process

1. **Input Processing**: For each data sample, the scorer extracts the instruction from `instruction` field (combined with `input` field if present)

2. **Tokenization**: The instruction text is tokenized according to the model's tokenizer specifications

3. **Forward Pass**: The instruction is fed into the Deita complexity scorer model

4. **Score Prediction**: The model predicts a complexity score ranging from 1 (simplest) to 6 (most complex)

## Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 4.5,
}
```

- `id`: Unique identifier for the input sample
- `score`: Complexity score ranging from 1-6, where higher values indicate more complex instructions

## Citation

```bibtex
@article{liu2023makes,
  title={What makes good data for alignment? a comprehensive study of automatic data selection in instruction tuning},
  author={Liu, Wei and Zeng, Weihao and He, Keqing and Jiang, Yong and He, Junxian},
  journal={arXiv preprint arXiv:2312.15685},
  year={2023}
}
```
