# DeitaQScorer

## Overview

The **Deita Quality Scorer** is a model-based evaluation tool designed to estimate the data quality of instruction-tuning (SFT) data. This scorer was proposed in the paper [Liu et al., 2024](https://arxiv.org/abs/2312.15685) as part of the DEITA (Data-Efficient Instruction Tuning for Alignment) framework.

The scorer is trained to predict quality scores for instruction-answer pairs by learning from data variants with different quality levels. It provides an automated way to assess the overall quality of supervised fine-tuning samples, helping practitioners select high-quality data for efficient model alignment.

## Metric Definition:

* **Definition:**
  
  1. First generate different quality variants of the same data using the In-Depth Evolving Prompt method.
  2. Collect these data-score pairs to train a LLM as a quality scorer.
  3. The trained scorer is used to predict quality scores (1-6) for other SFT data samples.

* **Explanation:** Intuitively, the Deita Quality score estimates the overall quality of an SFT sample.
  
  * A **higher Deita Quality score** implies that the response presents data in a clear, accurate, and meaningful way.
  * A **lower Deita Quality score** suggests that the response is vague, misleading, or poorly organized in terms of data content.

## YAML Configuration

```yaml
name: DeitaQScorer
model: hkust-nlp/deita-quality-scorer
max_length: 2048
batch_size: 32
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"DeitaQScorer"` | Identifier for the scorer |
| `model` | string | `"hkust-nlp/deita-quality-scorer"` | HuggingFace model path for the quality scoring model |
| `max_length` | integer | `2048` | Maximum sequence length for tokenization |
| `batch_size` | integer | `32` | Number of samples to process in parallel per forward pass |


## Underlying Model

The scorer uses [hkust-nlp/deita-quality-scorer](https://huggingface.co/hkust-nlp/deita-quality-scorer), which is introduced in [Liu et al., 2024](https://arxiv.org/abs/2312.15685).

## Scoring Process

## Scoring Process

1. **Input Processing**: For each data sample, the scorer extracts the instruction from `instruction` field (combined with `input` field if present) and `output` field

2. **Tokenization**: The combined text is tokenized according to the model's tokenizer specifications

3. **Forward Pass**: The combined is fed into the Deita quality scorer model

4. **Score Prediction**: The model predicts a quality score ranging from 1 (lowest quality) to 6 (highest quality)

## Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 4.523
}
```

- `id`: Unique identifier of the data sample (inherited from input data)
- `score`: Predicted quality score as a float value, typically in the range [1.0, 6.0]

## Citation

```bibtex
@article{liu2023makes,
  title={What makes good data for alignment? a comprehensive study of automatic data selection in instruction tuning},
  author={Liu, Wei and Zeng, Weihao and He, Keqing and Jiang, Yong and He, Junxian},
  journal={arXiv preprint arXiv:2312.15685},
  year={2023}
}
```
