# CleanlinessScorer

## Overview

The **Cleanliness Scorer** is a model-based evaluation tool designed to assess the **format quality** and **noise-free content** of SFT data. Introduced as part of the Meta-rater framework in [Zhuang et al., 2025](https://arxiv.org/abs/2504.14194), this scorer evaluates how well-formatted, complete, and structurally sound a text is, focusing on presentation quality rather than semantic content. The scorer leverages a fine-tuned ModernBERT-base model trained on 747K examples to provide reliable cleanliness assessments on a continuous 0-5 scale.

## Metric Definition:

* **Definition:** 

  A continuous score ranging from 0 to 5 that quantifies the format quality and structural integrity of text, calculated using a classification model with 6 labels (0-5) and weighted probability averaging.

* **Explanation:** 

  The Cleanliness metric evaluates text across three primary dimensions:
  
  1. **Correct Formatting** - Text appears human-edited with proper structure and no corrupted characters
  2. **Appropriate Content** - No irrelevant links, advertisements, or spam; sufficient content length
  3. **Completeness** - Complete sentences with coherent structure and natural flow

* **Score Interpretation:**
  
  * **4.0-5.0**: High-quality content with perfect or near-perfect formatting
  * **3.0-3.9**: Acceptable content with minor issues that don't seriously impact readability
  * **2.0-2.9**: Obvious problems that noticeably affect reading fluency
  * **1.0-1.9**: Serious formatting or structural issues
  * **0.0-0.9**: Absolute noisy content unsuitable for training

## YAML Configuration

```yaml
name: CleanlinessScorer
model: opendatalab/meta-rater-cleanliness-rating
batch_size: 16
max_model_len: 8192
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"CleanlinessScorer"` | Identifier for the scorer |
| `model` | string | `"opendatalab/meta-rater-cleanliness-rating"` | HuggingFace model path or local path to the cleanliness rating model |
| `batch_size` | integer | `16` | Number of samples to process in parallel (adjust based on GPU memory availability) |
| `max_model_len` | integer | `8192` | Maximum sequence length for tokenization (texts exceeding this length will be truncated) |

## Underlying Model

The scorer uses [opendatalab/meta-rater-cleanliness-rating](https://huggingface.co/opendatalab/meta-rater-cleanliness-rating), a fine-tuned version of ModernBERT-base with the following specifications:

* **Base Model**: ModernBERT-base
* **Parameters**: 149M
* **Context Window**: 4,096 tokens (extended to 8,192 in default configuration)
* **Training Data**: 747,422 examples from SlimPajama dataset
* **Annotation Model**: Llama-3.3-70B-Instruct
* **Performance**: 87.88% F1 score, 92.25% accuracy
* **Task Type**: Text classification (6-way classification, labels 0-5)

## Scoring Process

The Cleanliness Scorer follows a systematic evaluation pipeline:

1. **Text Concatenation**: For each data sample, the scorer concatenates the fields in the following order:
   ```
   content = instruction + "\n" + input + "\n" + output
   ```
   If the `input` field is empty, it uses:
   ```
   content = instruction + "\n" + output
   ```

2. **Tokenization**: The concatenated text is tokenized using the ModernBERT tokenizer with:
   * Left-side padding for batch processing
   * Truncation at `max_model_len` (default 8,192 tokens)
   * Automatic padding for batch inference

3. **Model Inference**: The tokenized input is passed through the classification model to obtain logits for 6 classes (0-5).

4. **Score Calculation**: Instead of using the argmax prediction, the scorer computes a **continuous score** using weighted probability averaging:
   ```
   score = Σ(i * P(class_i)) for i in [0, 1, 2, 3, 4, 5]
   ```
   where `P(class_i)` is the softmax probability of class `i`.

5. **Batch Processing**: Samples are processed in batches according to `batch_size` for efficiency, with automatic CUDA cache clearing after each batch.

## Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 3.85
}
```

- `id`: The unique identifier from the input dataset (preserved from input)
- `score`: A floating-point value between 0.0 and 5.0 representing the cleanliness score (higher values indicate better formatting and structural quality)

## Citation

```bibtex
@article{zhuang2025meta,
  title={Meta-rater: A Multi-dimensional Data Selection Method for Pre-training Language Models},
  author={Zhuang, Xinlin and Peng, Jiahui and Ma, Ren and Wang, Yinfan and Bai, Tianyi and Wei, Xingjian and Qiu, Jiantao and Zhang, Chi and Qian, Ying and He, Conghui},
  journal={arXiv preprint arXiv:2504.14194},
  year={2025}
}
```

