# ReasoningScorer

## Overview

The **Reasoning Scorer** is a model-based evaluation tool designed to assess the **complexity of logical thinking and analytical reasoning** in SFT data. Introduced as part of the Meta-rater framework in [Zhuang et al., 2025](https://arxiv.org/abs/2504.14194), this scorer evaluates the depth and sophistication of reasoning processes demonstrated in text content, measuring how complex the logical thinking, inference, and analytical capabilities required are. The scorer leverages a fine-tuned ModernBERT-base model trained on 747K examples to provide reliable reasoning complexity assessments on a continuous 0-5 scale.

## Metric Definition:

* **Definition:** 

  A continuous score ranging from 0 to 5 that quantifies the complexity of logical thinking and analytical reasoning required in the text, calculated using a classification model with 6 labels (0-5) and weighted probability averaging.

* **Explanation:** The Reasoning metric evaluates the sophistication of cognitive processes based on:
  1. **Logical Complexity** - The depth of logical reasoning and inference chains
  2. **Analytical Depth** - The sophistication of analysis and critical thinking
  3. **Deductive/Inductive Reasoning** - The quality of reasoning patterns and conclusions
  4. **Problem-Solving Sophistication** - The complexity of problem decomposition and solution strategies

* **Score Interpretation:**
  * **0.0-0.9**: No discernible reasoning; simple statements or facts with no logical connections
  * **1.0-1.9**: Minimal reasoning with significant logical flaws or very basic cause-effect relationships
  * **2.0-2.9**: Basic reasoning with some inconsistencies; simple logical connections present
  * **3.0-3.9**: Moderate reasoning with occasional lapses; clear logical structure with some complexity
  * **4.0-4.9**: Strong reasoning with minor issues; sophisticated analysis and well-developed arguments
  * **5.0**: Exceptional reasoning with flawless logic; complex multi-step reasoning and deep analytical insights

## YAML Configuration

```yaml
name: ReasoningScorer
model: opendatalab/meta-rater-reasoning-rating
batch_size: 16
max_length: 8192
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"ReasoningScorer"` | Identifier for the scorer |
| `model` | string | `"opendatalab/meta-rater-reasoning-rating"` | HuggingFace model path or local path to the reasoning rating model |
| `batch_size` | integer | `16` | Number of samples to process in parallel (adjust based on GPU memory availability) |
| `max_length` | integer | `8192` | Maximum sequence length for tokenization (texts exceeding this length will be truncated with warnings) |

## Underlying Model

The scorer uses [opendatalab/meta-rater-reasoning-rating](https://huggingface.co/opendatalab/meta-rater-reasoning-rating), a fine-tuned version of ModernBERT-base with the following specifications:

* **Base Model**: ModernBERT-base
* **Parameters**: 149M
* **Context Window**: 4,096 tokens (extended to 8,192 in default configuration)
* **Training Data**: 747,422 examples from SlimPajama dataset
* **Annotation Model**: Llama-3.3-70B-Instruct
* **Performance**: 91.57% F1 score, 93.78% accuracy
* **Task Type**: Text classification (6-way classification, labels 0-5)

## Scoring Process

The Reasoning Scorer follows a systematic evaluation pipeline:

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
   * Truncation at `max_length` (default 8,192 tokens)
   * Automatic padding for batch inference
   * Explicit truncation warnings for texts exceeding max length

3. **Model Inference**: The tokenized input is passed through the classification model to obtain logits for 6 classes (0-5).

4. **Score Calculation**: Instead of using the argmax prediction, the scorer computes a **continuous score** using weighted probability averaging:
   ```
   score = Σ(i * P(class_i)) for i in [0, 1, 2, 3, 4, 5]
   ```
   where `P(class_i)` is the softmax probability of class `i`.

5. **Batch Processing**: Samples are processed in batches according to `batch_size` for efficiency, with automatic CUDA cache clearing after each batch to optimize memory usage.

## Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 3.85
}
```

- `id`: The unique identifier from the input dataset (preserved from input)
- `score`: A floating-point value between 0.0 and 5.0 representing the reasoning complexity score (higher values indicate more sophisticated logical thinking and analytical depth)

## Citation

```bibtex
@article{zhuang2025meta,
  title={Meta-rater: A Multi-dimensional Data Selection Method for Pre-training Language Models},
  author={Zhuang, Xinlin and Peng, Jiahui and Ma, Ren and Wang, Yinfan and Bai, Tianyi and Wei, Xingjian and Qiu, Jiantao and Zhang, Chi and Qian, Ying and He, Conghui},
  journal={arXiv preprint arXiv:2504.14194},
  year={2025}
}
```
