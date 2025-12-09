# ReadabilityScorer

## Overview

The **Readability Scorer** is a model-based evaluation tool designed to assess the **ease of understanding and text clarity** of SFT data. Introduced as part of the Meta-rater framework in [Zhuang et al., 2025](https://arxiv.org/abs/2504.14194), this scorer evaluates how easily text can be read and comprehended by assessing factors such as clarity, coherence, vocabulary complexity, sentence structure, and grammatical correctness. The scorer leverages a fine-tuned ModernBERT-base model trained on 747K examples to provide reliable readability assessments on a continuous 0-5 scale.

## Metric Definition:

* **Definition:** 

  A continuous score ranging from 0 to 5 that quantifies the ease of understanding and clarity of text, calculated using a classification model with 6 labels (0-5) and weighted probability averaging.

* **Explanation:** 

  The Readability metric evaluates text comprehensibility based on multiple linguistic dimensions:
  1. **Clarity and Coherence** - Logical flow and organization of ideas
  2. **Vocabulary Complexity** - Appropriate use of words and terminology
  3. **Sentence Structure** - Sentence length, complexity, and construction
  4. **Grammar and Spelling** - Correctness of language use

* **Score Interpretation:**
  
  * **0.0-0.9**: Absolutely not readable; severe clarity issues, incoherent content
  * **1.0-1.9**: Somewhat readable but contains significant clarity or coherence issues, complex vocabulary, or numerous errors
  * **2.0-2.9**: Generally clear and coherent with occasional grammar, spelling errors, or convoluted structures
  * **3.0-3.9**: Clear and coherent for the most part, using appropriate vocabulary with minor grammar/spelling issues
  * **4.0-4.9**: Very clear and coherent with very few or no errors, proper punctuation, and easy-to-follow structures
  * **5.0**: Outstanding clarity and coherence, effective communication with minimal errors that don't interfere with understanding

## YAML Configuration

```yaml
name: ReadabilityScorer
model: opendatalab/meta-rater-readability-rating
batch_size: 16
max_length: 8192
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"ReadabilityScorer"` | Identifier for the scorer |
| `model` | string | `"opendatalab/meta-rater-readability-rating"` | HuggingFace model path or local path to the readability rating model |
| `batch_size` | integer | `16` | Number of samples to process in parallel (adjust based on GPU memory availability) |
| `max_length` | integer | `8192` | Maximum sequence length for tokenization (texts exceeding this length will be truncated with warnings) |

## Underlying Model

The scorer uses [opendatalab/meta-rater-readability-rating](https://huggingface.co/opendatalab/meta-rater-readability-rating), a fine-tuned version of ModernBERT-base with the following specifications:

* **Base Model**: ModernBERT-base
* **Parameters**: 149M
* **Context Window**: 4,096 tokens (extended to 8,192 in default configuration)
* **Training Data**: 747,422 examples from SlimPajama dataset
* **Annotation Model**: Llama-3.3-70B-Instruct
* **Performance**: 87.47% F1 score, 94.13% accuracy
* **Task Type**: Text classification (6-way classification, labels 0-5)

## Scoring Process

1. **Text Concatenation**: For each data sample, the scorer concatenates the fields in the following order:
   - `content = instruction + "\n" + input + "\n" + output`
   - If the `input` field is empty, it uses: `content = instruction + "\n" + output`

2. **Tokenization**: The concatenated text is tokenized using the ModernBERT tokenizer with left-side padding, truncation at `max_length` (default 8,192 tokens), and automatic padding for batch inference

3. **Model Inference**: The tokenized input is passed through the classification model to obtain logits for 6 classes (0-5)

4. **Score Calculation**: The scorer computes a **continuous score** using weighted probability averaging:
   - `score = Σ(i * P(class_i)) for i in [0, 1, 2, 3, 4, 5]`
   - where `P(class_i)` is the softmax probability of class `i`

5. **Batch Processing**: Samples are processed in batches according to `batch_size` for efficiency, with automatic CUDA cache clearing after each batch

## Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 3.67
}
```

- `id`: The unique identifier from the input dataset (preserved from input)
- `score`: A floating-point value between 0.0 and 5.0 representing the readability score (higher values indicate better readability and clearer communication)

## Citation

```bibtex
@article{zhuang2025meta,
  title={Meta-rater: A Multi-dimensional Data Selection Method for Pre-training Language Models},
  author={Zhuang, Xinlin and Peng, Jiahui and Ma, Ren and Wang, Yinfan and Bai, Tianyi and Wei, Xingjian and Qiu, Jiantao and Zhang, Chi and Qian, Ying and He, Conghui},
  journal={arXiv preprint arXiv:2504.14194},
  year={2025}
}
```
