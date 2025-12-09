# ProfessionalismScorer

## Overview

The **Professionalism Scorer** is a model-based evaluation tool designed to assess the **degree of expertise and technical knowledge** required to understand SFT data. Introduced as part of the Meta-rater framework in [Zhuang et al., 2025](https://arxiv.org/abs/2504.14194), this scorer evaluates the professional depth and technical complexity of text content, measuring how much specialized knowledge is needed to comprehend the material.

The scorer leverages a fine-tuned ModernBERT-base model trained on 747K examples to provide reliable professionalism assessments on a continuous 0-5 scale, enabling fine-grained assessment of the technical complexity and expertise requirements of instruction-following datasets.

## Metric Definition:

* **Definition:** 
  
  A continuous score ranging from 0 to 5 that quantifies the degree of expertise and technical knowledge required to understand the text, calculated using a classification model with 6 labels (0-5) and weighted probability averaging.

* **Explanation:** The Professionalism metric evaluates the technical depth and specialized knowledge requirements of text content based on:
  1. **Technical Complexity** - The level of domain-specific terminology and concepts
  2. **Expertise Requirements** - The background knowledge needed to comprehend the content
  3. **Accessibility** - How specialized or general-audience the content is

* **Score Interpretation:**
  * **0.0-0.9**: No technical knowledge required; content accessible to everyone
  * **1.0-1.9**: Minimal technical knowledge needed; simple content (e.g., children's books, basic tutorials)
  * **2.0-2.9**: Basic professional knowledge required; general-audience content (e.g., popular science articles)
  * **3.0-3.9**: Moderate expertise needed; intermediate complexity (e.g., advanced articles, technical documentation)
  * **4.0-4.9**: Significant professional knowledge required; complex content (e.g., academic papers, technical reports)
  * **5.0**: Advanced expertise essential; highly professional content (e.g., advanced research papers, patents)

## YAML Configuration

```yaml
name: ProfessionalismScorer
model: opendatalab/meta-rater-professionalism-rating
batch_size: 16
max_length: 8192
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"ProfessionalismScorer"` | Identifier for the scorer |
| `model` | string | `"opendatalab/meta-rater-professionalism-rating"` | HuggingFace model path or local path to the professionalism rating model |
| `batch_size` | integer | `16` | Number of samples to process in parallel (adjust based on GPU memory availability) |
| `max_length` | integer | `8192` | Maximum sequence length for tokenization (texts exceeding this length will be truncated with warnings) |

## Underlying Model

The scorer uses [opendatalab/meta-rater-professionalism-rating](https://huggingface.co/opendatalab/meta-rater-professionalism-rating), a fine-tuned version of ModernBERT-base with the following specifications:

* **Base Model**: ModernBERT-base
* **Parameters**: 149M
* **Context Window**: 4,096 tokens (extended to 8,192 in default configuration)
* **Training Data**: 747,422 examples from SlimPajama dataset
* **Annotation Model**: Llama-3.3-70B-Instruct
* **Performance**: 91.57% F1 score, 93.78% accuracy
* **Task Type**: Text classification (6-way classification, labels 0-5)

## Scoring Process

1. **Text Concatenation**: For each data sample, the scorer concatenates the fields in the following order:
   - `content = instruction + "\n" + input + "\n" + output`
   - If the `input` field is empty, it uses: `content = instruction + "\n" + output`

2. **Tokenization**: The concatenated text is tokenized using the ModernBERT tokenizer with left-side padding for batch processing, truncation at `max_length` (default 8,192 tokens), and explicit truncation warnings for texts exceeding max length

3. **Model Inference**: The tokenized input is passed through the classification model to obtain logits for 6 classes (0-5)

4. **Score Calculation**: Instead of using the argmax prediction, the scorer computes a continuous score using weighted probability averaging: `score = Σ(i * P(class_i)) for i in [0, 1, 2, 3, 4, 5]` where `P(class_i)` is the softmax probability of class `i`

5. **Batch Processing**: Samples are processed in batches according to `batch_size` for efficiency, with automatic CUDA cache clearing after each batch to optimize memory usage

## Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 3.42
}
```

- `id`: The unique identifier from the input dataset (preserved from input)
- `score`: A floating-point value between 0.0 and 5.0 representing the professionalism score. Higher values indicate greater technical depth and expertise requirements

## Citation

```bibtex
@article{zhuang2025meta,
  title={Meta-rater: A Multi-dimensional Data Selection Method for Pre-training Language Models},
  author={Zhuang, Xinlin and Peng, Jiahui and Ma, Ren and Wang, Yinfan and Bai, Tianyi and Wei, Xingjian and Qiu, Jiantao and Zhang, Chi and Qian, Ying and He, Conghui},
  journal={arXiv preprint arXiv:2504.14194},
  year={2025}
}
```
