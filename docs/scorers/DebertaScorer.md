# DebertaScorer

## Overview

The **Deberta Quality Classifier** is a model-based evaluation tool designed to assess the overall quality of supervised fine-tuning (SFT) data. Built upon the DeBERTa-v3 architecture, this classifier assigns each instruction-response pair a quality score (0, 1, or 2), representing three distinct quality levels: **Low**, **Medium**, and **High**. The model was trained on 22,828 Common Crawl text samples annotated by human evaluators who assessed quality based on factors such as content accuracy, clarity, coherence, grammar, depth of information, and overall usefulness.

## Metric Definition:

* **Definition:** The model assigns each SFT sample a quality score (0, 1, or 2) based on the concatenated text of instruction, input (if present), and output:
  * **Score = 2 (High Quality):** The content demonstrates excellent accuracy, clarity, coherence, proper grammar, substantial depth of information, and high overall usefulness.
  * **Score = 1 (Medium Quality):** The content shows acceptable quality with reasonable clarity and coherence, but may have minor issues in grammar, depth, or organization.
  * **Score = 0 (Low Quality):** The content exhibits poor quality, with significant issues in accuracy, clarity, coherence, grammar, or lacks meaningful information.

* **Explanation:** The quality score reflects the overall suitability of the data for supervised fine-tuning:
  * A **score of 2** indicates that the sample is well-suited for SFT and likely to improve model performance.
  * A **score of 1** suggests the sample has acceptable quality but may benefit from additional filtering or refinement.
  * A **score of 0** indicates the sample should likely be filtered out as it may negatively impact model training.

## YAML Configuration

```yaml
name: DebertaScorer
model: nvidia/quality-classifier-deberta
max_length: 2048
batch_size: 32
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"DebertaScorer"` | Identifier for the scorer |
| `model` | string | `"nvidia/quality-classifier-deberta"` | Path to the model, either a local directory or a HuggingFace model ID |
| `max_length` | integer | `2048` | Maximum token length for input text. Texts exceeding this length will be truncated |
| `batch_size` | integer | `32` | Number of samples to process in each batch |

## Underlying Model

The scorer uses [nvidia/quality-classifier-deberta](https://huggingface.co/nvidia/quality-classifier-deberta), a text classification model based on the **DeBERTa-v3 Base** architecture with a context length of 1024 tokens. The model consists of:

* A DeBERTa-v3 Base encoder for contextual text representation
* A dropout layer for regularization
* A linear classification head that outputs probabilities for three quality classes

The model was trained on human-annotated Common Crawl data and achieves an accuracy of **0.8252** on the evaluation set where all three annotators agreed on the label.

If the specified model fails to load, the scorer automatically falls back to the default `nvidia/quality-classifier-deberta` model.

## Scoring Process

1. **Text Concatenation**: For each data sample, the scorer concatenates the following fields:
   - `instruction`: The instruction text
   - `input`: Additional input context (if present and non-empty)
   - `output`: The response or completion text
   - The final text format is: `instruction\n[input\n]output`

2. **Batch Tokenization**: Text samples are tokenized in batches using the DeBERTa tokenizer with padding, truncation to `max_length` tokens, and automatic addition of special tokens

3. **Truncation Warning**: If any text exceeds `max_length`, a warning is displayed indicating the number of truncated samples

4. **Model Inference**: The tokenized inputs are passed through the quality classification model:
   - The DeBERTa encoder generates contextual embeddings
   - Dropout is applied for regularization
   - The classification head outputs a probability distribution over the three quality classes
   - The class with the highest probability is selected via argmax and returned as an integer score (0, 1, or 2)

5. **Batch Processing**: All samples in the dataset are processed in batches of size `batch_size`, with a progress bar displaying the current status

## Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 1
}
```

- `id`: The unique identifier of the sample, extracted from the input data's `id` field. If no `id` is present in the input, this will be an empty string
- `score`: An integer quality score where **0 = Low quality**, **1 = Medium quality**, **2 = High quality**

## Citation

```bibtex
@article{he2021debertav3,
  title={Debertav3: Improving deberta using electra-style pre-training with gradient-disentangled embedding sharing},
  author={He, Pengcheng and Gao, Jianfeng and Chen, Weizhu},
  journal={arXiv preprint arXiv:2111.09543},
  year={2021}
}
```

