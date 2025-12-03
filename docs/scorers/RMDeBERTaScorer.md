# RMDeBERTaScorer

## Overview

The **RMDeBERTa Scorer** is a reward model-based evaluation tool designed to assess the quality of instruction-response pairs in supervised fine-tuning (SFT) data. It leverages the OpenAssistant reward model trained on human feedback to predict how preferable a generated response is for a given instruction. The model was trained to distinguish between better and worse responses as judged by humans, making it valuable for quality assessment, QA model evaluation, and toxic response detection.

The scorer utilizes a DeBERTa-v3-large architecture fine-tuned on multiple high-quality preference datasets, achieving strong performance across diverse evaluation benchmarks. It provides a scalar reward score indicating response quality, where higher scores suggest better alignment with human preferences.

## Metric Definition:

* **Definition:** 

  Given an instruction-response pair (Q, A), the reward model outputs a scalar score representing the expected human preference for the response A given instruction Q. The score is computed as `score = model(Q, A).logits[0]`.

* **Explanation:** The reward score quantifies how well a response aligns with human judgment of quality, helpfulness, and appropriateness.
  
  * A **higher reward score** indicates that the response is more likely to be preferred by humans, suggesting better quality, helpfulness, and instruction-following.
  * A **lower reward score** suggests that the response may be less helpful, less accurate, or potentially problematic.

## YAML Configuration

```yaml
name: RMDeBERTaScorer
model: OpenAssistant/reward-model-deberta-v3-large-v2
max_length: 512
batch_size: 32
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"RMDeBERTaScorer"` | Scorer identifier used for logging and output organization |
| `model` | string | `"OpenAssistant/reward-model-deberta-v3-large-v2"` | HuggingFace model path or local model directory |
| `max_length` | integer | `512` | Maximum token length for input sequences (instruction + response). Sequences exceeding this length will be truncated |
| `batch_size` | integer | `32` | Number of samples to process simultaneously. Larger values increase throughput but require more GPU memory |


## Underlying Model

The scorer uses [OpenAssistant/reward-model-deberta-v3-large-v2](https://huggingface.co/OpenAssistant/reward-model-deberta-v3-large-v2), a reward model based on the DeBERTa-v3-large architecture. This model was trained on diverse human feedback datasets including:

* **webgpt_comparisons**: Web-based question answering comparisons
* **summarize_from_feedback**: Summarization preference data
* **synthetic-instruct-gptj-pairwise**: Synthetic instruction-following pairs
* **anthropic_hh-rlhf**: Anthropic's helpfulness and harmlessness dataset

The model achieves strong validation accuracy across benchmarks, with 61.57% on WebGPT, 71.47% on Summary tasks, 99.88% on SyntheticGPT, and 69.25% on Anthropic RLHF datasets.

If the specified model cannot be loaded, the scorer automatically falls back to the default OpenAssistant model. You can also use other compatible reward models following the same architecture and tokenization scheme.

## Scoring Process

The RMDeBERTa Scorer evaluates instruction-response pairs through the following pipeline:

1. **Input Preparation:**
   * For each data item, the scorer extracts the `instruction` field and optional `input` field
   * If `input` exists, it concatenates with instruction: `question = instruction + "\n" + input`
   * Otherwise, uses instruction alone: `question = instruction`
   * The `output` field serves as the response to be evaluated

2. **Batch Tokenization:**
   * Question-answer pairs are tokenized together using the model's tokenizer
   * Sequences are padded to uniform length and truncated to `max_length` if necessary
   * A warning is issued for truncated samples with their IDs logged

3. **Model Inference:**
   * Tokenized inputs are passed through the reward model in batches
   * The model outputs logits representing preference scores
   * Scores are extracted as `logits[:, 0]` and converted to CPU tensors

4. **Score Extraction:**
   * Raw scores are returned as floating-point values
   * No normalization or scaling is applied to preserve interpretability

## Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 2.347
}
```

- `id`: Unique identifier for the data sample, taken from the input data's `id` field. If not present, defaults to `"index_{i}"` where `i` is the sample position
- `score`: Floating-point reward score indicating response quality. Higher values indicate better quality and stronger alignment with human preferences

## Citation

```bibtex
@misc{openassistant_debertav3_rewardmodel_v2,
  title        = {OpenAssistant Reward Model - DeBERTa-v3-large-v2},
  author       = {{OpenAssistant Team}},
  howpublished = {\url{https://huggingface.co/OpenAssistant/reward-model-deberta-v3-large-v2}},
  note         = {Accessed: 2025-03-xx},
  year         = {2023}
}
```
