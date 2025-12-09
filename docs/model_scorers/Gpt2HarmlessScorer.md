# Gpt2HarmlessScorer

## Overview

The **GPT2 Harmless Scorer** is a model-based evaluation tool designed to assess the harmlessness of instruction-response pairs in supervised fine-tuning (SFT) data. This scorer leverages a GPT2-large reward model specifically trained on the Anthropic/hh-rlhf harmless dataset to detect potentially harmful responses and evaluate alignment safety. The model achieves an accuracy of **0.73698** on the test set, matching the performance of larger models while maintaining computational efficiency.

## Metric Definition:

* **Definition:** 

  Given an instruction-response pair `(Q, A)`, the harmlessness reward model assigns a scalar reward score representing the safety and harmlessness of the response in the context of the instruction.

* **Explanation:** The reward score quantifies how safe and harmless a response is relative to the given instruction:
  
  * A **higher reward score** indicates the response is **more harmless and safer**, suggesting good alignment with safety preferences.
  * A **lower reward score** suggests the response may contain **harmful, toxic, or unsafe content**, indicating potential alignment issues.

* **Formulation:** Following the Anthropic/hh-rlhf dataset format:
  
  ```
  Input: "\n\nHuman: {instruction}\n\nAssistant:"
  Output: {response}
  ```

## YAML Configuration

```yaml
name: Gpt2HarmlessScorer
model: Ray2333/gpt2-large-harmless-reward_model
batch_size: 8
max_length: 1024
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"Gpt2HarmlessScorer"` | Identifier for the scorer. Must be set to `Gpt2HarmlessScorer` |
| `model` | string | `"Ray2333/gpt2-large-harmless-reward_model"` | Path or name of the harmless reward model from Hugging Face Hub. You can specify alternative GPT2-based harmlessness models trained on similar datasets |
| `batch_size` | integer | `8` | Number of samples to process simultaneously during inference. Adjust based on GPU memory availability |
| `max_length` | integer | `1024` | Maximum sequence length (in tokens) for tokenization. Sequences exceeding this length will be truncated |

## Underlying Model

The scorer uses [**Ray2333/gpt2-large-harmless-reward_model**](https://huggingface.co/Ray2333/gpt2-large-harmless-reward_model), a GPT2-large model fine-tuned as a reward model on the **Anthropic/hh-rlhf harmless dataset**. 

### Model Characteristics

* **Base Architecture:** GPT2-large (774M parameters)
* **Training Data:** Anthropic/hh-rlhf harmless subset
* **Task:** Binary sequence classification (harmlessness scoring)
* **Precision:** bfloat16 for efficient inference
* **Performance:** 73.7% accuracy on the harmless test set

**Important Note:** This reward model differs from other open-source reward models trained on the full Anthropic/hh-rlhf dataset, as it focuses exclusively on the harmless subset for specialized harmlessness evaluation.

## Scoring Process

The GPT2 Harmless Scorer follows a structured pipeline to evaluate instruction-response pairs:

### 1. **Input Formatting**

For each data item containing `instruction`, `input` (optional), and `output` fields:

```python
# If input field exists:
Q = "\n\nHuman: {instruction}\n{input}\n\nAssistant:"

# If input field is empty:
Q = "\n\nHuman: {instruction}\n\nAssistant:"

A = {output}
```

This formatting follows the Anthropic/hh-rlhf dataset convention, ensuring compatibility with the reward model's training format.

### 2. **Tokenization**

The instruction `Q` and response `A` are tokenized together with:
* **Truncation:** Enabled with `max_length` parameter
* **Padding:** Applied to create uniform batch sizes
* **Warning System:** Samples exceeding `max_length` (estimated at ~4 characters per token) trigger warnings before truncation

### 3. **Model Inference**

The tokenized inputs are passed through the reward model:
```python
with torch.no_grad():
    logits = model(**inputs).logits  # Shape: [batch_size, 1]
    rewards = logits.squeeze(-1)     # Extract scalar rewards
```

### 4. **Batch Processing**

* Samples are processed in batches of size `batch_size`
* Progress is tracked with a progress bar showing completion status
* Remaining samples in the final incomplete batch are processed separately

## Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 2.45
}
```

- `id`: Unique identifier for the data sample, extracted from the input data's `id` field. If the input lacks an `id` field, an empty string is used
- `score`: The harmlessness reward score assigned by the model. Higher values (e.g., > 0) indicate harmless, safe responses. Lower values (e.g., < 0) indicate potentially harmful or unsafe responses. The score range depends on the model's training distribution

## Citation

This reward model was developed and utilized for multi-objective alignment research, particularly focusing on harmlessness and helpfulness alignment objectives:

```bibtex
@article{yang2024rewards,
  title={Rewards-in-Context: Multi-objective Alignment of Foundation Models with Dynamic Preference Adjustment},
  author={Yang, Rui and Pan, Xiaoman and Luo, Feng and Qiu, Shuang and Zhong, Han and Yu, Dong and Chen, Jianshu},
  journal={International Conference on Machine Learning},
  year={2024}
}
```

