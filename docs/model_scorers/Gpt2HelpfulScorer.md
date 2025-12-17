# Gpt2HelpfulScorer

## Overview

The **GPT2 Helpful Scorer** is a model-based evaluation tool designed to assess the helpfulness of instruction-response pairs in supervised fine-tuning (SFT) data. This scorer leverages a GPT2-large reward model specifically trained on the Anthropic/hh-rlhf helpful dataset to evaluate how useful and informative responses are in addressing user instructions. The model achieves an accuracy of **0.72621** on the test set, matching the performance of larger models while maintaining computational efficiency.

## Metric Definition:

* **Definition:** 

  Given an instruction-response pair `(Q, A)`, the helpfulness reward model assigns a scalar reward score representing the usefulness and informativeness of the response in addressing the instruction.

* **Explanation:** 

  The reward score quantifies how helpful and informative a response is relative to the given instruction:
  
  * A **higher reward score** indicates the response is **more helpful and informative**, providing relevant, accurate, and actionable information that effectively addresses the user's query.
  * A **lower reward score** suggests the response is **less helpful**, potentially vague, irrelevant, or failing to adequately address the instruction.

* **Formulation:** 

  Following the Anthropic/hh-rlhf dataset format:
  ```
  Input: "\n\nHuman: {instruction}\n\nAssistant:"
  Output: {response}
  ```

## YAML Configuration

```yaml
name: Gpt2HelpfulScorer
model: Ray2333/gpt2-large-helpful-reward_model
batch_size: 8
max_length: 1024
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"Gpt2HelpfulScorer"` | Identifier for the scorer. Must be set to `Gpt2HelpfulScorer` |
| `model` | string | `"Ray2333/gpt2-large-helpful-reward_model"` | Path or name of the helpful reward model from Hugging Face Hub. You can specify alternative GPT2-based helpfulness models trained on similar datasets |
| `batch_size` | integer | `8` | Number of samples to process simultaneously during inference. Adjust based on GPU memory availability |
| `max_length` | integer | `1024` | Maximum sequence length (in tokens) for tokenization. Sequences exceeding this length will be truncated |

## Underlying Model

The scorer uses [**Ray2333/gpt2-large-helpful-reward_model**](https://huggingface.co/Ray2333/gpt2-large-helpful-reward_model), a GPT2-large model fine-tuned as a reward model on the **Anthropic/hh-rlhf helpful dataset**. 

### Model Characteristics

* **Base Architecture:** GPT2-large (774M parameters)
* **Training Data:** Anthropic/hh-rlhf helpful subset
* **Task:** Binary sequence classification (helpfulness scoring)
* **Precision:** bfloat16 for efficient inference
* **Performance:** 72.6% accuracy on the helpful test set

**Important Note:** This reward model differs from other open-source reward models trained on the full Anthropic/hh-rlhf dataset, as it focuses exclusively on the helpful subset for specialized helpfulness evaluation.

## Scoring Process

1. **Input Formatting**: For each data item containing `instruction`, `input` (optional), and `output` fields:
   ```python
   # If input field exists:
   Q = "\n\nHuman: {instruction}\n{input}\n\nAssistant:"
   
   # If input field is empty:
   Q = "\n\nHuman: {instruction}\n\nAssistant:"
   
   A = {output}
   ```
   This formatting follows the Anthropic/hh-rlhf dataset convention, ensuring compatibility with the reward model's training format.

2. **Tokenization**: The instruction `Q` and response `A` are tokenized together with truncation enabled (`max_length` parameter), padding applied to create uniform batch sizes, and warning system for samples exceeding `max_length`

3. **Model Inference**: The tokenized inputs are passed through the reward model to extract scalar reward scores
   ```python
   with torch.no_grad():
       logits = model(**inputs).logits  # Shape: [batch_size, 1]
       rewards = logits.squeeze(-1)     # Extract scalar rewards
   ```

4. **Batch Processing**: Samples are processed in batches of size `batch_size` with progress tracking

## Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 3.12
}
```

- `id`: Unique identifier for the data sample, extracted from the input data's `id` field. If the input lacks an `id` field, an empty string is used
- `score`: The helpfulness reward score assigned by the model. Higher values (e.g., > 0) indicate helpful, informative responses that effectively address the instruction. Lower values (e.g., < 0) indicate less helpful responses that may be vague, irrelevant, or inadequate

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

