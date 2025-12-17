# AtheneScorer

## Overview

The **AtheneScorer** is a sample-wise quality evaluation metric that leverages the Athene reward model to assess the quality of instruction-response pairs in supervised fine-tuning (SFT) datasets. Unlike traditional metrics that focus on surface-level features, AtheneScorer provides a learned quality signal by using a reward model trained to distinguish between high-quality and low-quality responses.

This metric is particularly useful for:
- **Response quality assessment**: Evaluating the helpfulness and appropriateness of model outputs
- **Dataset curation**: Filtering high-quality instruction-response pairs for training
- **RLHF data preparation**: Identifying superior responses for reinforcement learning from human feedback
- **Model evaluation**: Comparing the quality of responses from different models

The AtheneScorer supports batch processing for efficient evaluation and is based on the Llama-3-8B-Instruct architecture fine-tuned specifically as a reward model.

## Metric Definition:

* **Definition:**

  The reward score is a scalar value computed by the Athene reward model that represents the quality of a given instruction-response pair. The model processes the conversation in chat format and extracts a reward signal from the final hidden states at the CLS token position.

  ```
  Reward(instruction, input, output) = RewardModel(conversation)
  ```

  where:
  - `instruction` is the task description or question
  - `input` is additional context (optional)
  - `output` is the model's response
  - `conversation` is formatted as a chat template with user and assistant roles

* **Explanation:** The Athene reward model provides a quality assessment based on learned preferences:

  * A **higher reward score** indicates **better quality**, meaning the response is more helpful, accurate, appropriate, and well-aligned with the instruction
  * A **lower reward score** indicates **lower quality**, suggesting the response may be unhelpful, inaccurate, inappropriate, or poorly aligned with the instruction
  * The score is unbounded but typically ranges from negative to positive values, with the relative ranking being more meaningful than absolute values

## YAML Configuration

```yaml
name: AtheneScorer
model: Nexusflow/Athene-RM-8B
batch_size: 32
max_length: 4096
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"AtheneScorer"` | Identifier for the scorer |
| `model` | string | `"Nexusflow/Athene-RM-8B"` | HuggingFace model path or local checkpoint for the reward model |
| `batch_size` | integer | `32` | Number of samples to process in parallel per forward pass |
| `max_length` | integer | `4096` | Maximum sequence length for tokenization; longer sequences are truncated from the left |


## Underlying Model

The scorer uses the **Nexusflow/Athene-RM-8B** reward model, which is based on the **Llama-3-8B-Instruct** architecture fine-tuned specifically for reward modeling. The model consists of a Llama-3-8B transformer backbone with a linear reward head that projects hidden states to a scalar reward score. A special CLS token is appended to conversations for reward extraction from the final hidden states.

## Scoring Process

1. **Input Formatting**: For each sample, construct a chat conversation with user role (instruction + input) and assistant role (output)

2. **Chat Template Application**: Apply the Llama-3 chat template to format the conversation and append the CLS token

3. **Tokenization**: Tokenize the formatted text; truncate from the left if exceeding `max_length`

4. **Batch Processing**: Process samples in batches with padding for efficient GPU computation

5. **Reward Computation**: Forward pass through the model to extract reward value at the CLS token position from the final hidden states

6. **Score Extraction**: Convert tensor outputs to float values and return scores for each sample

## Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 2.456
}
```

- `id`: Unique identifier of the sample from the input dataset
- `score`: Reward score computed by the Athene model (unbounded float value; higher is better)

## Citation

```bibtex
@misc{Athene2024,
    title = {Athene-70B: Redefining the Boundaries of Post-Training for Open Models},
    url = {https://nexusflow.ai/blogs/athene},
    author = {Frick, Evan and Jin, Peter and Li, Tianle and Ganesan, Karthik and Zhang, Jian and Jiao, Jiantao and Zhu, Banghua},    
    month = {July},
    year = {2024}
}
```

