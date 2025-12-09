# ThinkOrNotScorer

## Overview

The **ThinkOrNot Scorer** is a lightweight, rule-based evaluation tool designed to identify whether a response contains explicit reasoning traces in reasoning-augmented datasets. Motivated by the methodology in [OpenCodeReasoning (Ahmad et al., 2025)](https://arxiv.org/pdf/2504.01943), this scorer performs binary classification to detect the presence of thinking tags (`<think>...</think>` or `<redacted_reasoning>...</redacted_reasoning>`) in model-generated responses.

Unlike more sophisticated scorers that evaluate reasoning quality, ThinkOrNotScorer focuses solely on **presence detection**, making it extremely fast and suitable for large-scale dataset preprocessing, data filtering, and quality control in reasoning distillation pipelines.

## Metric Definition:

* **Definition:** 

  The ThinkOrNot Score is a binary indicator:
  
  * **Score = 1**: The response **contains** thinking tags (`<think>...</think>` or `<redacted_reasoning>...</redacted_reasoning>`)
  * **Score = 0**: The response **does not contain** any thinking tags

* **Explanation:** This metric provides a simple presence/absence classification:
  
  * A **score of 1** indicates the response includes explicit reasoning traces, which is typical for distilled reasoning models or chain-of-thought augmented data.
  * A **score of 0** indicates the response is a standard format without explicit reasoning sections, either because the model directly provided an answer or the thinking tags were not generated.

## YAML Configuration

```yaml
name: ThinkOrNotScorer
field: output
max_workers: 8
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"ThinkOrNotScorer"` | Identifier for the scorer |
| `field` | string | `"output"` | The field name in the dataset to examine (model-generated responses) |
| `max_workers` | integer | CPU count | Number of parallel worker processes for scoring |


## Underlying Model

This scorer **does not require any language model**. It operates entirely through rule-based pattern matching using compiled regular expressions. The scorer detects the following tag patterns (case-insensitive):
* `<think>` and `</think>`
* `<think >` and `</think >` (with optional trailing spaces)
* `<redacted_reasoning>` and `</redacted_reasoning>`
* `<redacted_reasoning >` and `</redacted_reasoning >` (with optional trailing spaces)

## Scoring Process

1. **Extract Text**: Read the specified field (e.g., `output`) from each data item in the dataset

2. **Validate Field**: Check if the field exists and contains valid string content (missing or empty field returns score 0)

3. **Pattern Matching**: Search for thinking tag patterns using pre-compiled regular expressions with case-insensitive matching

4. **Binary Classification**: If any thinking tag is found, return score 1; otherwise return score 0

5. **Parallel Processing**: Distribute data items across multiple worker processes using Python's ProcessPoolExecutor for efficient large-scale processing

## Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 1.0
}
```

- `id`: The unique identifier of the data item (defaults to "unknown" if not present)
- `score`: The binary ThinkOrNot score (1.0 for reasoning present, 0.0 for standard response)

## Citation

```bibtex
@misc{opendataarena_tool_2025,
  author       = {OpenDataArena},
  title        = {{OpenDataArena-Tool}},
  year         = {2025},
  url          = {https://github.com/OpenDataArena/OpenDataArena-Tool},
  note         = {GitHub repository},
  howpublished = {\url{https://github.com/OpenDataArena/OpenDataArena-Tool}},
}
```
