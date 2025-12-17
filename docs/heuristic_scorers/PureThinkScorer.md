# PureThinkScorer

## Overview

The **PureThink Scorer** is a rule-based evaluation tool designed to assess the quality of reasoning-based code generation data, particularly for datasets containing explicit thinking processes. Inspired by the methodology in [OpenCodeReasoning (Ahmad et al., 2025)](https://arxiv.org/pdf/2504.01943), this scorer evaluates whether the reasoning traces (enclosed in `<think>...</think>` or `<redacted_reasoning>...</redacted_reasoning>` tags) are **pure thinking** - containing only logical reasoning without executable code - while the final solution code exists separately outside the thinking tags.

This scorer is particularly useful for filtering and evaluating distilled reasoning datasets where models generate chain-of-thought reasoning before producing code solutions, ensuring that the thinking process remains conceptual and the implementation details are properly separated.

## Metric Definition:

* **Definition:** 

  The PureThink Score is a categorical metric that evaluates the structure and quality of reasoning-augmented code generation data:
  
  * **Score = 1**: **Ideal** - Contains thinking tags with pure reasoning (no code blocks inside thinking section), and has executable code blocks outside the thinking section.
  * **Score = 0**: **Mixed** - Contains thinking tags, has code blocks outside thinking section, but also contains code blocks within the thinking section (reasoning is contaminated with implementation details).
  * **Score = -1**: **Incomplete** - Contains thinking tags but lacks executable code blocks after removing the thinking section (missing final implementation).
  * **Score = -2**: **No Reasoning** - No thinking tags detected in the response.

* **Explanation:** 

  This metric ensures the separation of concerns in reasoning-based code generation:
  
  * A **score of 1** indicates high-quality data where reasoning and implementation are properly separated, which is ideal for training models to think before coding.
  * A **score of 0** suggests that code appears prematurely in the reasoning phase, potentially reducing the quality of the thinking process.
  * A **score of -1** indicates incomplete responses where reasoning exists but implementation is missing.
  * A **score of -2** indicates responses without explicit reasoning traces.

## YAML Configuration

```yaml
name: PureThinkScorer
field: output  # The field name in the dataset to examine (default: "output")
max_workers: 8  # Number of parallel worker processes for scoring (default: CPU count)
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"PureThinkScorer"` | Identifier for the scorer |
| `field` | string | `"output"` | The key in each data item's JSON structure that contains the text to be evaluated. This should point to the field containing model-generated responses with potential thinking tags |
| `max_workers` | integer | CPU count | Number of parallel processes for multi-core processing. Higher values increase processing speed for large datasets |


## Underlying Model

This scorer **does not require any language model**. It operates purely through rule-based pattern matching and regex-based text analysis, making it extremely fast and resource-efficient. The scorer:

* Uses compiled regular expressions to detect thinking tags: `<think>`, `</think>`, `<redacted_reasoning>`, `</redacted_reasoning>`
* Employs pattern matching to identify markdown code blocks (` ```language...``` `)
* Performs deterministic text processing without any neural network inference

## Scoring Process

1. **Extract Text**: Read the specified field (e.g., `output`) from each data item

2. **Detect Thinking Tags**: Check if the text contains `<think>...</think>` or `<redacted_reasoning>...</redacted_reasoning>` tags
   - If **no thinking tags found** → Return score **-2**

3. **Extract Components**: If thinking tags exist:
   - Extract content **within** thinking tags (thinking content)
   - Extract content **outside** thinking tags (remaining text)

4. **Check Code in Remaining Text**: Examine if remaining text contains markdown code blocks (` ```...``` `)
   - If **no code blocks in remaining text** → Return score **-1**

5. **Check Code in Thinking Content**: If code blocks exist in remaining text, check if thinking content also contains code blocks
   - If **thinking content contains code blocks** → Return score **0**
   - If **thinking content has NO code blocks** → Return score **1** (ideal)

6. **Code Block Detection**: The scorer identifies code blocks using markdown syntax with pattern ` ```language\ncode\n``` ` or ` ```\ncode\n``` `, supporting common languages like `python`, `java`, `cpp`, `javascript`, etc.

7. **Parallel Processing**: The scorer leverages Python's `ProcessPoolExecutor` to distribute data items across multiple worker processes for efficient processing, with progress tracked using `tqdm` progress bars

## Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 1.0
}
```

- `id`: The unique identifier of the data item (extracted from the original data's `"id"` field). If no `id` field exists in the input, defaults to `"unknown"`
- `score`: The PureThink score (float value)
  - `1.0`: Pure thinking with proper code separation (ideal)
  - `0.0`: Thinking contaminated with code blocks
  - `-1.0`: Thinking exists but no final code implementation
  - `-2.0`: No thinking tags detected

## Citation

```bibtex
@article{ahmad2025opencodereasoning,
  title={Opencodereasoning: Advancing data distillation for competitive coding},
  author={Ahmad, Wasi Uddin and Narenthiran, Sean and Majumdar, Somshubra and Ficek, Aleksander and Jain, Siddhartha and Huang, Jocelyn and Noroozi, Vahid and Ginsburg, Boris},
  journal={arXiv preprint arXiv:2504.01943},
  year={2025}
}
```

