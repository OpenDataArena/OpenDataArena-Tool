# ThinkingProbScorer

## Overview

The **Thinking Probability Scorer** is a model-based evaluation tool designed to quantify the *necessity of deep reasoning* for a given **math problem**. This scorer uses a language model RL-trained with the [AdaptThink](https://github.com/THU-KEG/AdaptThink) framework to estimate the probability that the model would *engage in explicit thinking* ("Thinking" mode) versus *directly providing a solution* ("NoThinking" mode), based on the perceived difficulty of the problem.

By measuring the model's propensity to invoke a reasoning process, this scorer provides an interpretable proxy for **problem difficulty** in mathematical reasoning tasks.

## Metric Definition:

* **Definition:** 

  `Thinking_Prob = 1 - P(</think>)`

* **Explanation:** This metric estimates the *difficulty* of a problem by measuring how unlikely the model is to immediately output the `</think>` token (i.e., to choose NoThinking mode).
  
  * A **higher value** (closer to 1) indicates that the model is **less likely to skip thinking**, suggesting the problem is *hard* and requires deeper reasoning.
  * A **lower value** (closer to 0) indicates the model would confidently produce a final answer **without any thinking**, suggesting the problem is *simple* and straightforward.
  * The metric ranges from **0 to 1**, where higher scores indicate greater problem complexity.

* **Key Advantages:**
  
  * **Adaptive difficulty assessment:** Automatically detects when problems require explicit reasoning steps
  * **Model-agnostic interpretation:** Based on learned behavior from RL training rather than hand-crafted heuristics
  * **Single-token efficiency:** Requires only one forward pass to compute the thinking probability

## YAML Configuration

```yaml
name: ThinkingProbScorer
model: THU-KEG/AdaptThink-7B-delta0.05
batch_size: 128
num_gpu_per_job: 1
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"ThinkingProbScorer"` | Identifier for the scorer |
| `model` | string | `"THU-KEG/AdaptThink-7B-delta0.05"` | HuggingFace model path or local directory for the AdaptThink model |
| `batch_size` | integer | `128` | Number of samples to process in parallel per forward pass |
| `num_gpu_per_job` | integer | `1` | Number of GPUs allocated per scoring job |


## Underlying Model

The scorer uses [THU-KEG/AdaptThink-7B-delta0.05](https://huggingface.co/THU-KEG/AdaptThink-7B-delta0.05), a language model trained with reinforcement learning to **adaptively choose** between two modes:

* **Thinking Mode:** `[thinking process]</think>[final solution]` - The model generates explicit reasoning steps before providing the final answer
* **NoThinking Mode:** `</think>[final solution]` - The model directly outputs the final answer by immediately emitting the `</think>` token

The RL training enables the model to learn when deep reasoning is necessary versus when a direct answer is sufficient, making the first-token probability of `</think>` an effective proxy for problem difficulty.

## Scoring Process

1. **Input Processing**: Math problems are passed through the tokenizer with the default chat template applied to format the input correctly.

2. **Single Token Generation**: The model is instructed to generate only **one token**, and the forward pass computes the logits for all possible first tokens.

3. **Probability Extraction**: The probability of generating `</think>` as the first token is extracted from the logprobs output.

4. **Score Computation**: The metric `Thinking_Prob` is computed as `1 - P(</think>)`, where:
   - Higher probabilities of `</think>` indicate simpler problems (low Thinking_Prob)
   - Lower probabilities of `</think>` indicate harder problems requiring reasoning (high Thinking_Prob)

## Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "thinking_prob": 0.8523,
  "no_thinking_prob": 0.1477,
  "score": 0.8523
}
```

- `thinking_prob`: The computed Thinking_Prob score (1 - P(</think>))
- `no_thinking_prob`: The raw probability P(</think>) of immediately outputting the end-of-thinking token
- `score`: The final difficulty score (same as thinking_prob)

## Citation

```bibtex
@article{zhang2025adaptthink,
  title={Adaptthink: Reasoning models can learn when to think},
  author={Zhang, Jiajie and Lin, Nianyi and Hou, Lei and Feng, Ling and Li, Juanzi},
  journal={arXiv preprint arXiv:2505.13417},
  year={2025}
}
```
