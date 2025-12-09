# FailRateScorer

## Overview

The **FailRateScorer** is a comprehensive evaluation framework designed to quantify the **failure rate** of **mathematical problems** by leveraging strong language models to estimate problem difficulty through multi-sample inference. This pipeline calculates the probability that a model will fail to solve a specific mathematical problem, providing an objective measure of problem complexity.

## Metric Definition:

* **Definition:** `Fail_Rate = 1 - sample_n_pass@1`

* **Explanation:** This metric estimates the **difficulty** of a problem by measuring the probability that a model gives the correct answer across multiple attempts.
  
  * A **higher value** (closer to 1) indicates the model is **more likely to fail**, suggesting the problem is **difficult**.
  * A **lower value** (closer to 0) indicates the model can **consistently provide correct answers**, suggesting the problem is **simple**.

## YAML Configuration

```yaml
name: FailRateScorer
model: Qwen/Qwen3-8B
metrics_sample_size: 4
generation_size: 4096
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"FailRateScorer"` | Identifier for the scorer |
| `model` | string | `"Qwen/Qwen3-8B"` | HuggingFace model path or local directory for the language model used in evaluation |
| `metrics_sample_size` | integer | `4` | Number of sampling attempts per problem (options: 1, 4, 8, 16, 32, 64) |
| `generation_size` | integer | `4096` | Maximum generation length for model outputs |

**Note:** Currently, `metrics_sample_size` can only be set to 1, 4, 8, 16, 32, or 64, as these are the only preset options provided by the LightEval framework. If you need other metrics or custom evaluation methods, you can refer to the LightEval official documentation: [Adding a New Metric](https://huggingface.co/docs/lighteval/main/en/adding-a-new-metric) to add them yourself.

## Underlying Framework

The evaluation pipeline is implemented using Hugging Face's [**LightEval**](https://github.com/huggingface/lighteval) framework, which provides a robust and scalable evaluation infrastructure. The pipeline uses configurable language models (e.g., `Qwen/Qwen3-8B`) and operates through the following process:

1. **Task Generation:** Custom evaluation tasks are dynamically created for each split
2. **Parallel Evaluation:** Each split is evaluated on separate GPUs using the LightEval framework
3. **Result Aggregation:** Results are collected and merged back into the original dataset format

## Scoring Process

1. **Input Processing**: For each mathematical problem, the scorer extracts:
   - Problem statement (from `instruction` and optional `input` fields)
   - Ground truth answer (from `answer`)

2. **Multi-Sample Generation**: The model generates `metrics_sample_size` solutions for each problem using the specified language model

3. **Answer Extraction**: Extracts answers from generated outputs (typically from `\boxed{...}` notation or structured answer format)

4. **Correctness Verification**: Each generated answer is compared against the ground truth to determine correctness

5. **Fail Rate Calculation**: Computes the failure rate as `1 - (number of correct answers / total attempts)`

## Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 0.75,
}
```

- `id`: Unique identifier for the input sample
- `score`: Fail rate score ranging from 0-1, where higher values indicate more difficult problems (0 = always correct, 1 = always failed)

## Citation

```bibtex
@misc{lighteval,
  author = {Habib, Nathan and Fourrier, Clémentine and Kydlíček, Hynek and Wolf, Thomas and Tunstall, Lewis},
  title = {LightEval: A lightweight framework for LLM evaluation},
  year = {2023},
  version = {0.8.0},
  url = {https://github.com/huggingface/lighteval}
}
```
