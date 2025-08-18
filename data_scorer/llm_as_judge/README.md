# LLM-as-Judge Evaluation Framework

<p align="center">
  English | <a href="./README_zh-CN.md">简体中文</a>
</p>

This is a framework that uses a LLM as a judge to automatically evaluate the output of generative AI.

## Implemented Metrics

The framework includes several evaluation dimensions, divided into two categories: **Q (Question-only)** and **QA (Question-Answer)**.

- **Q (Question-only)**: Evaluates only the question (`instruction`) itself.
  - All (include all the metrics below)
  - Clarity
  - Code_Difficulty (from OpenThoughts)
  - Coherence
  - Completeness
  - Complexity
  - Correctness
  - Math_Difficulty (from OpenThoughts)
  - Meaningness
- **QA (Question-Answer)**: Comprehensively evaluates both the question (`instruction`) and the answer (`output`).
  - All (include all the metrics below)
  - Clarity
  - Coherence
  - Completeness
  - Complexity
  - Correctness
  - Meaningness
  - Relevance

## How to Use

1. **Configure `config.yaml`**:
    - Set your `api_key` and `base_url`.
    - Specify the `model` to be used and model `temperture` and `top_p`.
    - Configure `concurrency` (number of concurrent requests).
    - Configure `chunk_size`: Sets the chunk size for processing, determining the number of data rows read into memory at once and write interval.
    - Configure the `id_track_file` (optional): To prevent reprocessing of already scored data, you can enable ID tracking. The program will read a specified tracking file at startup, record the IDs of all scored items, and automatically skip them during processing.
    - In the `metrics` section, define the list of metrics for evaluating Q and QA. These names must correspond to the filenames in the `prompts/` directory.
    - Specify `output_dir`.

2. **Prepare Data**:
    - Configure `input_file` in `config.yaml` to a single `jsonl` file.
    - Ensure the file format matchs [`example_input.jsonl`](/data_scorer/data_process/example_input.jsonl).

3. **Prepare Prompts**:
    - In the `prompts/` directory, create a prompt file for each metric defined in `config.yaml`.
    - The filename format is `Mode_MetricName.txt` (e.g., `QA_Correctness.txt` or `Q_Clarity.txt`).
    - Use `{instruction}` and `{output}` (QA only) as placeholders in the prompt files.

4. **Run Evaluation**:

    ```bash
    python -m llm_as_judge.main
    ```

    Alternatively, if you want to use a different configuration file:

    ```bash
    python -m llm_as_judge.main --config-path /path/to/your/config.yaml
    ```

5. **View Results**:
    - After the evaluation is complete, corresponding `*_scored.jsonl` files will be generated in the `output/` directory.
    - Each file is in standard JSON Lines format, with each line being a separate JSON object containing `id` and `scores` fields. For example:

        ```json
        {
            "id": 0,
            "scores": {
                "Q": {
                    "All": {
                        "Clarity": 10,
                        "Coherence": 10,
                        "Completeness": 10,
                        "Complexity": 3,
                        "Correctness": 10,
                        "Meaningfulness": 8
                    }
                },
                "QA": {
                    "All": {
                        "Clarity": 9,
                        "Coherence": 9,
                        "Completeness": 9,
                        "Complexity": 3,
                        "Correctness": 10,
                        "Meaningfulness": 8,
                        "Relevance": 10
                    }
                }
            }
        }
        ```

6. **View Error Logs**:
    - Entries that still fail to be evaluated after multiple retries are automatically logged in the corresponding `*_errors.jsonl` file in the `output/` directory.
    - Each JSON object includes the original data, the failed evaluation dimension, and detailed error information for easy troubleshooting. For example:

        ```json
        {
            "original_item": {
                "id": "0",
                "instruction": "...",
                "response": "..."
            },
            "metric": "Correctness",
            "mode": "QA",
            "error_details": {
                "error": "Error after 4 attempts: Output truncated by model (finish_reason='length')"
            }
        }
        ```

7. **Post-Process**:
    - To align the output with other tools and maintain flexibility, you can perform an optional post-processing step. The main evaluation script outputs a `scores.jsonl` file containing only the `id` and `scores`. You can use `process_scores.py` to merge these scores back into your original data file, which may contain other fields.

        ```bash
        python tools/process_scores.py --scores_file [PATH_TO_SCORES_FILE] --data_file [PATH_TO_DATA_FILE] --output_file [PATH_TO_OUTPUT_FILE]
        ```

        The result will be a output file that corresponding placeholder `null` replaced.

        ```json
        {
            "instruction": "At what point do the graphs of $y=5x-26$ and $y=-\\frac{3}{4}x+19$ intersect?",
            "output": "To find the point of intersection ...",
            "id": 0,
            "Q_scores": {
                "Clarity": 9,
                "Coherence": 10,
                "Completeness": 9,
                "Complexity": 4,
                "Correctness": 10,
                "Meaningfulness": 8,
                "Difficulty": null,
                "Deita_Complexity": null,
                "Thinking_Prob": null
            },
            "QA_scores": {
                "Clarity": 9,
                "Coherence": 9,
                "Completeness": 10,
                "Complexity": 7,
                "Correctness": 10,
                "Meaningfulness": 8,
                "Relevance": 10,
                "IFD": null,
                "Deita_Quality": null,
                "Reward_Model": null,
                "A_Length": null,
                "Fail_Rate": null
            }
        }
        ```
