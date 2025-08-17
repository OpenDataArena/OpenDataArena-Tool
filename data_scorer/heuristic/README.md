# Heuristic Evaluation Framework

<p align="center">
  English | <a href="./README_zh-CN.md">简体中文</a>
</p>

Heuristic evaluation methods leverage heuristic signals like response length to assess data.

## Features

- **Configuration-Driven**: Easily configure models, evaluation metrics, and more through YAML files in `configs/`.
- **Multiple Evaluation Metrics**: Supports evaluation of multiple metrics in one run. The scorers are executed in sequence and each scorer is executed with data parallelism.
- **Structured Output**: The output of each scorer will be saved in a separate file, and the final output will be merged into a single file.

## Implemented Metrics

The framework includes several evaluation dimensions, divided into two categories: **Q (Question-only)** and **QA (Question-Answer)**.

- **QA (Question-Answer)**: Comprehensively evaluates both the question (`instruction`) and the answer (`output`).
  - **Output Token Length**

## How to Use

1. **Configure YAML File in `configs/`** folder:
    - Set the `input_path` and `output_path` of your data.
    - Configure `scorers`
      - `name`: the name of the scorer.
      - `model`: the model to be used in the scorer.
      - Other parameters are scorer-specific. **Please refer to the [wiki page](https://opendataarena-tool.readthedocs.io/en/latest/model-based-evaluation/) or the example YAML files in `configs/` for more details.**

2. **Prepare Data**:
    - Configure `input_file` in `config.yaml` to a single `jsonl` file.
    - Ensure the file format matchs [`example_input_add_key.jsonl`](data_process/example_input_add_key.jsonl).

3. **Run Evaluation**:
    - Run the following command:
    ```bash
    bash sh/Length.sh
    ```
      You can also use other scorers by running the corresponding shell script.
    - The output will be saved in the `output_path` specified in YAML, which will contain the following files:
        - `temp/`: temporary files, including the splitted data and the scores of each scorer. The scores are formatted as follows:
            ```json
            {
              "id": 0, 
              "Deita_Quality": 2.23,
            }
            ```
        - `output.jsonl`: the final output of the evaluation, where the scores are merged into the original data. For example:
          ```json
          {
              "instruction": "At what point do the graphs of $y=5x-26$ and $y=-\\frac{3}{4}x+19$ intersect?",
              "output": "To find the point of intersection ...",
              "id": 0,
              "Q_scores": {
                  "Clarity": null,
                  "Coherence": null,
                  "Completeness": null,
                  "Complexity": null,
                  "Correctness": null,
                  "Meaningfulness": null,
                  "Difficulty": null,
                  "Deita_Complexity": 2.23,
                  "Thinking_Prob": null,
              },
              "QA_scores": {
                  "Clarity": null,
                  "Coherence": null,
                  "Completeness": null,
                  "Complexity": null,
                  "Correctness": null,
                  "Meaningfulness": null,
                  "Relevance": null,
                  "IFD": null,
                  "Deita_Quality": null,
                  "Reward_Model": null,
                  "A_Length": null,
                  "Fail_Rate": null
              }
          }
          ```