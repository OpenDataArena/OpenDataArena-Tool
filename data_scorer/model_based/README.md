# Model-based Evaluation Framework

<p align="center">
  English | <a href="./README_zh-CN.md">简体中文</a>
</p>

Model-based evaluation methods leverage internal signals from existing deep learning models to assess data.

## Features

- **Configuration-Driven**: Easily configure models, evaluation metrics, and more through YAML files in `configs/`.
- **Data Parallelism**: Evaluate data on multiple GPUs in parallel.
- **Multiple Evaluation Metrics**: Supports evaluation of multiple metrics in one run. The scorers are executed in sequence and each scorer is executed with data parallelism.
- **Structured Output**: The output of each scorer will be saved in a separate file, and the final output will be merged into a single file.

## Implemented Metrics

The framework includes several evaluation dimensions, divided into two categories: **Q (Question-only)** and **QA (Question-Answer)**.

- **Q (Question-only)**: Evaluates only the question (`instruction`) itself.
  - **Deita Complexity**
  - **Thinking Probability**
- **QA (Question-Answer)**: Comprehensively evaluates both the question (`instruction`) and the answer (`output`).
  - **Deita Quality**
  - **Instruction Following Difficulty (IFD)**
  - **Reward_Model**
  - **Fail Rate**

## How to Use

1. **Configure YAML File in `configs/`** folder:
    - Set the `input_path` and `output_path` of your data.
    - Specify the `num_gpu` to parallelize the evaluation. The input data will be split into `num_gpu` parts, and each part will be evaluated in a single GPU. We assume the default runtime environment is a single-node, multi-GPU setup.
    - Configure `scorers`
      - `name`: the name of the scorer.
      - `model`: the model to be used in the scorer.
      - Other parameters are scorer-specific. **Please refer to the [wiki page](https://opendataarena-tool.readthedocs.io/en/latest/model-based-evaluation/) or the example YAML files in `configs/` for more details.**

2. **Prepare Data**:
    - Configure `input_path` in `config.yaml` to a single `jsonl` file.
    - Ensure the file format matchs [`example_input.jsonl`](../../data_scorer/data_process/example_input.jsonl).

3. **Run Evaluation**:
    - Take the `IFDScorer` as an example, run the following command:
      ```bash
      python main_para.py --config configs/IFDScorer.yaml
      ```
      You can also use other scorers running commands in run.sh script.
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
                  "Thinking_Prob": null
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
