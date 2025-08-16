# 基于模型的评估框架

<p align="center">
  <a href="./README.md">English</a> | 简体中文
</p>

基于模型的评估方法利用现有深度学习模型的内部信号来评估数据。

## 特性

- **Configuration-Driven**: 通过 `configs/` 中的 YAML 文件轻松配置模型、评估指标等。
- **Data Parallelism**: 数据并行。
- **Multiple Evaluation Metrics**: 支持在一次运行中评估多个指标。评分器按顺序执行，每个评分器都使用数据并行。
- **Structured Output**: 每个评分器的输出将保存在单独的文件中，最终输出将合并到单个文件中。

## 支持的指标

该框架包括几个评估维度，分为两个类别: **Q (Question-only)** 和 **QA (Question-Answer)**。

- **Q (Question-only)**: 仅评估问题 (`instruction`) 本身。
  - **Deita Complexity**
  - **Thinking Probability**
- **QA (Question-Answer)**: 全面评估问题 (`instruction`) 和答案 (`output`)。
  - **Deita Quality**
  - **Instruction Following Difficulty (IFD)**
  - **Reward_Model**
  - **Fail Rate**

## 如何使用

1. **配置 `configs/` 中的 YAML 文件**:
    - 设置数据 `input_path` 和 `output_path`。
    - 指定 `num_gpu` 以并行评估。输入数据将分成 `num_gpu` 份，每份将在单个 GPU 上评估。我们假设默认运行环境是单节点多 GPU。
    - 配置 `scorers`
      - `name`: 评分器的名称。
      - `model`: 评分器使用的模型。
      - 其他参数是评分器特定的。**请参阅 [wiki 页面](https://opendataarena-tool.readthedocs.io/en/latest/model-based-evaluation/) 或 `configs/` 中的示例 YAML 文件获取更多详细信息。**

2. **准备数据**:
    - 在 `config.yaml` 中配置 `input_path`，使之指向单个 `jsonl` 文件。
    - 确保文件格式与 [`example_input.jsonl`](../../data_scorer/data_process/example_input.jsonl) 一致。

3. **运行评估**:
    - 以 `IFDScorer` 为例，运行以下命令:
      ```bash
      python main_para.py --config configs/IFDScorer.yaml
      ```
      您也可以在 run.sh 脚本中使用其他评分器的运行命令。
    - 输出将保存在 YAML 中指定的 `output_path` 中，其中包含以下文件:
        - `temp/`: 临时文件，包括分割的数据和每个评分器的分数。分数格式如下:
            ```json
            {
              "id": 0, 
              "Deita_Quality": 2.23,
            }
            ```
        - `output.jsonl`: 评估的最终输出，分数会被合并到原始数据中。例如:
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
