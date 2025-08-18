# LLM-作为评审的自动评估框架

<p align="center">
  <a href="./README.md">English</a> | 简体中文
</p>

这是一个利用 LLM 作为评审器（Judge）来自动评估生成式 AI 输出的框架。

## 已实现的评估指标

框架包含多个评估维度，分为两类：**Q（仅问题）** 和 **QA（问题-答案）**。

- **Q（仅问题）**：只评估问题（`instruction`）。
  - All（包含以下所有指标）
  - Clarity（清晰度）
  - Code_Difficulty（代码难度，来自 OpenThoughts）
  - Coherence（连贯性）
  - Completeness（完整性）
  - Complexity（复杂性）
  - Correctness（正确性）
  - Math_Difficulty（数学难度，来自 OpenThoughts）
  - Meaningness（意义性）

- **QA（问题-答案）**：综合评估问题（`instruction`）和答案（`output`）。
  - All（包含以下所有指标）
  - Clarity（清晰度）
  - Coherence（连贯性）
  - Completeness（完整性）
  - Complexity（复杂性）
  - Correctness（正确性）
  - Meaningness（意义性）
  - Relevance（相关性）

## 使用方法

1. **配置 `config.yaml`**：
    - 设置 `api_key` 和 `base_url`。
    - 指定使用的 `model`，以及 `temperature` 和 `top_p`。
    - 配置 `concurrency`（并发请求数）。
    - 配置 `chunk_size`：设定处理的块大小，即一次读入内存和写入的行数。
    - 配置 `id_track_file`（可选）：用于避免重复评分。程序会在启动时读取该追踪文件，记录已评分的 ID，并自动跳过它们。
    - 在 `metrics` 部分定义用于评估 Q 和 QA 的指标，这些名称必须与 `prompts/` 目录下的文件名对应。
    - 指定 `output_dir`。

2. **准备数据**：
    - 在 `config.yaml` 中配置 `input_file` 为单个 `jsonl` 文件。
    - 确保文件格式符合 [`example_input.jsonl`](/data_scorer/data_process/example_input.jsonl)。

3. **准备提示词（Prompts）**：
    - 在 `prompts/` 目录下为每个配置的指标创建提示词文件。
    - 文件名格式为 `模式_指标名.txt`（如 `QA_Correctness.txt` 或 `Q_Clarity.txt`）。
    - 在 QA 模式下，提示词文件中可使用 `{instruction}` 和 `{output}` 占位符。

4. **运行评估**：

    ```bash
    python -m llm_as_judge.main
    ```

    或者，使用其他配置文件：

    ```bash
    python -m llm_as_judge.main --config-path /path/to/your/config.yaml
    ```

5. **查看结果**：
    - 评估完成后，会在 `output/` 目录下生成相应的 `*_scored.jsonl` 文件。
    - 文件为标准 JSON Lines 格式，每行是一个 JSON 对象，包含 `id` 和 `scores` 字段。例如：

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

6. **查看错误日志**：
    - 多次重试后仍未能评估的条目，会被自动记录在 `output/` 目录下的 `*_errors.jsonl` 文件中。
    - 每个 JSON 对象包含原始数据、失败的维度以及详细错误信息。例如：

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

7. **后处理（Post-Process）**：
    - 为了与其他工具对齐并保持灵活性，可以进行可选的后处理步骤。
    - 主评估脚本输出的 `scores.jsonl` 文件仅包含 `id` 和 `scores`。
    - 可以使用 `process_scores.py` 将这些分数合并回原始数据文件中：

        ```bash
        python tools/process_scores.py --scores_file [PATH_TO_SCORES_FILE] --data_file [PATH_TO_DATA_FILE] --output_file [PATH_TO_OUTPUT_FILE]
        ```

        输出文件中，原本的 `null` 占位符会被替换。例如：

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
