# 启发式评估框架

<p align="center">
  <a href="./README.md">English</a> | 简体中文
</p>

启发式评估方法利用诸如响应长度等启发式信号来评估数据。

## 特性

* **配置驱动**：可以通过 `configs/` 文件夹中的 YAML 文件轻松配置模型、评估指标等。
* **多重评估指标**：支持在一次运行中评估多个指标。Scorer 会顺序执行，并且每个 scorer 都会以数据并行方式运行。
* **结构化输出**：每个 scorer 的输出会保存在单独的文件中，最终结果会合并为一个文件。

## 已实现的指标

该框架包含多个评估维度，分为两类：**Q (Question-only)** 和 **QA (Question-Answer)**。

* **QA (Question-Answer)**：综合评估问题 (`instruction`) 与答案 (`output`)。

  * **Output Token Length**

## 使用方法

1. **在 `configs/` 文件夹中配置 YAML 文件**：

   * 设置数据的 `input_path` 和 `output_path`。
   * 配置 `scorers`：

     * `name`：scorer 的名称。
     * `model`：在 scorer 中使用的模型。
     * 其他参数是 scorer 特有的。**请参考 [wiki page](https://opendataarena-tool.readthedocs.io/en/latest/model-based-evaluation/) 或 `configs/` 中的示例 YAML 文件了解更多细节。**

2. **准备数据**：

   * 在 `config.yaml` 中配置 `input_file` 为一个 `jsonl` 文件。
   * 确保文件格式符合 [`example_input_add_key.jsonl`](data_process/example_input_add_key.jsonl)。

3. **运行评估**：

   * 运行以下命令：

     ```bash
     bash sh/Length.sh
     ```

     你也可以通过运行对应的 shell 脚本使用其他 scorer。
   * 输出将保存在 YAML 中指定的 `output_path` 下，包括以下文件：

     * `temp/`：临时文件，包括拆分后的数据和每个 scorer 的分数。分数格式如下：

       ```json
       {
         "id": 0,
         "Deita_Quality": 2.23
       }
       ```
     * `output.jsonl`：最终评估输出，分数会合并到原始数据中。例如：

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