# OpenDataArena 数据评分工具

<p align="center">
  <a href="./README.md">English</a> | 简体中文
</p>

## Introduction

[OpenDataArena-Tool](https://github.com/OpenDataArena/OpenDataArena-Tool) 中的数据评分工具通过一系列自动化、多方面的评分和处理方法，为 [OpenDataArena](https://opendataarena.github.io/) 提供了多维度的评估。

## Wiki 文档
更多关于数据评分的详细信息，请参阅 [OpenDataArena-Tool 数据评分文档](https://opendataarena-tool.readthedocs.io/en/latest/)。

## 核心模块

本项目集成了各种先进的数据处理和评分技术，主要包括以下三个核心模块。每个指标评估 Q（指令）、QA（指令 + 输出）或两者，具体如下。

* 📊 **基于模型的评分器**: 利用模型的内部信号评估数据。
  * Deita Complexity (Q)
  * Thinking Probability (Q)
  * Deita Quality (QA)
  * Instruction Following Difficulty (IFD) (QA)
  * Reward Model (QA)
  * Fail Rate (QA)

* ⚖️ **LLM-as-a-Judge 评分器**: 利用强大的 LLM 作为 "法官" ，通过模拟人类的判断来评分数据。
  * Difficulty (Q)
  * Relevance (QA)
  * Clarity (Q & QA)
  * Coherence (Q & QA)
  * Completeness (Q & QA)
  * Complexity (Q & QA) 
  * Correctness (Q & QA)
  * Meaningfulness (Q & QA)

* 🧠 **启发式评分器**: 使用启发式方法评分数据。
  * Response Length (QA)

## 安装

```bash
conda create -n oda python=3.10 -y
conda activate oda
git clone https://github.com/OpenDataArena/OpenDataArena-Tool.git
cd OpenDataArena-Tool/data_scorer
pip install -r requirements.txt
pip install flash_attn==2.7.4.post1 --no-build-isolation
# if you want to calculate fail rate, run the following command, which will install the lighteval package
cd model_based/fail_rate
pip install -e .[dev]
```

## 如何使用

首先，确保您的输入数据符合预期格式。

### 数据格式

您的原始输入数据应主要包含两个键：`instruction` 和 `output`，**每行必须是一个有效的 JSON 对象**。这意味着您的文件应为 **JSONL 格式**。

**示例:** (您也可以参考 `data_process/example_input.jsonl`)

```jsonl
{"instruction": "What is the capital of France?", "output": "Paris"}
{"instruction": "Explain the concept of quantum entanglement.", "output": "Quantum entanglement is a phenomenon where two or more particles become linked in such a way that they share the same fate, regardless of the distance between them. Measuring the state of one entangled particle instantaneously influences the state of the other(s)."}
{"instruction": "List three benefits of regular exercise.", "output": "Regular exercise improves cardiovascular health, boosts mood and reduces stress, and strengthens muscles and bones."}
```

**重要提示:**
  * 如果您的原始数据包含 `input` 键（在 Alpaca 格式中很常见），您必须将 `input` 值与 `instruction` 值连接起来，使用 `\n` 作为分隔符。
  * 如果您使用 `FailRateScorer`，您必须在数据中添加 `answer` 键，作为问题的正确答案。请参考 `data_process/example_input_w_answer.jsonl` 的示例。
  * 如果您只使用评分器评估 Q（指令），您可以将 `output` 的值设置为 `None`。请参考 `data_process/example_input_wo_output.jsonl` 的示例。


### 运行数据评分脚本

本项目采用模块化结构，每个核心模块作为独立的子目录。有关运行特定评分器的详细说明，**请参考相应子目录中的 `README.md` 文件。**

### 后处理 - 评分归一化

为了确保公平比较和跨不同评分维度的聚合，对所有评分指标进行归一化，将它们缩放到 [0, 1] 范围内。这在组合不同原始范围的评分时尤其重要。已经处于 `[0, 1]` 范围内的指标**不会**进行归一化。

#### 使用方法
```bash
python data_process/normalize_scores.py --input_file <your_input_path> --output_file <your_output_path>
```
