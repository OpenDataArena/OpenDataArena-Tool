# OpenDataArena Data Scoring Toolkit

<p align="center">
  English | <a href="./README_zh-CN.md">简体中文</a>
</p>

## Introduction

The data scorer of [OpenDataArena-Tool](https://github.com/OpenDataArena/OpenDataArena-Tool) for [OpenDataArena](https://opendataarena.github.io/) offers multi-dimensional score assessments for datasets through a series of automated, multi-faceted scoring and processing methods.

## Wiki Documentation
More details about the data scoring can be found in [OpenDataArena-Tool Data Scorer Documentation](https://opendataarena-tool.readthedocs.io/en/latest/).

## Core Modules

This project integrates various advanced data processing and scoring technologies, primarily including the following three core modules. Each metric evaluates Q (instruction), QA (instruction + output), or both, as specified below.

* 📊 **Model-based Scorer**: leveraging internal model signals to assess data.
  * Deita Complexity (Q)
  * Thinking Probability (Q)
  * Deita Quality (QA)
  * Instruction Following Difficulty (IFD) (QA)
  * Reward Model (QA)
  * Fail Rate (QA)

* ⚖️ **LLM-as-a-Judge Scorer**: leveraging powerful LLMs as "judgers" to simulate human judgment in scoring the data.
  * Difficulty (Q)
  * Relevance (QA)
  * Clarity (Q & QA)
  * Coherence (Q & QA)
  * Completeness (Q & QA)
  * Complexity (Q & QA) 
  * Correctness (Q & QA)
  * Meaningfulness (Q & QA)

* 🧠 **Heuristic Scorer**: heuristic methods to score the data.
  * Response Length (QA)

## Installation

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

## How to Use

To begin, ensure your input data adheres to the expected format.

### Data Format

Your original input data should primarily consist of two keys: `instruction` and `output`, and **each line must be a valid JSON object**. This means your file should be in **JSONL format**.

**Example:** (You can also refer to `data_process/example_input.jsonl`)

```jsonl
{"instruction": "What is the capital of France?", "output": "Paris"}
{"instruction": "Explain the concept of quantum entanglement.", "output": "Quantum entanglement is a phenomenon where two or more particles become linked in such a way that they share the same fate, regardless of the distance between them. Measuring the state of one entangled particle instantaneously influences the state of the other(s)."}
{"instruction": "List three benefits of regular exercise.", "output": "Regular exercise improves cardiovascular health, boosts mood and reduces stress, and strengthens muscles and bones."}
```

**Important Note:**
  * If your original data contains an `input` key (common in formats like Alpaca), you must concatenate the `input` value with the `instruction` value, using a `\n` as a separator.
  * If you use `FailRateScorer`, you must add a `answer` key to your data, which is the correct answer to the problem. Please refer to `data_process/example_input_w_answer.jsonl` for an example.
  * If you use scorers only for Q (instruction), you can set the value of `output` to `None`. Please refer to `data_process/example_input_wo_output.jsonl` for an example.


### Running Data Scoring Scripts

This project adopts a modular structure, with each core module serving as an independent subdirectory. For detailed instructions on running specific scorers, **please refer to the `README.md` file within the corresponding subdirectory.**

### Post-processing - Score Normalization

In order to ensure fair comparison and aggregation across different scoring dimensions, normalization is performed to scale all scoring metrics to a common [0, 1] range. This is especially important when combining scores with different original ranges. Metrics already in `[0, 1]` range are **not** normalized.

#### Usage
```bash
python data_process/normalize_scores.py --input_file <your_input_path> --output_file <your_output_path>
```
