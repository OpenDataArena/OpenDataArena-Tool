# OpenDataArena Data Scoring Toolkit

<p align="center">
  English | <a href="./README_zh-CN.md">简体中文</a>
</p>

## Introduction

The data scorer of [OpenDataArena-Tool](https://github.com/OpenDataArena/OpenDataArena-Tool) for [OpenDataArena](https://opendataarena.github.io/) offers multi-dimensional score assessments for datasets through a series of automated, multi-faceted scoring and processing methods.

## Wiki Documentation

More details about the data scoring can be found in [OpenDataArena-Tool Data Scorer Documentation](https://opendataarena-tool.readthedocs.io/en/latest/).

## Core Modules

This project integrates various advanced data processing and scoring technologies, primarily including the following three core modules:

* 📊 **Model-based Scorer**: leveraging internal model signals to assess data. This framework integrates nearly 40 model-based scorers, covering multiple dimensions including quality, complexity, gradient analysis, and more:
  * **Quality**: SkyworkRewardScorer, AtheneScorer, RMDeBERTaScorer, Gpt2HarmlessScorer, Gpt2HelpfulScorer, InfOrmScorer, DeitaQScorer, DebertaScorer, FinewebEduScorer, TextbookScorer, QuRateScorer, CleanlinessScorer, ProfessionalismScorer, ReadabilityScorer, ReasoningScorer, UniEvalD2tScorer, UniEvalDialogScorer, UniEvalFactScorer, UniEvalSumScorer
  * **Complexity**: DeitaCScorer, IFDScorer, ThinkingProbScorer, PPLScorer, NormLossScorer, UPDScorer
  * **Others**: GraNdScorer, NuclearNormScorer, EffectiveRankScorer, Task2VecScorer, MIWVScorer, SelectitTokenScorer, SelectitSentenceScorer, SelectitModelScorer, HESScorer, AnswerProbScorer, AskLlmScorer, FailRateScorer, InstagScorer

* ⚖️ **LLM-as-a-Judge Scorer**: leveraging powerful LLMs as "judges" to simulate human judgment in scoring the data.  
  In this framework, commonly used dimensions include Q, A, and QA:
  * **Q**: Evaluates the "Question/Instruction" itself.
  * **A**: Evaluates the "Answer/Generated Content" itself.
  * **QA**: Evaluates the overall quality of the "Question-Answer Pair" (such as the relevance of the answer to the question).
  
  Currently built-in metrics include:
  * Difficulty (Q): The difficulty of the question
  * Relevance (QA): The relevance of the answer to the question
  * Clarity (Q & QA): Clarity of expression
  * Coherence (Q & QA): Content coherence
  * Completeness (Q & QA): Information completeness
  * Complexity (Q & QA): Level of complexity
  * Correctness (Q & QA): Content correctness
  * Meaningfulness (Q & QA): Meaningfulness/Value

* 🧠 **Heuristic Scorer**: using heuristic methods to score the data. This framework integrates 23 heuristic scorers, covering multiple dimensions including diversity, statistical features, content detection, and more:
  * **Diversity**: VendiScorer, KNNScorer, ApsScorer, ApjsScorer, RadiusScorer, ClusterInertiaScorer, PartitionEntropyScorer, NovelSumScorer, FacilityLocationScorer, UniqueNgramScorer, UniqueNtokenScorer, MtldScorer, VocdDScorer, TokenEntropyScorer, GramEntropyScorer, HddScorer
  * **Statistical Features**: TokenLengthScorer, StrLengthScorer, TreeInstructScorer, LogDetDistanceScorer
  * **Content Detection**: ThinkOrNotScorer, PureThinkScorer, TsPythonScorer

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
* Some scorers may require additional fields or special format requirements. Please be sure to consult the corresponding scorer's Wiki or README for specific descriptions of required fields/formats.

### Running Data Scoring Scripts

This project adopts a modular structure, with each core module serving as an independent subdirectory. For detailed instructions on running specific scorers, **please refer to the `README.md` file within the corresponding subdirectory.**

### Post-processing - Score Normalization

In order to ensure fair comparison and aggregation across different scoring dimensions, normalization is performed to scale all scoring metrics to a common [0, 1] range. This is especially important when combining scores with different original ranges. Metrics already in `[0, 1]` range are **not** normalized.

#### Usage

```bash
python data_process/normalize_scores.py --input_file <your_input_path> --output_file <your_output_path>
```
