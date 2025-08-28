# OpenDataArena-Tool Data Scorer Documentation

The data scorer of [OpenDataArena-Tool](https://github.com/OpenDataArena/OpenDataArena-Tool) for [OpenDataArena](https://opendataarena.github.io/) offers multi-dimensional score assessments for datasets through a series of automated, multi-faceted evaluation and processing methods.

## Installation
```bash
conda create -n oda python=3.10
conda activate oda
git clone https://github.com/OpenDataArena/OpenDataArena-Tool.git
cd OpenDataArena/data_evaluation
pip install -r requirements.txt
pip install flash_attn==2.7.4.post1 --no-build-isolation
# if you want to calculate fail rate, run the following command, which will install the lighteval package
cd model_based/fail_rate
pip install -e .[dev]
```

## Data Evaluation
The data scorer of [OpenDataArena-Tool](https://github.com/OpenDataArena/OpenDataArena-Tool) integrates various advanced data processing and scoring technologies, primarily including the following three core modules. Each metric evaluates Q (instruction), QA (instruction + output), or both, as specified below.

* [Model-based Evaluation](model-based-evaluation)
    * [Deita Complexity](model-based-evaluation#deita-complexity) (Q)
    * [Thinking Probability](model-based-evaluation#thinking-probability) (Q)
    * [Deita Quality](model-based-evaluation#deita-quality) (QA)
    * [Instruction Following Difficulty (IFD)](model-based-evaluation#instruction-following-difficulty) (QA)
    * [Reward Model](model-based-evaluation#reward-model) (QA)
    * [Fail Rate](model-based-evaluation#fail-rate) (QA)
* [LLM-as-Judge](llm-as-judge)
    * [Difficulty](llm-as-judge#difficulty) (Q)
    * [Relevance](llm-as-judge#relevance) (QA)
    * [Clarity](llm-as-judge#clarity) (Q & QA)
    * [Coherence](llm-as-judge#coherence) (Q & QA)
    * [Completeness](llm-as-judge#completeness) (Q & QA)
    * [Complexity](llm-as-judge#complexity) (Q & QA)
    * [Correctness](llm-as-judge#correctness) (Q & QA)
    * [Meaningfulness](llm-as-judge#meaningfulness) (Q & QA)
* [Heuristic](heuristic)
    * [Length](heuristic#length) (QA)
