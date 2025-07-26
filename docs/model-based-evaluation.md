# Model-based Evaluation

Model-based evaluation methods leverage specific models to assess the difficulty, quality, or instruction-following capability of SFT data.

## Deita Complexity (Q)

### Overview
The **Deita Complexity Scorer** is a model-based evaluation tool designed to estimate the *instruction complexity* of SFT data. Proposed in the paper [Liu et al., 2024](https://arxiv.org/abs/2312.15685), this method aims to measure how cognitively demanding an instruction is for a model to execute. Rather than relying on shallow heuristics, the Deita Complexity Scorer provides a learning-based, instruction-only metric that correlates with downstream performance and instruction-following capabilities.

### Metric Definition: **Deita Complexity**

* **Definition:**
  
    1. First generate variations of each instruction with increasing difficulty using the In-Depth Evolving Prompt method.
    2. Collect these data-score pairs to train a LLM as a complexity scorer.
    3. The trained scorer is used to predict complexity scores (1-6) for new instructions.

* **Explanation:** Intuitively, the complexity score estimates how *unexpected or difficult* an instruction is to follow for the SFT model.

  * A **higher Deita Complexity score** imply that the SFT model struggles with the instruction relative to the reference model, indicating **greater complexity**.
  * A **lower Deita Complexity score** suggest that the instruction is easy to complete and consistent with the SFT model’s learned behaviors.

### YAML Configuration
```yaml
name: DeitaCScorer
model: hkust-nlp/deita-complexity-scorer
max_length: 2048
batch_size: 32
```
### Underlying Model

The scorer uses [hkust-nlp/deita-complexity-scorer](https://huggingface.co/hkust-nlp/deita-complexity-scorer), which is introduced in [Liu et al., 2024](https://arxiv.org/abs/2312.15685).

### Citation

```bibtex
@article{liu2023makes,
  title={What makes good data for alignment? a comprehensive study of automatic data selection in instruction tuning},
  author={Liu, Wei and Zeng, Weihao and He, Keqing and Jiang, Yong and He, Junxian},
  journal={arXiv preprint arXiv:2312.15685},
  year={2023}
}
```

## Thinking Probability (Q)

### Overview

`ThinkingProbScorer` is a scoring module built to quantify the *necessity of deep reasoning* for a given **math problem**, using a model RL-trained with the [AdaptThink](https://github.com/THU-KEG/AdaptThink) framework. This scorer is designed to estimate the probability that the AdaptThink model would *engage in explicit thinking* ("Thinking" mode) versus *directly providing a solution* ("NoThinking" mode), based on the perceived difficulty of the problem.

### Metric Definition: **Thinking\_Prob**

* **Definition:** `Thinking_Prob = 1 - P(</think>)`
* **Explanation:** This metric estimates the *difficulty* of a problem by measuring how unlikely the model is to immediately output the `</think>` token (i.e., to choose NoThinking mode).
  * A **higher value** (closer to 1) indicates that the model is **less likely to skip thinking**, suggesting the problem is *hard*.
  * A **lower value** (closer to 0) indicates the model would confidently produce a final answer **without any thinking**, suggesting the problem is *simple*.

### Underlying Model

* The scorer uses [THU-KEG/AdaptThink-7B-delta0.05](https://huggingface.co/THU-KEG/AdaptThink-7B-delta0.05), a language model trained with reinforcement learning to **adaptively choose** between:

  * **Thinking Mode:** `[thinking process]</think>[final solution]`
  * **NoThinking Mode:** `</think>[final solution]`

### Scoring Process

1. **Math problems** are passed through the tokenizer with default chat templates.
2. The model is instructed to generate only **one token**, and the probability of generating `</think>` as the first token is extracted from the logprobs.
3. The metric `Thinking_Prob` is computed as `1 - P(</think>)`, interpreting it as **problem difficulty**.

### YAML Configuration

```yaml
name: ThinkingProbScorer # The name of the scorer
model: THU-KEG/AdaptThink-7B-delta0.05 # The model to use. You can use the model from HuggingFace or your local directory
batch_size: 128 # The batch size to use
```

### Citation

```bibtex
@article{zhang2025adaptthink,
  title={Adaptthink: Reasoning models can learn when to think},
  author={Zhang, Jiajie and Lin, Nianyi and Hou, Lei and Feng, Ling and Li, Juanzi},
  journal={arXiv preprint arXiv:2505.13417},
  year={2025}
}
```

## Deita Quality (QA)
### Overview
The **Deita Quality Scorer** is a model-based evaluation tool designed to estimate the data quality of SFT data. Proposed in the paper [Liu et al., 2024](https://arxiv.org/abs/2312.15685).

### Metric Definition: **Deita Quality**

* **Definition:**
  
  1. First generate different quality variants of the same data using the In-Depth Evolving Prompt method.
  2. Collect these data-score pairs to train a LLM as a quality scorer.
  3. The trained scorer is used to predict quality  scores(1-6) for other sft data samples.

* **Explanation:** Explanation: Intuitively, the Deita Quality score estimates the overall quality of an SFT sample.
  
  * A **higher Deita Quality score** imply that implies that the response presents data in a clear, accurate, and meaningful way.
  * A **lower Deita Quality score** suggests that the response is vague, misleading, or poorly organized in terms of data content.


### YAML Configuration
```yaml
# Deita Quality Configuration

# Scorer name
name: DeitaQScorer
model: hkust-nlp/deita-complexity-scorer
max_length: 2048
batch_size: 32
```
### Underlying Model

The scorer uses [hkust-nlp/deita-complexity-scorer](https://huggingface.co/hkust-nlp/deita-qulity-scorer), which is introduced in [Liu et al., 2024](https://arxiv.org/abs/2312.15685).

### Citation

```bibtex
@article{liu2023makes,
  title={What makes good data for alignment? a comprehensive study of automatic data selection in instruction tuning},
  author={Liu, Wei and Zeng, Weihao and He, Keqing and Jiang, Yong and He, Junxian},
  journal={arXiv preprint arXiv:2312.15685},
  year={2023}
}
```

## Instruction Following Difficulty (IFD) (QA)

### Overview

Instruction Following Difficulty (IFD) is a metric introduced to quantify the complexity of instruction following in large language models (LLMs). It compares the model's ability to generate responses with and without the instructional context, identifying discrepancies between expected and generated outputs. Higher IFD scores indicate more difficulty in aligning responses with the provided instructions, which in turn indicates the difficulty of an instruction.

### Metric Definition: **Instruction-Following Difficulty (IFD)**

* **Definition:** IFD(Q,A) = s(A|Q) / s(A)
* **Explanation:** This metric compares the **conditioned answer score** (the model's ability to generate a response when the instruction is provided) with the **direct answer score** (the model's ability to generate the answer without instruction). The ratio between these scores indicates how much the instruction helps in generating the answer:

  * A **higher IFD score** suggests the inability of the model to align responses to the given corresponding instructions, indicating **higher difficulty**.
  * A **lower IFD score** indicates the instruction provides **significant guidance** for response generation, suggesting the instruction-answer pair is **easier to follow**.

### Underlying Model

The scorer uses [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct), an instruction-tuned language model to predict:

  * **Direct Answer Score:** s(A)
  * **Conditioned Answer Score:** s(A|Q)

### Scoring Process

1. **Instruction-Answer Pair Processing:** Each sample containing an instruction `Q` and corresponding answer `A` is tokenized using the model's default tokenization scheme.

2. **Direct Answer Score Calculation:** The model generates the answer `A` **without any instructional context**, computing the cross-entropy loss across all answer tokens. This measures the model's **inherent ability** to generate the answer independently.

3. **Conditioned Answer Score Calculation:** The model generates the same answer `A` **given the full instruction context** `Q`, computing the cross-entropy loss across all answer tokens conditioned on instruction `Q`. This measures how well the model can generate the answer **when provided with instructions**.

4. **IFD Score Computation:** The final IFD metric is calculated as the ratio `IFD(Q,A) = s(A|Q) / s(A)`, effectively measuring how much the instruction **helps or hinders** the answer generation process compared to generating the answer alone.


### YAML Configuration

```yaml
# IFD Configuration

# Scorer name
name: IFDScorer
# Model name
model: Qwen/Qwen2.5-3B-Instruct
# Max Token Length
max_length: 2048
```

### Citation

```bibtex
@inproceedings{li2024quantity,
  title={From Quantity to Quality: Boosting LLM Performance with Self-Guided Data Selection for Instruction Tuning},
  author={Li, Ming and Zhang, Yong and Li, Zhitao and Chen, Jiuhai and Chen, Lichang and Cheng, Ning and Wang, Jianzong and Zhou, Tianyi and Xiao, Jing},
  booktitle={Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)},
  pages={7595--7628},
  year={2024}
}
```

## Reward Model (QA)

### Overview
We leverage Skywork Reward Model, a large-scale reward model, trained on 26 million high-quality preference pairs, designed to assess the alignment quality of supervised fine-tuning (SFT) data. Unlike heuristic or synthetic scoring strategies, the Skywork Reward Model is grounded in extensive human-LLM joint evaluations and sets a new standard for reward modeling. It is suitable for ranking, filtering, or curating SFT data for alignment training.

### Metric Definition: **Reward Model**

* **Definition:** Given an instruction-response pair, the reward scorer assigns a scalar reward score, representing how preferable or aligned the response is in context of the instruction.
  * A higher Skywork Reward Score indicates that the response is preferred by the reward model.
  * A lower score suggests deficiencies in quality, alignment, or task-following behavior.
    
### Underlying Model

We use the the largest model from the Skywork-Reward-V2 series, [Skywork/Skywork-Reward-V2-Llama-3.1-8B-40M](https://huggingface.co/Skywork/Skywork-Reward-V2-Llama-3.1-8B-40M). The model are trained on a vast corpus of human-LLM preference data, achieving state-of-the-art performance across benchmarks such as: RewardBench v1 & v2, RMB, RM-Bench, and JudgeBench.

### YAML Configuration 

```yaml
name: SkyworkRewardScorer
model: Skywork/Skywork-Reward-V2-Llama-3.1-8B-40M
max_length: 4096
batch_size: 16
```

### Citation

```bibtex
@article{liu2025skywork,
  title={Skywork-Reward-V2: Scaling Preference Data Curation via Human-AI Synergy},
  author = {Liu, Chris Yuhao and Zeng, Liang and Xiao, Yuzhen and He, Jujie and Liu, Jiacai and Wang, Chaojie and Yan, Rui and Shen, Wei and Zhang, Fuxiang and Xu, Jiacheng and Liu, Yang and Zhou, Yahui},
  journal={arXiv preprint arXiv:2507.01352},
  year={2025}
}
```

## Fail Rate (QA)

### Overview

`FailRateScorer` is a comprehensive evaluation framework designed to quantify the **failure rate** of **mathematical problems** by leveraging strong language models to estimate problem difficulty through multi-sample inference. This pipeline calculates the probability that a model will fail to solve a specific mathematical problem, providing an objective measure of problem complexity.

### Metric Definition: **Fail_Rate**

* **Definition:** `Fail_Rate = 1 - sample_n_pass@1`
* **Explanation:** This metric estimates the **difficulty** of a problem by measuring the probability that a model gives the correct answer across multiple attempts.
  * A **higher value** (closer to 1) indicates the model is **more likely to fail**, suggesting the problem is **difficult**.
  * A **lower value** (closer to 0) indicates the model can **consistently provide correct answers**, suggesting the problem is **simple**.

### Pipeline Architecture

The evaluation pipeline is implemented using Hugging Face's [**LightEval**](https://github.com/huggingface/lighteval) framework, which provides a robust and scalable evaluation infrastructure. The pipeline uses configurable language models (e.g., `Qwen/Qwen3-8B`) and operates through the following process:

1. **Task Generation:** Custom evaluation tasks are dynamically created for each split
2. **Parallel Evaluation:** Each split is evaluated on separate GPUs using the LightEval framework
3. **Result Aggregation:** Results are collected and merged back into the original dataset format

### YAML Configuration

```yaml
name: FailRateScorer # The name of the scorer
model: Qwen/Qwen3-8B # The model to use. You can use the model from HuggingFace or your local directory
metrics_sample_size: 4  # 1, 4, 8, 16, 32, 64
generation_size: 4096
```

### Citation

```bibtex
@misc{lighteval,
  author = {Habib, Nathan and Fourrier, Clémentine and Kydlíček, Hynek and Wolf, Thomas and Tunstall, Lewis},
  title = {LightEval: A lightweight framework for LLM evaluation},
  year = {2023},
  version = {0.8.0},
  url = {https://github.com/huggingface/lighteval}
}
```


