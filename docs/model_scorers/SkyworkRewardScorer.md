# SkyworkRewardScorer

## Overview

The **Skywork Reward Scorer** is a model-based evaluation tool that leverages the Skywork Reward Model, a large-scale reward model trained on 26 million high-quality preference pairs, designed to assess the alignment quality of supervised fine-tuning (SFT) data. Unlike heuristic or synthetic scoring strategies, the Skywork Reward Model is grounded in extensive human-LLM joint evaluations and sets a new standard for reward modeling. It is suitable for ranking, filtering, or curating SFT data for alignment training.

## Metric Definition:

* **Definition:** 

  Given an instruction-response pair, the reward scorer assigns a scalar reward score, representing how preferable or aligned the response is in the context of the instruction.

* **Explanation:**
  
  * A **higher Skywork Reward Score** indicates that the response is preferred by the reward model, demonstrating better quality, alignment, and task-following behavior.
  * A **lower score** suggests deficiencies in quality, alignment, or task-following behavior.
  * The reward model provides a **unified preference signal** trained on extensive human feedback data, making it more reliable than heuristic metrics.

## YAML Configuration 

```yaml
name: SkyworkRewardScorer
model: Skywork/Skywork-Reward-V2-Llama-3.1-8B-40M
max_length: 4096
batch_size: 16
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"SkyworkRewardScorer"` | Identifier for the scorer |
| `model` | string | `"Skywork/Skywork-Reward-V2-Llama-3.1-8B-40M"` | HuggingFace model path for the reward model |
| `max_length` | integer | `4096` | Maximum sequence length for tokenization |
| `batch_size` | integer | `16` | Number of samples to process in parallel per forward pass |


## Underlying Model

The scorer uses the largest model from the Skywork-Reward-V2 series, [Skywork/Skywork-Reward-V2-Llama-3.1-8B-40M](https://huggingface.co/Skywork/Skywork-Reward-V2-Llama-3.1-8B-40M). The model is trained on a vast corpus of human-LLM preference data, achieving state-of-the-art performance across benchmarks such as: RewardBench v1 & v2, RMB, RM-Bench, and JudgeBench.

## Scoring Process

1. **Input Processing**: For each data sample, the scorer extracts:
   - Instruction (from `instruction` and optional `input` fields)
   - Response (from `output` field)

2. **Prompt Construction**: The instruction and response are formatted according to the Skywork Reward Model's chat template

3. **Forward Pass**: The formatted conversation is passed through the reward model

4. **Score Extraction**: The model outputs a scalar reward score representing the preference strength for the response given the instruction

## Output Format

For each input sample, the scorer returns:

```json
{
  "id": "sample_identifier",
  "score": 2.5678
}
```

- `id`: Unique identifier for the sample
- `score`: Reward score assigned by the model (higher values indicate better alignment and quality)

## Citation

```bibtex
@article{liu2025skywork,
  title={Skywork-Reward-V2: Scaling Preference Data Curation via Human-AI Synergy},
  author = {Liu, Chris Yuhao and Zeng, Liang and Xiao, Yuzhen and He, Jujie and Liu, Jiacai and Wang, Chaojie and Yan, Rui and Shen, Wei and Zhang, Fuxiang and Xu, Jiacheng and Liu, Yang and Zhou, Yahui},
  journal={arXiv preprint arXiv:2507.01352},
  year={2025}
}
```
