# InfOrmScorer

## Overview

The **INF Outcome Reward Model (INF-ORM) Scorer** is a state-of-the-art reward model-based evaluation tool designed to assess the alignment quality and preference of supervised fine-tuning (SFT) data. Built upon the Llama-3.1-70B-Instruct architecture and trained with the INF-ORM-Preference-Magnitude-80K dataset, this scorer leverages advanced techniques including scaled Bradley-Terry loss, modified score head architecture, and model merging to achieve top-tier performance.

As of December 2024, INF-ORM-Llama3.1-70B ranks **first** on the [RewardBench leaderboard](https://huggingface.co/spaces/allenai/reward-bench) with a score of 95.1, demonstrating exceptional capability in evaluating chat responses, safety, and reasoning tasks.

Unlike traditional heuristic or synthetic scoring methods, the INF-ORM Scorer provides a learning-based evaluation that captures nuanced preferences in instruction-response pairs, making it particularly suitable for data curation, quality assessment, and alignment training in large language models.

## Metric Definition:

* **Definition:** 

  Given an instruction-response pair, the INF-ORM Scorer assigns a scalar reward score representing the overall quality and alignment of the response in the context of the given instruction.

* **Explanation:** 

  The reward score reflects the model's learned preferences from large-scale human-annotated preference data:
  
  * A **higher INF-ORM score** indicates that the response is well-aligned, helpful, accurate, and preferred according to human evaluation standards.
  * A **lower INF-ORM score** suggests deficiencies in quality, coherence, safety, or instruction-following behavior.

## YAML Configuration

```yaml
name: InfOrmScorer
model: infly/INF-ORM-Llama3.1-70B
batch_size: 32
max_length: 4096
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"InfOrmScorer"` | Identifier for the scorer |
| `model` | string | `"infly/INF-ORM-Llama3.1-70B"` | HuggingFace model path or local path to the reward model checkpoint |
| `batch_size` | integer | `32` | Number of samples to process in parallel during evaluation |
| `max_length` | integer | `4096` | Maximum token length for input sequences (sequences exceeding this will be truncated) |

## Underlying Model

The scorer uses [**infly/INF-ORM-Llama3.1-70B**](https://huggingface.co/infly/INF-ORM-Llama3.1-70B), a 70-billion parameter reward model that ranks **first** on the RewardBench leaderboard (December 2024) with a score of 95.1. 

**Key Features:**
- **Base Architecture:** Llama-3.1-70B-Instruct with a modified two-layer MLP score head (Linear → ReLU → Linear)
- **Training Data:** INF-ORM-Preference-Magnitude-80K dataset with magnitude annotations (1-3 scale)
- **Training Method:** Scaled Bradley-Terry loss with magnitude weighting and model merging techniques

**Performance Benchmarks (RewardBench, December 2024):**

| Rank | Model | Score | Chat | Chat Hard | Safety | Reasoning |
|------|-------|-------|------|-----------|--------|-----------|
| **1** | **INF-ORM-Llama3.1-70B** | **95.1** | **96.6** | **91.0** | **93.6** | **99.1** |
| 2 | Skywork-Reward-Gemma-2-27B-v0.2 | 94.3 | 96.1 | 89.9 | 93.0 | 98.1 |
| 3 | Llama-3.1-Nemotron-70B-Reward | 94.1 | 97.5 | 85.7 | 95.1 | 98.1 |

## Scoring Process

1. **Input Formatting**: For each data sample containing `instruction`, optional `input`, and `output` fields, the scorer constructs a conversation in chat template format:
   ```python
   [
       {"role": "user", "content": "<instruction> + <input>"},
       {"role": "assistant", "content": "<output>"}
   ]
   ```

2. **Tokenization**: Conversations are tokenized using the model's chat template with `apply_chat_template()`. Sequences exceeding `max_length` are truncated with a warning.

3. **Batch Processing**: Multiple samples are processed simultaneously based on `batch_size` configuration for efficient GPU utilization.

4. **Model Inference**: The model processes tokenized inputs through:
   - Llama-3.1-70B transformer layers to extract contextual representations
   - Modified two-layer MLP score head (Linear → ReLU → Linear) to produce logits
   - Extraction of the final sequence position's logit as the reward score

5. **Score Extraction**: Raw logits are converted to float values and returned as reward scores for each sample.

## Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 4.96875
}
```

- `id`: Unique identifier for the data sample (preserved from input dataset's `id` field)
- `score`: The reward score assigned by the INF-ORM model (higher values indicate better quality and alignment)

## Citation

```bibtex
@misc{INF-ORM-Llama3.1-70B, 
      url={https://huggingface.co/infly/INF-ORM-Llama3.1-70B},
      title={INF-ORM-Llama3.1-70B},
      year={2024},
      author={Minghao Yang, Chao Qu, Xiaoyu Tan}
}
```

