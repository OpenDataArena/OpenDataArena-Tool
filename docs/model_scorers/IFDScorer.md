# IFDScorer

## Overview

Instruction Following Difficulty (IFD) is a metric introduced to quantify the complexity of instruction following in large language models (LLMs). It compares the model's perplexity when generating outputs with and without instructional context. By calculating the ratio of conditional perplexity to direct perplexity, IFD measures how much an instruction affects the difficulty of generating the corresponding output. Higher IFD scores (> 1) indicate that the instruction increases generation difficulty, suggesting the instruction-output pair is harder to follow or poorly aligned.

## Metric Definition:

* **Definition:** 

  Given an instruction Q and answer/output A, the IFD score is computed as:
  
  **IFD(Q,A) = perplexity(A|Q) / perplexity(A)**
  
  Where:
  - `perplexity(A)` = perplexity of generating answer A without any instruction context
  - `perplexity(A|Q)` = perplexity of generating answer A given instruction Q

* **Explanation:** This metric compares the **conditional perplexity** (model's difficulty in generating the output when instruction is provided) with the **direct perplexity** (model's difficulty in generating the output without instruction). The ratio indicates how much the instruction affects answer generation:

  * A **higher IFD score (> 1)** suggests the model has **more difficulty** generating the answer when given the instruction, indicating the instruction-answer pair is **harder to follow** or poorly aligned.
  * A **lower IFD score (< 1)** indicates the instruction provides **guidance that reduces perplexity**, suggesting the instruction-answer pair is **easier to follow** and well-aligned.
  * An **IFD score ≈ 1** suggests the instruction provides **minimal effect** on the answer generation difficulty.

## YAML Configuration

```yaml
name: IFDScorer
model: Qwen/Qwen2.5-3B-Instruct
max_length: 2048
batch_size: 1
template: "<|im_start|>user\n{instruction}\n{input}<|im_end|>\n<|im_start|>assistant\n"
template_no_input: "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"IFDScorer"` | Identifier for the scorer |
| `model` | string | `"openai-community/gpt2"` | HuggingFace model path or local directory for the language model (fallback: `openai-community/gpt2`) |
| `max_length` | integer | `2048` | Maximum sequence length for tokenization |
| `batch_size` | integer | `1` | Number of samples to process in parallel per forward pass |
| `template` | string | See above | Template for formatting instruction with input field |
| `template_no_input` | string | See above | Template for formatting instruction without input field |


## Underlying Model

The scorer can use any causal language model from HuggingFace (e.g., [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)) to compute:

  * **Direct Answer Perplexity:** s(A) - perplexity of generating the answer without instruction
  * **Conditioned Answer Perplexity:** s(A|Q) - perplexity of generating the answer given instruction

## Scoring Process

1. **Input Processing**: For each data sample, the scorer extracts:
   - Instruction (from `instruction` field)
   - Input (from `input` field if present)
   - Output/Answer (from `output` field)

2. **Prompt Formatting**: Construct the prompt using the configured template:
   - If `input` is provided: use `template` with both instruction and input
   - If `input` is empty: use `template_no_input` with only instruction

3. **Direct Answer Perplexity Calculation**: Compute perplexity for the output `A` **without any instructional context** by:
   - Tokenizing the output alone
   - Computing cross-entropy loss for all output tokens
   - Calculating perplexity as `exp(loss)`
   - This measures the model's **inherent ability** to generate the output independently

4. **Conditioned Answer Perplexity Calculation**: Compute perplexity for the output `A` **given the full instruction context** by:
   - Concatenating prompt and output: `prompt + output`
   - Masking prompt tokens (setting labels to -100)
   - Computing cross-entropy loss only for output tokens
   - Calculating perplexity as `exp(loss)`
   - This measures how well the model can generate the output **when provided with instructions**

5. **IFD Score Computation**: The final IFD metric is calculated as the perplexity ratio:
   - `IFD(Q,A) = perplexity(A|Q) / perplexity(A)`
   - Higher ratio indicates the instruction **hinders** answer generation (more difficult)
   - Lower ratio indicates the instruction **helps** answer generation (easier)

## Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 1.25
}
```

- `id`: Unique identifier for the input sample (from the input data's `id` field)
- `score`: IFD score calculated as `perplexity(A|Q) / perplexity(A)`

## Citation

```bibtex
@inproceedings{li2024quantity,
  title={From Quantity to Quality: Boosting LLM Performance with Self-Guided Data Selection for Instruction Tuning},
  author={Li, Ming and Zhang, Yong and Li, Zhitao and Chen, Jiuhai and Chen, Lichang and Cheng, Ning and Wang, Jianzong and Zhou, Tianyi and Xiao, Jing},
  booktitle={Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)},
  pages={7595--7628},
  year={2024}
}
```
