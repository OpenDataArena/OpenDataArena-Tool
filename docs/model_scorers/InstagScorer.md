# InstagScorer

## Overview

The **InsTag Scorer** (Instruction Tagging Scorer) is a model-based evaluation tool designed to measure the *complexity* of instructions in supervised fine-tuning (SFT) datasets. Proposed in the paper [Lu et al., 2023](https://arxiv.org/abs/2308.07074), InsTag provides a fine-grained semantic and intention-based approach to analyzing user queries by identifying and tagging diverse instruction characteristics.

InsTag addresses the challenge of quantifying instruction diversity and complexity—two critical factors for successful SFT datasets. By leveraging a specialized tagging model, this scorer automatically identifies semantic tags and user intentions within instructions, providing an objective measure of instruction complexity based on the number and variety of identified tags.

## Metric Definition:

* **Definition:** 

  The InsTag Complexity score is calculated as the **number of semantic and intention tags** identified in a given instruction.

* **Explanation:** 
  
  The scorer prompts a fine-tuned language model to analyze the instruction and generate a JSON output containing identified tags with explanations. The complexity is quantified as:
  
  * **List output:** If the model returns a list of tag objects, the score equals the length of the list.
  * **Single dict output:** If the model returns a single tag dictionary, the score is 1.
  * **Parse failure:** If the output cannot be parsed as valid JSON, the score is 0.

* **Interpretation:**
  
  * A **higher InsTag Complexity score** indicates that the instruction contains multiple semantic dimensions or user intentions, suggesting **greater complexity and diversity**.
  * A **lower score** (0 or 1) suggests the instruction is **simple and uni-dimensional**.

## YAML Configuration

```yaml
name: InstagScorer
model: OFA-Sys/InsTagger
max_new_tokens: 512
batch_size: 8
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"InstagScorer"` | Identifier for the scorer |
| `model` | string | `"OFA-Sys/InsTagger"` | Path to the local model or Hugging Face model identifier for instruction tagging |
| `max_new_tokens` | integer | `512` | Maximum number of new tokens to generate for tag outputs (valid range: 1-2047) |
| `batch_size` | integer | `8` | Number of samples to process in parallel per forward pass |


## Underlying Model

The scorer uses [**OFA-Sys/InsTagger**](https://huggingface.co/OFA-Sys/InsTagger), a causal language model specifically fine-tuned for instruction tagging tasks. The model is trained to identify and categorize fine-grained semantic tags and user intentions across a comprehensive taxonomy of 6.6K tags covering diverse query types.

## Scoring Process

1. **Prompt Construction**: For each instruction, the scorer constructs a structured prompt following a chat template:
   ```
   A chat between a curious user and an artificial intelligence assistant. 
   The assistant gives helpful, detailed, and polite answers to the human's questions.
   
   USER: Please identify tags of user intentions in the following user query and 
   provide an explanation for each tag. Please respond in the JSON format 
   {"tag": str, "explanation": str}.
   User query: [INSTRUCTION]
   
   ASSISTANT:
   ```
   If the data item contains both `instruction` and `input` fields, they are concatenated as: `instruction + "\n" + input`.

2. **Batch Tokenization**: Instructions are tokenized with left-padding (required for causal language models). Input length is limited to `2048 - max_new_tokens` to reserve space for generation.

3. **Tag Generation**: The model generates tags in JSON format through greedy decoding with `max_new_tokens` parameter controlling output length.

4. **Score Calculation**: The generated JSON output is parsed to compute the complexity score:
   - **Valid JSON list:** `score = len(json_output)` (number of tags)
   - **Valid JSON dict:** `score = 1` (single tag)
   - **Invalid JSON:** `score = 0` (parsing failure)

## Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 3
}
```

- `id`: Unique identifier for the data sample (extracted from input data's `id` field)
- `score`: InsTag Complexity score representing the number of semantic/intention tags identified in the instruction (range: 0 to N, where higher values indicate more complex instructions)

## Citation

```bibtex
@article{lu2023instag,
  title={\# instag: Instruction tagging for analyzing supervised fine-tuning of large language models},
  author={Lu, Keming and Yuan, Hongyi and Yuan, Zheng and Lin, Runji and Lin, Junyang and Tan, Chuanqi and Zhou, Chang and Zhou, Jingren},
  journal={arXiv preprint arXiv:2308.07074},
  year={2023}
}
```

