# PPLScorer

## Overview

The **PPL (Perplexity) Scorer** is a model-based evaluation tool that measures how well a language model predicts a given text sequence. Perplexity is a fundamental metric in natural language processing that quantifies the uncertainty of a language model when generating text. A lower perplexity score indicates that the model finds the text more predictable and natural, while a higher score suggests the text is more surprising or difficult for the model to predict.

This scorer is particularly useful for assessing the quality and naturalness of SFT (Supervised Fine-Tuning) data, as it can identify samples that are either too simple (very low perplexity) or potentially noisy/anomalous (very high perplexity).

## Metric Definition:

* **Definition:** PPL = exp(L), where L is the average cross-entropy loss per token.
  
  Formally: PPL(x) = exp(-1/N × Σ log P(x_i | x_<i))
  
  where N is the number of tokens, and P(x_i | x_<i) is the probability of token x_i given all previous tokens.

* **Explanation:** Perplexity measures how "surprised" a language model is by a given text sequence.
  
  * A **lower PPL score** indicates the text is more predictable and natural according to the model, suggesting the sample follows common patterns and is of **higher quality**.
  * A **higher PPL score** suggests the text is less predictable, which could indicate either complex/diverse content or potentially **noisy or low-quality data**.

## YAML Configuration

```yaml
name: PPLScorer
model: meta-llama/Llama-3.1-8B
max_length: 2048
batch_size: 8
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"PPLScorer"` | Identifier for the scorer |
| `model` | string | **required** | HuggingFace model path or local path to a causal language model compatible with `AutoModelForCausalLM` (e.g., `meta-llama/Llama-3.1-8B`, `Qwen/Qwen2.5-7B`, `mistralai/Mistral-7B-v0.1`) |
| `max_length` | integer | `2048` | Maximum sequence length for tokenization. Sequences longer than this will be truncated |
| `batch_size` | integer | `8` | Number of samples to process simultaneously in each batch |

## Underlying Model

The PPL Scorer can work with **any causal language model** compatible with Hugging Face's `AutoModelForCausalLM` API. Common choices include:

* **Llama family**: `meta-llama/Llama-3.1-8B`, `meta-llama/Llama-2-7b-hf`
* **Qwen family**: `Qwen/Qwen2.5-7B`, `Qwen/Qwen2.5-14B`
* **Mistral family**: `mistralai/Mistral-7B-v0.1`
* **GPT-2 family**: `gpt2`, `gpt2-medium`, `gpt2-large`

The choice of model depends on your evaluation needs. Larger models generally provide more accurate perplexity estimates but require more computational resources.

## Scoring Process

The PPL Scorer follows this pipeline to evaluate each sample:

1. **Text Concatenation**: For each data item, the scorer concatenates the `instruction`, `input` (if present), and `output` fields with newlines, forming a complete text sequence.

2. **Tokenization with Padding**: The text is tokenized using the model's tokenizer with:
   * **Padding enabled**: Ensures all sequences in a batch have the same length
   * **Truncation enabled**: Limits sequences to `max_length`
   * **Pad token handling**: If the model lacks a pad token (e.g., Llama, Qwen), the EOS token is used as the pad token

3. **Label Preparation**: The scorer creates labels by:
   * Cloning the `input_ids` as labels
   * Replacing all padding token positions with `-100` to ensure they are ignored during loss calculation

4. **Loss Calculation**: For each sample individually:
   * The model computes the cross-entropy loss between predicted and actual tokens
   * Only non-padding tokens contribute to the loss (padding tokens with label `-100` are ignored)
   * The loss represents the average negative log-likelihood per valid token

5. **Perplexity Computation**: The final perplexity is calculated as: **PPL = exp(loss)**

6. **Batch Processing**: Samples are processed in batches according to the configured `batch_size` for efficiency.

## Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 12.45
}
```

- `id`: The unique identifier of the data sample, extracted from the input data's `id` field. If no `id` exists, an empty string is used
- `score`: The perplexity score for this sample. Lower values indicate the text is more predictable and natural to the model, while higher values suggest greater difficulty or unexpectedness

## Citation

```bibtex
@article{jelinek1977perplexity,
  title={Perplexity—a measure of the difficulty of speech recognition tasks},
  author={Jelinek, Fred and Mercer, Robert L and Bahl, Lalit R and Baker, James K},
  journal={The Journal of the Acoustical Society of America},
  volume={62},
  number={S1},
  pages={S63--S63},
  year={1977},
  publisher={Acoustical Society of America}
}
```

