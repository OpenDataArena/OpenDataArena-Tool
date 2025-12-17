# UniEvalFactScorer

## Overview

The **UniEvalFactScorer** is a model-based evaluation tool designed to assess the factual consistency of generated text with respect to source documents. Introduced in [Zhong et al., 2022](https://arxiv.org/abs/2210.07197), this scorer is part of the UniEval unified evaluation framework for text generation tasks. It specifically focuses on detecting factual errors, hallucinations, and inconsistencies in generated outputs by comparing them against source information.

This scorer is particularly valuable for tasks such as summarization, question answering, and information extraction, where maintaining factual accuracy is critical. Unlike simple lexical overlap metrics, UniEval Fact leverages a fine-tuned language model to capture semantic-level consistency and identify subtle factual discrepancies.

## Metric Definition:

* **Definition:** 

  UniEval Fact evaluates generated text along a single primary dimension of **Consistency (Factual Consistency)**, which measures how factually consistent the generated output is with the source document. This metric assesses whether the generated text accurately reflects the information in the source without introducing false claims, contradictions, or hallucinated content.

* **Explanation:** The score is normalized between 0 and 1:
  
  * A **higher consistency score** (closer to 1) indicates that the generated output is highly faithful to the source document, with no factual errors or hallucinations.
  * A **lower consistency score** (closer to 0) suggests that the output contains factual inconsistencies, contradictions, or information not supported by the source.

* **Key Advantages:**
  
  * **Semantic-level evaluation:** Unlike simple lexical overlap metrics, UniEval Fact leverages a fine-tuned language model to capture semantic-level consistency
  * **Hallucination detection:** Specifically designed to identify subtle factual discrepancies and hallucinated content
  * **Source-grounded:** Provides objective assessment by comparing generated text against source documents

## YAML Configuration

```yaml
name: UniEvalFactScorer
model: MingZhong/unieval-fact
max_length: 1024
batch_size: 8
device: cuda:0
cache_dir: null
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"UniEvalFactScorer"` | Identifier for the scorer |
| `model` | string | `"MingZhong/unieval-fact"` | Path or identifier for the UniEval Fact model. Can be a Hugging Face model identifier or local path |
| `max_length` | integer | `1024` | Maximum sequence length for tokenization. Text exceeding this limit will be truncated with a warning |
| `batch_size` | integer | `8` | Number of samples to process in each batch during evaluation |
| `device` | string | `"cuda:0"` | Device for model inference (e.g., `cuda:0`, `cpu`). Auto-selects if not specified |
| `cache_dir` | string or null | `null` | Directory to cache downloaded models. Uses default Hugging Face cache location if `null` |

## Underlying Model

The scorer uses **[MingZhong/unieval-fact](https://huggingface.co/MingZhong/unieval-fact)** by default, which is a T5-based model specifically fine-tuned for factual consistency evaluation. The model is trained to detect factual errors, hallucinations, and inconsistencies by comparing generated text against source documents.

## Scoring Process

The UniEval Fact scoring process operates as follows:

### Data Requirements

The scorer expects each data item to contain:
- **`instruction`**: The original instruction or prompt
- **`input`** (optional): Additional source information or context
- **`output`**: The generated text to be evaluated for factual consistency

### Single Item Evaluation

1. **Data Extraction**: Extract and construct the required fields from each data item:
   - Combine `instruction` and `input` (if present) to form the source document
   - The source serves as the ground truth against which factual consistency is measured
   - Extract the `output` (generated text to be evaluated)

2. **Truncation Check**: Tokenize the concatenated text (`source + output`) and check if it exceeds `max_length`. If truncation occurs, a warning is issued with the item ID to alert users of potential information loss.

3. **Consistency Evaluation**: Pass the source-output pair through the UniEval Fact evaluator, which:
   - Encodes both the source and output using the T5-based model
   - Computes a consistency score based on semantic alignment and factual accuracy
   - Detects contradictions, hallucinations, and unsupported claims

4. **Score Return**: Return a dictionary containing the consistency score.

### Batch Evaluation

For efficiency when processing entire datasets:

1. **Data Loading**: Read all data items from the input JSONL file and extract source-output pairs.

2. **Truncation Validation**: Check all items for potential truncation issues and report the total count of items that exceed the maximum length.

3. **Batch Processing**: Process data in batches of size `batch_size`, where each batch is evaluated in parallel to maximize throughput.

4. **Result Aggregation**: Collect consistency scores for all items and format them with corresponding item IDs.

## Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "UniEval_Fact_Consistency": 0.8945
}
```

- `id`: Unique identifier for the data item (copied from the input data)
- `UniEval_Fact_Consistency`: Factual consistency score (0-1 range), measuring how well the generated output aligns with the source document without introducing factual errors or hallucinations. Higher scores indicate better factual consistency

## Citation

```bibtex
@article{zhong2022towards,
  title={Towards a unified multi-dimensional evaluator for text generation},
  author={Zhong, Ming and Liu, Yang and Yin, Da and Mao, Yuning and Jiao, Yizhu and Liu, Pengfei and Zhu, Chenguang and Ji, Heng and Han, Jiawei},
  journal={arXiv preprint arXiv:2210.07197},
  year={2022}
}
```

