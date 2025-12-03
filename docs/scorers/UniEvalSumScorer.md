# UniEvalSumScorer

## Overview

The **UniEval Summarization Scorer** is a model-based evaluation tool designed to assess the quality of text summarization outputs. Introduced in [Zhong et al., 2022](https://arxiv.org/abs/2210.07197), this scorer is part of the UniEval unified evaluation framework for text generation tasks. It provides comprehensive multi-dimensional evaluation of summaries, measuring coherence, consistency, fluency, and relevance.

Unlike traditional summarization metrics such as ROUGE that rely on lexical overlap, UniEval Summarization leverages a fine-tuned language model to capture semantic quality and provide more nuanced assessments that better correlate with human judgments across multiple quality dimensions.

## Metric Definition:

* **Definition:** 

  UniEval Summarization evaluates generated summaries along the following dimensions:
  
  1. **Coherence:** Measures the logical flow and structural organization of the summary
  2. **Consistency:** Measures the factual consistency between the summary and the source document
  3. **Fluency:** Measures the grammatical correctness and readability of the summary
  4. **Relevance:** Measures how well the summary captures the key information from the source document
  5. **Overall** (optional): Aggregate quality score combining all dimensions

* **Explanation:** Each dimension provides specific quality insights:
  
  * **Coherence**: A **higher score** indicates well-organized content with smooth transitions between sentences and ideas, while a **lower score** suggests disjointed or poorly structured content.
  * **Consistency**: A **higher score** indicates accurate reflection of source information without contradictions or hallucinations, while a **lower score** suggests factual errors or misrepresentations.
  * **Fluency**: A **higher score** indicates grammatically correct, natural-sounding, and easy-to-read text, while a **lower score** suggests grammatical errors, awkward phrasing, or unclear expression.
  * **Relevance**: A **higher score** indicates focus on important content and omission of irrelevant details, while a **lower score** suggests missed key points or unnecessary information.

* **Key Advantages:**
  
  * **Multi-dimensional assessment:** Provides comprehensive evaluation across multiple quality aspects
  * **Semantic understanding:** Leverages fine-tuned language models instead of lexical overlap
  * **Human correlation:** Better aligns with human judgments compared to traditional metrics
  * **Normalized scores:** All scores range from 0 to 1 for consistent interpretation

## YAML Configuration

```yaml
name: UniEvalSumScorer
model: MingZhong/unieval-sum
dimensions: ['coherence', 'consistency', 'fluency', 'relevance']
max_length: 1024
batch_size: 8
overall: true
device: cuda:0
cache_dir: null
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"UniEvalSumScorer"` | Identifier for the scorer |
| `model` | string | `"MingZhong/unieval-sum"` | HuggingFace model path or local directory for the UniEval Summarization model |
| `dimensions` | list | `['coherence', 'consistency', 'fluency', 'relevance']` | List of evaluation dimensions: `'coherence'`, `'consistency'`, `'fluency'`, `'relevance'` |
| `max_length` | integer | `1024` | Maximum sequence length for tokenization (text exceeding this will be truncated) |
| `batch_size` | integer | `8` | Number of samples to process in each batch (larger values require more GPU memory) |
| `overall` | boolean | `true` | Whether to compute an overall aggregate score combining all dimensions |
| `device` | string | `"cuda:0"` (auto) | Device for model inference (`cuda:0`, `cpu`, etc.) |
| `cache_dir` | string/null | `null` | Directory to cache downloaded models (null uses default HuggingFace cache) |

## Underlying Model

The scorer uses **[MingZhong/unieval-sum](https://huggingface.co/MingZhong/unieval-sum)** by default, which is a T5-based model specifically fine-tuned for multi-dimensional summarization evaluation. The model is trained to predict human evaluation scores across various summary quality dimensions.

**Alternative Models**: While `MingZhong/unieval-sum` is the default and recommended model for summarization evaluation, you can specify any compatible UniEval summarization checkpoint or locally fine-tuned model by providing its path in the `model` configuration parameter.

## Scoring Process

The UniEval Summarization scoring process operates as follows:

### Data Requirements

The scorer expects each data item to contain:
- **`instruction`**: The summarization instruction or prompt
- **`input`** (optional): Additional context or source document information
- **`output`**: The generated summary to be evaluated
- **`reference`**: Required reference summary (must not be empty)

### Single Item Evaluation

1. **Data Extraction**: Extract and construct the required fields from each data item:
   - Combine `instruction` and `input` (if present) to form the source document
   - Extract the `output` (generated summary to be evaluated)
   - Extract and validate the `reference` (must not be empty)

2. **Truncation Check**: Tokenize the concatenated text (`source + output + reference`) and check if it exceeds `max_length`. If truncation occurs, a warning is issued with the item ID to alert users of potential information loss.

3. **Multi-dimensional Evaluation**: Pass the source-output-reference triplet through the UniEval Summarization evaluator, which:
   - Encodes all components using the T5-based model
   - Computes dimension-specific scores (coherence, consistency, fluency, relevance) based on the model's learned representations
   - Optionally computes an overall quality score combining all dimensions

4. **Score Return**: Return a dictionary containing scores for each specified dimension.

### Batch Evaluation

For efficiency when processing entire datasets:

1. **Data Loading**: Read all data items from the input JSONL file and extract source-output-reference triplets.

2. **Truncation Validation**: Check all items for potential truncation issues and report the total count of items that exceed the maximum length.

3. **Batch Processing**: Process data in batches of size `batch_size`, where each batch is evaluated in parallel to maximize throughput and GPU utilization.

4. **Result Aggregation**: Collect scores for all items and format them with corresponding item IDs.

## Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "UniEval_Sum_Coherence": 0.8756,
  "UniEval_Sum_Consistency": 0.8934,
  "UniEval_Sum_Fluency": 0.9012,
  "UniEval_Sum_Relevance": 0.8623,
  "UniEval_Sum_Overall": 0.8831
}
```

- `id`: Unique identifier for the data item (copied from input data)
- `UniEval_Sum_Coherence`: Coherence score (0-1) measuring logical flow and structural organization (only if `'coherence'` in dimensions)
- `UniEval_Sum_Consistency`: Consistency score (0-1) measuring factual consistency with source document (only if `'consistency'` in dimensions)
- `UniEval_Sum_Fluency`: Fluency score (0-1) measuring grammatical correctness and readability (only if `'fluency'` in dimensions)
- `UniEval_Sum_Relevance`: Relevance score (0-1) measuring coverage of key information (only if `'relevance'` in dimensions)
- `UniEval_Sum_Overall`: Overall aggregate quality score (0-1) combining all dimensions (only if `overall=true`)

## Citation

```bibtex
@article{zhong2022towards,
  title     = {Towards a unified multi-dimensional evaluator for text generation},
  author    = {Zhong, Ming and Liu, Yang and Yin, Da and Mao, Yuning and Jiao, Yizhu and Liu, Pengfei and Zhu, Chenguang and Ji, Heng and Han, Jiawei},
  journal   = {arXiv preprint arXiv:2210.07197},
  year      = {2022}
}
```
