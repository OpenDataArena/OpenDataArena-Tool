# UniEvalD2tScorer

## Overview

The **UniEval Data-to-Text (D2t) Scorer** is a model-based evaluation tool designed to assess the quality of data-to-text generation outputs. Introduced in [Zhong et al., 2022](https://arxiv.org/abs/2210.07197), UniEval provides a unified multi-dimensional evaluation framework for text generation tasks. This scorer specifically evaluates the naturalness and informativeness of generated text in data-to-text scenarios, where structured data is transformed into natural language descriptions.

Unlike traditional metrics that rely on simple n-gram matching (e.g., BLEU, ROUGE), UniEval leverages a fine-tuned language model to capture semantic quality and provides more nuanced assessments aligned with human judgments.

## Metric Definition:

* **Definition:** 

  UniEval D2t evaluates generated text along multiple quality dimensions:
  
  1. **Naturalness:** Measures how fluent, grammatical, and human-like the generated text appears
  2. **Informativeness:** Measures how much relevant information from the reference data is captured in the generated output
  3. **Overall:** (optional) An aggregate quality score that combines multiple dimensions into a single unified metric

* **Explanation:** 

  * A **higher naturalness score** indicates that the output reads smoothly and naturally, while a **lower score** suggests awkward phrasing, grammatical errors, or unnatural language constructs.
  * A **higher informativeness score** indicates that the generated text effectively conveys the key information, while a **lower score** suggests missing or insufficient content coverage.
  * All scores are **normalized between 0 and 1**, where higher values indicate better quality.

* **Key Advantages:**
  
  * **Multi-dimensional evaluation:** Provides separate scores for different quality aspects rather than a single monolithic metric
  * **Semantic understanding:** Leverages fine-tuned language models to capture semantic quality beyond simple n-gram matching
  * **Human alignment:** Trained on human evaluation data to better correlate with human judgments
  * **Reference-based assessment:** Compares generated text against gold standard references for more grounded evaluation

## YAML Configuration

```yaml
name: UniEvalD2tScorer
model: MingZhong/unieval-sum
dimensions: ['naturalness', 'informativeness']
max_length: 1024
batch_size: 8
overall: true
device: cuda:0
cache_dir: null
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"UniEvalD2tScorer"` | Identifier for the scorer |
| `model` | string | `"MingZhong/unieval-sum"` | Path or identifier for the UniEval model. Can be either a Hugging Face model identifier or a local path to a downloaded model directory |
| `dimensions` | list of strings | `['naturalness', 'informativeness']` | List of evaluation dimensions to compute. Valid options: `'naturalness'` (evaluates text fluency and grammaticality), `'informativeness'` (evaluates information coverage) |
| `max_length` | integer | `1024` | Maximum sequence length for tokenization. Text exceeding this limit will be truncated with a warning |
| `batch_size` | integer | `8` | Number of samples to process in each batch during evaluation. Larger values speed up evaluation but require more GPU memory |
| `overall` | boolean | `true` | Whether to compute an overall aggregate score combining all dimensions |
| `device` | string | auto-detect | Device for model inference (e.g., `cuda:0`, `cpu`). If not specified, automatically selects `cuda:0` if available, otherwise `cpu` |
| `cache_dir` | string or null | `null` | Directory to cache downloaded models. If `null`, uses the default Hugging Face cache location |


## Underlying Model

The scorer uses **[MingZhong/unieval-sum](https://huggingface.co/MingZhong/unieval-sum)** by default, which is a T5-based model fine-tuned on multiple text generation evaluation tasks. The model is trained to predict human evaluation scores across various quality dimensions.

**Alternative Models**: While `MingZhong/unieval-sum` is the default and recommended model, you can specify any compatible UniEval checkpoint or locally fine-tuned model by providing its path in the `model` configuration parameter.

## Scoring Process

The UniEval D2t scoring process operates as follows:

### Single Item Evaluation

1. **Data Extraction**: Extract the `output` (generated text) and `reference` (ground truth text) from each data item. The `reference` field is required and must not be empty.

2. **Truncation Check**: Tokenize the concatenated text (`output + reference`) and check if it exceeds `max_length`. If truncation occurs, a warning is issued with the item ID.

3. **Evaluation**: Pass the output-reference pair through the UniEval D2t evaluator, which:
   - Encodes both texts using the T5-based model
   - Computes dimension-specific scores based on the model's learned representations
   - Optionally computes an overall quality score

4. **Score Return**: Return a dictionary containing scores for each specified dimension.

### Batch Evaluation

For efficiency when processing entire datasets:

1. **Data Loading**: Read all data items from the input JSONL file and extract output-reference pairs.

2. **Truncation Validation**: Check all items for potential truncation issues and report the total count.

3. **Batch Processing**: Process data in batches of size `batch_size`, where each batch is evaluated in parallel.

4. **Result Aggregation**: Collect scores for all items and format them with corresponding item IDs.

## Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "UniEval_D2t_Naturalness": 0.8723,
  "UniEval_D2t_Informativeness": 0.7891,
  "UniEval_D2t_Overall": 0.8307
}
```

- `id`: Unique identifier for the data item (copied from the input data)
- `UniEval_D2t_Naturalness`: Naturalness score (0-1 range), measuring text fluency and grammaticality. Only present if `'naturalness'` is in the `dimensions` configuration
- `UniEval_D2t_Informativeness`: Informativeness score (0-1 range), measuring information coverage. Only present if `'informativeness'` is in the `dimensions` configuration
- `UniEval_D2t_Overall`: Overall aggregate quality score (0-1 range), combining all dimensions. Only present if `overall` is set to `true` in the configuration

**Note**: The output will only include dimensions that were specified in the configuration. Dimension names are automatically capitalized in the output field names.

## Citation

```bibtex
@article{zhong2022towards,
  title={Towards a unified multi-dimensional evaluator for text generation},
  author={Zhong, Ming and Liu, Yang and Yin, Da and Mao, Yuning and Jiao, Yizhu and Liu, Pengfei and Zhu, Chenguang and Ji, Heng and Han, Jiawei},
  journal={arXiv preprint arXiv:2210.07197},
  year={2022}
}
```

