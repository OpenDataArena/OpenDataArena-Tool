# UniEvalDialogScorer

## Overview

The **UniEval Dialog Scorer** is a model-based evaluation tool designed to assess the quality of dialogue generation systems. Introduced in [Zhong et al., 2022](https://arxiv.org/abs/2210.07197), UniEval provides a unified multi-dimensional evaluation framework for text generation tasks. This scorer specifically evaluates dialogue responses across multiple quality dimensions including naturalness, coherence, engagingness, groundedness, and understandability.

Unlike traditional dialogue evaluation metrics that focus on single aspects or require extensive human annotations, UniEval Dialog leverages a fine-tuned language model to provide comprehensive, automated assessments that align closely with human judgments across multiple quality facets.

## Metric Definition:

* **Definition:** 

  UniEval Dialog evaluates generated dialogue responses along the following dimensions:

  * **Naturalness**: Measures how fluent, grammatical, and human-like the dialogue response appears. A higher naturalness score indicates that the response reads smoothly and naturally, while a lower score suggests awkward phrasing, grammatical errors, or robotic language.

  * **Coherence**: Measures how logically consistent and contextually relevant the response is to the dialogue history. A higher coherence score indicates strong logical flow and contextual appropriateness, while a lower score suggests disconnected or irrelevant responses.

  * **Engagingness**: Measures how interesting, engaging, and conversationally appropriate the response is. A higher engagingness score indicates that the response is likely to maintain user interest and encourage continued conversation, while a lower score suggests bland or disengaging content.

  * **Groundedness**: Measures how well the response is grounded in the provided context or knowledge source. A higher groundedness score indicates that the response accurately reflects and utilizes the given context, while a lower score suggests the response deviates from or contradicts the contextual information.

  * **Understandability**: Measures how clear and easy to understand the response is for users. A higher understandability score indicates that the response is straightforward and comprehensible, while a lower score suggests confusing or ambiguous content.

  * **Overall** (optional): When enabled, provides an aggregate quality score that combines multiple dimensions into a single unified metric.

* **Explanation:** All scores are normalized between 0 and 1, where higher values indicate better quality across all dimensions.

## YAML Configuration

```yaml
name: UniEvalDialogScorer
model: MingZhong/unieval-dialog
dimensions: ['naturalness', 'coherence', 'engagingness', 'groundedness', 'understandability']
max_length: 1024
batch_size: 8
overall: true
device: cuda:0
cache_dir: null
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"UniEvalDialogScorer"` | Identifier for the scorer |
| `model` | string | `"MingZhong/unieval-dialog"` | Path or identifier for the UniEval Dialog model. Can be a Hugging Face model identifier or a local path to a downloaded model directory |
| `dimensions` | list of strings | `['naturalness', 'coherence', 'engagingness', 'groundedness', 'understandability']` | List of evaluation dimensions to compute. Valid options: `'naturalness'`, `'coherence'`, `'engagingness'`, `'groundedness'`, `'understandability'` |
| `max_length` | integer | `1024` | Maximum sequence length for tokenization. Text exceeding this limit will be truncated with a warning |
| `batch_size` | integer | `8` | Number of samples to process in each batch during evaluation. Larger values speed up evaluation but require more GPU memory |
| `overall` | boolean | `true` | Whether to compute an overall aggregate score combining all dimensions |
| `device` | string | auto-detect | Device for model inference (e.g., `cuda:0`, `cpu`). If not specified, automatically selects `cuda:0` if available, otherwise `cpu` |
| `cache_dir` | string or null | `null` | Directory to cache downloaded models. If `null`, uses the default Hugging Face cache location |


## Underlying Model

The scorer uses **[MingZhong/unieval-dialog](https://huggingface.co/MingZhong/unieval-dialog)** by default, which is a T5-based model specifically fine-tuned for multi-dimensional dialogue evaluation. The model is trained to predict human evaluation scores across various dialogue quality dimensions.

## Scoring Process

1. **Input Processing**: For each data sample, the scorer extracts:
   - Instruction (from `instruction` field)
   - Input (optional, from `input` field)
   - Output (from `output` field) - the generated dialogue response to be evaluated
   - Context (from `context` field) - required contextual information or knowledge base (must not be empty)

2. **Data Preparation**: 
   - Combine `instruction` and `input` (if present) to form the dialogue history (`source`)
   - Format the source with proper line breaks (ending with `\n\n` as required by UniEval)
   - Validate the `context` (must not be empty, automatically adds trailing `\n` if missing)

3. **Tokenization & Truncation Check**: Tokenize the concatenated text (`source + output + context`) and check if it exceeds `max_length`. If truncation occurs, a warning is issued

4. **Model Evaluation**: Pass the source-output-context triplet through the UniEval Dialog evaluator, which:
   - Encodes all components using the T5-based model
   - Computes dimension-specific scores based on the model's learned representations
   - Optionally computes an overall quality score

5. **Batch Processing**: For efficiency, data is processed in batches of size `batch_size`, with each batch evaluated in parallel

6. **Score Computation**: Return a dictionary containing scores for each specified dimension (naturalness, coherence, engagingness, groundedness, understandability, and optionally overall)

## Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "UniEval_Dialog_Naturalness": 0.8923,
  "UniEval_Dialog_Coherence": 0.8567,
  "UniEval_Dialog_Engagingness": 0.8234,
  "UniEval_Dialog_Groundedness": 0.8791,
  "UniEval_Dialog_Understandability": 0.8645,
  "UniEval_Dialog_Overall": 0.8632
}
```

- `id`: Unique identifier for the data item
- `UniEval_Dialog_Naturalness`: Naturalness score (0-1 range), measuring text fluency and grammaticality
- `UniEval_Dialog_Coherence`: Coherence score (0-1 range), measuring logical consistency and contextual relevance
- `UniEval_Dialog_Engagingness`: Engagingness score (0-1 range), measuring conversational quality and user engagement
- `UniEval_Dialog_Groundedness`: Groundedness score (0-1 range), measuring fidelity to the provided context
- `UniEval_Dialog_Understandability`: Understandability score (0-1 range), measuring clarity and comprehensibility
- `UniEval_Dialog_Overall`: Overall aggregate quality score (0-1 range), combining all dimensions (only present if `overall=true`)

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

