# QuRateScorer

## Overview

The **QuRating Scorer** is a model-based evaluation tool designed to assess the quality of training data for language models across multiple dimensions. Proposed in the paper [Wettig et al., 2024](https://arxiv.org/abs/2402.09739), QuRating provides a principled approach to data selection by evaluating text quality along several interpretable axes. Unlike single-dimensional quality metrics, QuRating enables fine-grained analysis of what makes training data valuable, allowing practitioners to curate datasets based on specific quality attributes relevant to their use case.

The scorer employs a specialized sequence classification model trained to predict multiple quality dimensions simultaneously, providing both chunk-level and document-level quality assessments.

## Metric Definition:

* **Definition:** 

  QuRating evaluates text across multiple quality dimensions, each scored independently. The default dimensions include:
  
  - **writing_style**: Assesses the clarity, coherence, and stylistic quality of the text
  - **required_expertise**: Measures the level of domain knowledge or expertise reflected in the content
  - **facts_and_trivia**: Evaluates the presence and accuracy of factual information
  - **educational_value**: Quantifies how informative and instructive the content is

* **Explanation:** 

  Each dimension is scored independently, and the final score represents a weighted average across text chunks (weighted by token count).
  
  * **Higher scores** in each dimension indicate stronger presence of that quality attribute
  * **Lower scores** suggest deficiency in that particular aspect
  * The multi-dimensional nature allows for targeted data selection based on specific quality requirements

## YAML Configuration

```yaml
name: QuRateScorer
model: princeton-nlp/QuRater-1.3B
labels:
  - writing_style
  - required_expertise
  - facts_and_trivia
  - educational_value
chunk_size: 512
batch_size: 8
device_batch_size: 16
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"QuRateScorer"` | Identifier for the scorer |
| `model` | string | `"princeton-nlp/QuRater-1.3B"` | Path to the QuRater model (local or HuggingFace model ID) |
| `labels` | list | `["writing_style", "required_expertise", "facts_and_trivia", "educational_value"]` | List of quality dimensions to evaluate |
| `chunk_size` | integer | `512` | Maximum tokens per chunk for processing |
| `batch_size` | integer | `8` | Number of complete samples to process simultaneously |
| `device_batch_size` | integer | `16` | Number of text chunks to process per GPU forward pass |

## Underlying Model

The scorer uses [princeton-nlp/QuRater-1.3B](https://huggingface.co/princeton-nlp/QuRater-1.3B), a 1.3B parameter sequence classification model specifically trained for multi-dimensional data quality assessment. The model is based on transformer architecture and outputs scores for four quality dimensions simultaneously.

The model accepts tokenized text input and produces logits for each quality dimension, which are then used as quality scores. It processes text in chunks to handle documents of arbitrary length efficiently.

## Scoring Process

1. **Text Construction**: For each data sample, the scorer constructs the full text by concatenating:
   - Instruction
   - Input (if present)
   - Output/Response

2. **Tokenization and Chunking**: The concatenated text is tokenized without special tokens and split into chunks of size `chunk_size` (default 512 tokens)

3. **Chunk Scoring**: Each chunk is processed through the QuRater model:
   - Chunks are batched by similar length for efficiency
   - The model outputs logits for each quality dimension
   - Scores are computed on CPU for memory efficiency

4. **Aggregation**: Final scores are computed using a weighted average across all chunks:
   - Each chunk's score is weighted by its token count
   - Formula: `score = Σ(chunk_score_i × token_count_i) / Σ(token_count_i)`

5. **Result Compilation**: The scorer produces aggregate scores (weighted average per dimension) and per-chunk scores for detailed analysis

## Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "length": 1234,
  "num_chunks": 3,
  "writing_style_score": 4.56,
  "writing_style_chunks": [4.2, 4.8, 4.7],
  "required_expertise_score": 3.21,
  "required_expertise_chunks": [3.1, 3.3, 3.2],
  "facts_and_trivia_score": 5.12,
  "facts_and_trivia_chunks": [5.0, 5.2, 5.2],
  "educational_value_score": 4.89,
  "educational_value_chunks": [4.7, 5.0, 5.0]
}
```

- `id`: Unique identifier for the sample (from input data)
- `length`: Total number of tokens in the processed text
- `num_chunks`: Number of chunks the text was split into
- `{label}_score`: Weighted average score for the quality dimension (float)
- `{label}_chunks`: List of scores for each chunk in that dimension (list of floats)

*Note: The `{label}` placeholder is replaced with each configured label name (e.g., `writing_style`, `required_expertise`, etc.)*

## Citation

```bibtex
@inproceedings{wettig2024qurating,
  title={{QuRating}: Selecting High-Quality Data for Training Language Models},
  author={Wettig, Alexander and Gupta, Aatmik and Malik, Saumya and Chen, Danqi},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2024}
}
```
