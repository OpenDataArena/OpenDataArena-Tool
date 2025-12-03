# StrLengthScorer

## Overview

The **String Length Scorer** is a heuristic-based evaluation tool designed to measure the total character length of specified fields in SFT (Supervised Fine-Tuning) data samples. This scorer provides a simple, efficient, and model-free metric to assess the verbosity or content volume of training data. It is particularly useful for filtering data based on length requirements, identifying overly brief or excessively verbose samples, or analyzing dataset characteristics.

Unlike model-based approaches, the String Length Scorer requires no GPU resources and operates purely through string operations, making it extremely fast and scalable for large datasets. It supports multiprocessing for efficient parallel computation across multiple CPU cores.

## Metric Definition:

* **Definition:** 

  `Str_Length = len(concatenated_text)`
  
  Where `concatenated_text` is formed by joining specified fields (e.g., `instruction`, `input`, `output`) with newline separators.

* **Explanation:** This metric quantifies the total number of characters in the combined text of selected fields:
  
  * A **higher Str_Length value** indicates more verbose or content-rich data samples, which may contain more detailed instructions, longer responses, or comprehensive information.
  * A **lower Str_Length value** suggests concise or minimal content, which may indicate simple instructions or brief responses.

## YAML Configuration

```yaml
name: StrLengthScorer
fields:
  - instruction
  - input
  - output
max_workers: 8
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"StrLengthScorer"` | Identifier for the scorer |
| `fields` | list of strings | `["instruction", "input", "output"]` | Specifies which data fields should be included in the length calculation. The fields are concatenated with newline separators (`\n`) before measuring length |
| `max_workers` | integer | CPU core count | The number of parallel worker processes to use for processing the dataset. Higher values can significantly speed up processing on multi-core systems |

## Scoring Process

1. **Field Extraction**: For each data item, the scorer extracts the values of specified fields (e.g., `instruction`, `input`, `output`) from the JSON record.

2. **Text Concatenation**: All extracted field values are converted to strings and joined together using newline characters (`\n`) as separators. Empty or missing fields are automatically excluded.

3. **Length Calculation**: The total character count of the concatenated text is computed using Python's built-in `len()` function. This includes all characters: letters, numbers, spaces, punctuation, and special characters.

4. **Parallel Processing**: To handle large datasets efficiently, the scorer uses Python's `ProcessPoolExecutor` to distribute the workload across multiple CPU cores. Each worker process independently computes scores for a subset of data items.

5. **Progress Tracking**: A progress bar (via `tqdm`) displays real-time processing status, showing the number of items processed and estimated time remaining.

6. **Error Handling**: If an error occurs during processing of any individual item (e.g., malformed JSON), the scorer returns a score of `0` for that item and logs the error, allowing the pipeline to continue.

## Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 1247
}
```

- `id`: The unique identifier of the data item, extracted from the `"id"` field in the input JSON
- `score`: The total character length of the concatenated text from specified fields (non-negative integer)

## Citation

```bibtex
@misc{opendataarena_tool_2025,
  author       = {OpenDataArena},
  title        = {{OpenDataArena-Tool}},
  year         = {2025},
  url          = {https://github.com/OpenDataArena/OpenDataArena-Tool},
  note         = {GitHub repository},
  howpublished = {\url{https://github.com/OpenDataArena/OpenDataArena-Tool}},
}
```
