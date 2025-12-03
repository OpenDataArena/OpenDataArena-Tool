# TsPythonScorer

## Overview

The **TsPythonScorer** is a syntax-based code quality scorer that validates Python code correctness using static analysis. This scorer leverages the Tree-sitter parsing library to check whether Python code snippets are syntactically valid. It is designed to filter out code samples with syntax errors, which are typically low-quality or incomplete code fragments unsuitable for model training.

The scorer supports both plain Python code and code embedded within Markdown code blocks (triple backticks format), making it versatile for processing various data formats commonly found in code-related datasets. It employs multiprocessing to efficiently handle large-scale datasets.

This scorer aligns with the data quality control principles discussed in the Seed-Coder paper, where ensuring syntactic correctness is a fundamental prerequisite for code pretraining data.

## Metric Definition:

* **Definition:** 

  The scorer outputs a **binary score (0.0 or 1.0)** indicating whether the code is syntactically correct:
  
  - **1.0**: All Python code in the sample is syntactically valid
  - **0.0**: The sample contains syntax errors, is empty, or cannot be parsed

* **Explanation:** 
  
  For samples containing multiple Markdown code blocks, **all blocks must be syntactically correct** for the sample to receive a score of 1.0. If any code block fails parsing, the entire sample is scored as 0.0.

* **Key Features:**
  
  - **Format-agnostic:** Handles both plain Python code and Markdown-embedded code blocks
  - **Strict validation:** Any syntax error results in a zero score
  - **Efficient processing:** Leverages multiprocessing for scalable batch evaluation

## YAML Configuration

```yaml
scorers:
  - name: ts_python_syntax
    type: TsPythonScorer
    config:
      field: "output"
      max_workers: 16
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `field` | string | `"output"` | Specifies the JSON field name containing the code to be evaluated |
| `max_workers` | integer | CPU count | Number of parallel worker processes for batch evaluation. Higher values increase processing speed but consume more system resources |

## Underlying Model

This scorer does not use any language model. Instead, it relies on the **Tree-sitter** parsing library with the `tree-sitter-python` grammar for syntactic analysis. Tree-sitter is a parser generator tool that builds concrete syntax trees and is widely used in code editors and static analysis tools for syntax validation.

## Scoring Process

1. **Text Extraction**: Extract the text content from the specified field (default: `"output"`) in the JSON data item

2. **Markdown Code Block Detection**: 
   - Attempt to identify Markdown code blocks using the pattern: ` ```[language]\n...\n``` `
   - If code blocks are found, extract all of them for individual validation
   - If no code blocks are detected, treat the entire text as Python code

3. **Syntax Parsing**:
   - For each code snippet (either extracted blocks or the full text), use Tree-sitter to parse the Python code
   - Check if the resulting Abstract Syntax Tree (AST) contains any error nodes

4. **Score Determination**:
   - If all code snippets parse successfully without errors: **score = 1.0**
   - If any snippet contains syntax errors or is empty: **score = 0.0**
   - If the field is missing or invalid: **score = 0.0**

5. **Parallel Processing**: When evaluating datasets in batch mode, the scorer distributes samples across multiple worker processes for efficient parallel processing

## Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 1.0
}
```

- `id`: The unique identifier from the input data item. If missing, defaults to `"unknown"`
- `score`: Binary score indicating syntax validity (1.0 = syntactically correct, 0.0 = contains syntax errors)

## Citation

```bibtex
@misc{tree_sitter,
  title        = {tree-sitter},
  author       = {Max Brunsfeld and contributors},
  year         = {2018},
  howpublished = {\url{https://github.com/tree-sitter/tree-sitter}},
  note         = {GitHub repository},
}
```
