# TreeInstructScorer

## Overview

The **TreeInstruct Complexity Scorer** is a syntax-tree-based evaluation tool designed to measure instruction complexity through structural analysis of text. Inspired by the paper [Zhao et al., 2023](https://arxiv.org/abs/2308.05696), this scorer quantifies complexity by analyzing the syntactic dependency tree structure rather than relying on surface-level features. The core insight is that more complex instructions naturally exhibit deeper and more elaborate syntactic trees with more nodes and hierarchical levels.

Unlike token-count-based metrics, TreeInstruct captures the **structural complexity** of instructions by examining how words relate to each other grammatically, providing a linguistically grounded measure of instruction difficulty.

## Metric Definition:

* **Definition:**
  
  1. Parse the text using dependency parsing to construct a syntactic tree
  2. Extract two structural metrics:
     - **TreeInstruct_Nodes**: Total number of nodes in the syntactic tree
     - **TreeInstruct_Depth**: Maximum depth (number of hierarchical levels) of the syntactic tree

* **Explanation:** The complexity is measured through syntactic tree structure:
  
  * **Higher node counts** indicate more words and relationships, suggesting greater information density and complexity
  * **Greater tree depth** reflects deeper nested grammatical structures, indicating more sophisticated linguistic constructions
  
  These metrics align with the Tree-Instruct methodology, which systematically enhances instruction complexity by adding nodes to semantic trees.

## YAML Configuration

```yaml
name: TreeInstructScorer
model: en_core_web_sm
max_workers: 8
text_fields:
  - instruction
  - input
  - output
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"TreeInstructScorer"` | Identifier for the scorer |
| `model` | string | `"en_core_web_sm"` | spaCy language model for dependency parsing (Options: `en_core_web_sm`, `en_core_web_md`, `en_core_web_lg` for English; for other languages, use corresponding spaCy models like `zh_core_web_sm` for Chinese) |
| `max_workers` | integer | CPU count | Number of parallel processes for scoring (higher values speed up processing but consume more memory; recommended to set to number of CPU cores available) |
| `text_fields` | list | `["instruction", "input", "output"]` | List of fields to concatenate for analysis (concatenates specified fields with newlines before parsing; customize based on your dataset schema) |

## Underlying Model

The scorer uses **spaCy** NLP models for syntactic dependency parsing. spaCy is an industrial-strength natural language processing library that provides accurate and efficient linguistic analysis.

**Default Model:** `en_core_web_sm` (English, small model)

**Alternative Models:**
- `en_core_web_md`: Medium English model (more accurate)
- `en_core_web_lg`: Large English model (most accurate)
- Language-specific models for non-English text

**Installation:**
```bash
python -m spacy download en_core_web_sm
```

For other languages or model sizes, visit [spaCy's model documentation](https://spacy.io/models).

## Scoring Process

1. **Text Preparation:** Concatenate text from specified fields (`text_fields`) with newline separators

2. **Dependency Parsing:** Process the text through the spaCy model to construct a syntactic dependency tree, where:
   - Each word becomes a node
   - Edges represent grammatical relationships
   - The sentence root serves as the tree root

3. **Node Counting:** Recursively traverse the tree starting from the root(s) to count all nodes (words)

4. **Depth Calculation:** Recursively compute the maximum depth by finding the longest path from root to leaf
   - A single word has depth 1
   - Depth increases by 1 for each level of nested dependencies

5. **Parallel Processing:** For large datasets, the scorer distributes work across multiple processes using `ProcessPoolExecutor` for efficient computation

## Output Format

For each input sample, the scorer returns:

```json
{
  "id": "sample_001",
  "TreeInstruct_Nodes": 45,
  "TreeInstruct_Depth": 8
}
```

- `id`: Unique identifier from the original data item (preserved from input)
- `TreeInstruct_Nodes`: Total number of nodes (tokens) in the syntactic tree (Range: [0, ∞); higher values indicate more complex syntactic structures with more words and relationships)
- `TreeInstruct_Depth`: Maximum depth of the syntactic tree (Range: [0, ∞); higher values indicate deeper nested grammatical constructions)

**Note:** If parsing fails or text is empty, both metrics return 0, with an optional `error` field containing the error message.

## Citation

```bibtex
@article{zhao2023preliminary,
  title={A Preliminary Study of the Intrinsic Relationship between Complexity and Alignment},
  author={Yingxiu Zhao and Bowen Yu and Binyuan Hui and Haiyang Yu and Fei Huang and Yongbin Li and Nevin L. Zhang},
  year={2023},
  eprint={2308.05696},
  archivePrefix={arXiv},
}
```
