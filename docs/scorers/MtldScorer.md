# MtldScorer

## Overview

The **MTLD (Measure of Textual Lexical Diversity) Scorer** is a statistical evaluation tool designed to measure **lexical diversity** in text through a novel sequential analysis approach. Proposed by McCarthy & Jarvis (2010), MTLD calculates lexical diversity as the mean length of sequential word strings that maintain a specified Type-Token Ratio (TTR) threshold. This approach addresses the well-known sensitivity of traditional TTR measures to text length, providing a more robust and length-independent assessment of vocabulary richness.

Unlike model-based scorers, MTLD is a **statistical metric** that does not require any pre-trained models. It is particularly valuable for evaluating the linguistic complexity and vocabulary variety of instruction-following datasets, offering complementary insights to other lexical diversity measures like HD-D.

## Metric Definition:

* **Definition:** 

  MTLD is calculated as the average of forward and backward traversals through the text:

  ```
  MTLD = (MTLD_forward + MTLD_backward) / 2
  ```

  where each directional MTLD is computed as:

  ```
  MTLD_directional = total_tokens / factor_count
  ```

  A **factor** is defined as a contiguous sequence of tokens where the TTR (unique types / total tokens) remains above a specified threshold. When TTR drops to or below the threshold, a new factor begins.

* **Explanation:** This metric estimates **lexical diversity** by measuring how long the text can maintain vocabulary variety before repeating words:
  
  * A **higher MTLD score** indicates **greater lexical diversity**, meaning the text maintains vocabulary richness over longer sequences, suggesting sophisticated and varied language use.
  * A **lower MTLD score** indicates **lower lexical diversity**, meaning the text quickly exhausts its vocabulary and begins repeating words, suggesting simpler or more repetitive language.

* **Key Advantages:**
  
  * **Length-invariant:** Unlike traditional TTR measures, MTLD is not biased by text length
  * **Bidirectional calculation:** Forward and backward traversals help mitigate artifacts from word ordering
  * **Partial factor inclusion:** Provides more accurate measures for texts that don't end on factor boundaries

## YAML Configuration

```yaml
name: MtldScorer
ttr_threshold: 0.72
max_workers: 8
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"MtldScorer"` | Identifier for the scorer |
| `ttr_threshold` | float | `0.72` | The Type-Token Ratio threshold used to determine factor boundaries. When TTR drops to or below this value, it marks the end of a factor. Range: (0, 1). The default 0.72 is established in the original research as providing optimal discrimination |
| `max_workers` | integer | CPU core count | Number of parallel worker processes for multiprocessing. Higher values speed up processing but consume more memory |

## Underlying Model

**Not applicable.** MTLD is a statistical metric based on sequential Type-Token Ratio analysis and does not require any pre-trained language models. The calculation is purely algorithmic and depends only on the sequential ordering and frequency of words in the text.

## Scoring Process

1. **Input Processing**: For each data sample, the scorer concatenates the `instruction`, `input`, and `output` fields into a single text string. If the `input` field is empty, it is omitted from concatenation

2. **Tokenization**: The concatenated text is split into tokens (words) using whitespace separation

3. **Preprocessing**: Each token is processed by removing all punctuation marks and converting to lowercase

4. **Forward MTLD Calculation**: Sequentially process tokens from start to end, tracking TTR. Each time TTR drops to or below the threshold, count a complete factor. Calculate `MTLD_forward = total_tokens / (factor_count + partial_factor)`

5. **Backward MTLD Calculation**: Repeat the forward calculation with the token sequence reversed to get `MTLD_backward`

6. **Score Computation**: Calculate final score as `(MTLD_forward + MTLD_backward) / 2`

## Output Format

For each input sample, the scorer returns:

```json
{
  "id": 1,
  "score": 67.34
}
```

- `id`: The unique identifier of the data sample, extracted from the input data's `id` field. If no `id` is present, defaults to `"unknown"`
- `score`: The MTLD lexical diversity score for the sample. Typically ranges from a few (for highly repetitive text) to hundreds or more (for highly diverse text). Higher values indicate greater lexical diversity. Value of 0.0 indicates empty text or processing error
- `error` (optional): Only present if an error occurred during processing. Contains the error message for debugging purposes

## Citation

```bibtex
@article{mccarthy2010mtld,
  title={MTLD, vocd-D, and HD-D: A validation study of sophisticated approaches to lexical diversity assessment},
  author={McCarthy, Philip M and Jarvis, Scott},
  journal={Behavior research methods},
  volume={42},
  number={2},
  pages={381--392},
  year={2010},
  publisher={Springer}
}
```

