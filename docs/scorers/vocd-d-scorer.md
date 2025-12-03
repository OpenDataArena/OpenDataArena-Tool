# VocdDScorer

## Overview

The **VOCD-D (Vocabulary Diversity-D) Scorer** is a statistical evaluation tool designed to measure **lexical diversity** in text through a sophisticated mathematical modeling approach. Proposed by Malvern, Richards, Chipere, and Durán (2004), VOCD-D addresses the fundamental limitation of traditional Type-Token Ratio (TTR) measures: their sensitivity to text length.

The measure is based on a mathematical model that describes the relationship between types and tokens, and uses curve-fitting procedures to estimate a single parameter `D` that quantifies lexical diversity. This approach provides a robust, length-independent assessment of vocabulary richness that remains stable across varying text lengths, making it particularly valuable for comparing texts of different sizes.

Unlike model-based scorers, VOCD-D is a **statistical metric** that does not require any pre-trained models. It is particularly valuable for evaluating the linguistic complexity and vocabulary variety of instruction-following datasets, offering insights into the lexical sophistication of both instructions and responses.

## Metric Definition:

* **Definition:** VOCD-D is calculated through a curve-fitting procedure that models the relationship between types (unique words) and tokens (total words) in randomly sampled text segments of varying lengths. The algorithm:

  1. Takes random samples of tokens from the text at multiple sample sizes (typically from 35 to `ntokens`)
  2. For each sample size, calculates the average Type-Token Ratio across multiple trials (`within_sample`)
  3. Fits these observed TTR values to a mathematical model curve
  4. Extracts the parameter `D` that best describes the curve, representing the text's lexical diversity

* **Explanation:** The D parameter represents the **inherent lexical diversity** of the text, independent of its length:

  * A **higher D score** (typically ranging from 50-100+) indicates **greater lexical diversity**, meaning the text demonstrates rich and varied vocabulary usage with less repetition, suggesting sophisticated and diverse language use.
  * A **lower D score** (typically below 50) indicates **lower lexical diversity**, meaning the text relies on a more limited vocabulary with higher word repetition, suggesting simpler or more constrained language.
  * A **score of 0.0** indicates insufficient text for analysis (fewer tokens than the minimum required for sampling).

The mathematical modeling approach makes VOCD-D more robust than simple TTR measures, as it accounts for the stochastic nature of word occurrence and provides consistent results regardless of text length, making it suitable for comparing diverse text samples.

## YAML Configuration

```yaml
name: VocdDScorer
ntokens: 50
within_sample: 100
seed: 42
max_workers: 128
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"VocdDScorer"` | Identifier for the scorer |
| `ntokens` | integer | `50` | Maximum number of tokens used for sampling in the curve-fitting procedure. This determines the upper bound of sample sizes used when modeling the type-token relationship. Texts must contain at least this many words to receive a non-zero score. Typical values range from 35-100. |
| `within_sample` | integer | `100` | Number of random samples taken at each sample size during the curve-fitting procedure. Higher values (e.g., 100-200) provide more reliable estimates but take longer to compute. Lower values (e.g., 20-50) are faster but may be less stable. |
| `seed` | integer | `42` | Random seed for reproducible sampling. Ensures that repeated calculations on the same text yield identical results. Setting a fixed seed is recommended for reproducible research and consistent evaluation results. |
| `max_workers` | integer | `128` | Number of parallel worker processes for multiprocessing. Higher values can significantly speed up processing for large datasets but consume more CPU resources and memory. |

## Underlying Model

VOCD-D is a statistical metric based on mathematical modeling of the type-token relationship and does not require any pre-trained language models. The calculation is purely algorithmic, using curve-fitting techniques to estimate the diversity parameter D from observed type-token patterns in randomly sampled text segments. The implementation uses the [lexicalrichness](https://github.com/LSYS/lexicalrichness) Python library, which provides an efficient implementation of the VOCD algorithm originally proposed by Malvern et al.

## Scoring Process

1. **Text Concatenation**: For each data sample, the `instruction`, `input`, and `output` fields are concatenated into a single text string. If the `input` field is empty, it is omitted from concatenation.

2. **Text Validation**: Check if the text contains sufficient content. Empty or whitespace-only texts receive a score of 0.0. Texts with fewer tokens than the configured `ntokens` value receive a score of 0.0.

3. **Tokenization**: The text is split into tokens (words)

4. **Random Sampling**: For sample sizes ranging from 35 to `ntokens`, take `within_sample` random samples of each size and calculate the average Type-Token Ratio

5. **Curve Fitting**: Fit the observed TTR values to the theoretical VOCD model curve

6. **Parameter Estimation**: Extract the D parameter that best describes the curve, representing lexical diversity

7. **Error Handling**: If the calculation fails for any reason, the scorer returns 0.0 and logs the error

## Output Format

For each input sample, the scorer returns:

```json
{
  "id": "1",
  "score": 67.45
}
```

- `id`: The unique identifier of the data sample, extracted from the input data's `id` field. If no `id` is present, defaults to `"unknown"`.
- `score`: The VOCD-D lexical diversity score (D parameter) for the sample. Typically ranges from 0 to 100+, with most texts falling between 20-100. Higher values (typically 70-100+) indicate very rich and diverse vocabulary. Lower values (typically below 40) indicate limited vocabulary diversity. A value of 0.0 indicates empty text, insufficient text length (fewer than `ntokens` words), or processing error.
- `error` (optional): Only present if an error occurred during processing. Contains the error message for debugging purposes.

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
