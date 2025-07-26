import json
from typing import Dict, Any
import os
import numpy as np
from .utils_jsonl import load_jsonl, save_jsonl


def process_dict(input_dict):
    """
    Process a dictionary starting from the second key-value pair:
    - If the value is a list, compute its mean and replace the original value.
    - Otherwise, keep the value unchanged.

    Args:
        input_dict (dict): The dictionary to process.

    Returns:
        dict: The processed dictionary.
    """
    if not isinstance(input_dict, dict):
        raise TypeError("Input must be a dictionary")

    if len(input_dict) < 2:
        return input_dict

    keys = list(input_dict.keys())

    for key in keys[1:]:
        value = input_dict[key]

        if isinstance(value, list):
            try:
                mean = sum(value) / len(value)
                input_dict[key] = mean
            except (TypeError, ZeroDivisionError):
                pass

    return input_dict


def save_results(results: Dict[str, Any], output_path: str, detailed: bool = False):
    """
    Save scoring results. The main file keeps average scores; detailed mode retains full records.

    Args:
        results: Dictionary containing all scoring results.
        output_path: Directory path where results will be saved.
        detailed: Whether to generate a detailed file (default is False).
    """
    os.makedirs(output_path, exist_ok=True)

    simplified_results = {"dataset": results["dataset"]}
    for scorer_name, data in results.items():
        if scorer_name == "dataset":
            continue
        elif isinstance(data, list):
            try:
                scores = [item["score"] for item in data]
                mean_score = float(np.mean(scores))
                simplified_results[scorer_name] = {"score": mean_score}
            except (KeyError, TypeError):
                average_score = {}
                for key, value in data[0].items():
                    if "score" in key:
                        average_score[key] = float(
                            np.mean([item[key] for item in data]))
                simplified_results[scorer_name] = average_score
        else:
            simplified_results[scorer_name] = data

    main_output = os.path.join(output_path, "summary.json")
    with open(main_output, "w") as json_file:
        json.dump(simplified_results, json_file, indent=4, ensure_ascii=False)
    print(f"Simplified results saved to {main_output}")

    if detailed:
        detailed_dir = os.path.join(output_path, "detailed")
        os.makedirs(detailed_dir, exist_ok=True)

        for scorer_name, data in results.items():
            if scorer_name == "dataset":
                continue

            scorer_path = os.path.join(detailed_dir, f"{scorer_name}.json")

            with open(scorer_path, "w") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            print(f"Detailed results for {scorer_name} saved to {scorer_path}")
