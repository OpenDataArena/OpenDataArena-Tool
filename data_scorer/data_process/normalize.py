import argparse
import json
from tqdm import tqdm
from typing import Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        default='/mnt/petrelfs/gaoxin/OpenDataArena/data_evaluation/example_input_add_key.jsonl',
    )
    parser.add_argument(
        "--output_file",
        default='/mnt/petrelfs/gaoxin/OpenDataArena/data_evaluation/example_input_add_key1.jsonl',
    )
    return parser.parse_args()


def normalize(value: Optional[float], min_val: float, max_val: float) -> Optional[float]:
    if value is None:
        return None
    if max_val == min_val:
        return 0.0
    return (value - min_val) / (max_val - min_val)


def process_jsonl_file(input_path: str, output_path: str):
    print("Pass 1: Scanning file to get min/max for Reward_Model and A_Length...")
    reward_model_values = []
    a_length_values = []

    total_lines = sum(1 for _ in open(input_path, 'r', encoding='utf-8'))
    with open(input_path, 'r', encoding='utf-8') as infile:
        for line in tqdm(infile, total=total_lines, desc="Scanning"):
            data = json.loads(line)
            qa_scores = data.get("QA_scores", {})
            if qa_scores.get("Reward_Model") is not None:
                reward_model_values.append(qa_scores["Reward_Model"])
            if qa_scores.get("A_Length") is not None:
                a_length_values.append(qa_scores["A_Length"])

    reward_min, reward_max = (min(reward_model_values), max(
        reward_model_values)) if reward_model_values else (0, 1)
    alen_min, alen_max = (min(a_length_values), max(
        a_length_values)) if a_length_values else (0, 1)

    print(f"Reward_Model min/max: {reward_min}/{reward_max}")
    print(f"A_Length min/max: {alen_min}/{alen_max}")
    print("Pass 2: Normalizing and writing output...")

    # Normalization config
    q_score_ranges = {
        "Clarity": (1, 10), "Coherence": (1, 10), "Completeness": (1, 10),
        "Complexity": (1, 10), "Correctness": (1, 10), "Meaningfulness": (1, 10),
        "OpenThoughts": (1, 10), "Deita_Complexity": (1, 6)
        # Thinking_Prob is skipped
    }

    qa_score_ranges = {
        "Clarity": (1, 10), "Coherence": (1, 10), "Completeness": (1, 10),
        "Complexity": (1, 10), "Correctness": (1, 10), "Meaningfulness": (1, 10),
        "Relevance": (1, 10), "Deita_Quality": (1, 6)
        # IFD, Fail_Rate skipped
    }

    with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        for line in tqdm(infile, total=total_lines, desc="Normalizing"):
            data = json.loads(line)

            # Normalize Q_scores
            q_scores = data.get("Q_scores", {})
            for key, (low, high) in q_score_ranges.items():
                if key in q_scores:
                    q_scores[key] = normalize(q_scores[key], low, high)

            # Normalize QA_scores
            qa_scores = data.get("QA_scores", {})
            for key, (low, high) in qa_score_ranges.items():
                if key in qa_scores:
                    qa_scores[key] = normalize(qa_scores[key], low, high)
            # Special normalization
            if "Reward_Model" in qa_scores:
                qa_scores["Reward_Model"] = normalize(
                    qa_scores["Reward_Model"], reward_min, reward_max)
            if "A_Length" in qa_scores:
                qa_scores["A_Length"] = normalize(
                    qa_scores["A_Length"], alen_min, alen_max)

            data["Q_scores"] = q_scores
            data["QA_scores"] = qa_scores

            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

    print(f"Normalization completed. Output saved to: {output_path}")


if __name__ == '__main__':
    args = parse_args()
    process_jsonl_file(
        input_path=args.input_file,
        output_path=args.output_file
    )
