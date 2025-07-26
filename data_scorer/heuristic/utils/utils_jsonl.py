from typing import List
import json
from tqdm import tqdm
import os
import sys


def save_jsonl(data, path):
    if '/' in path:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for sample in tqdm(data):
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def load_jsonl(path, max_lines=None):
    data = []
    skip_count = 0

    # Improve integer-to-string conversion limits (Python 3.11+)
    sys.set_int_max_str_digits(0)

    try:
        with open(path, "r", encoding="utf-8", errors='ignore') as f:
            for i, line in enumerate(tqdm(f), start=1):
                if max_lines is not None and i > max_lines:
                    break
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError as e:
                    print(f"[Warning] JSON decode error at line {i}: {e}")
                    skip_count += 1
                    continue
                except Exception as e:
                    print(f"[Warning] Unexpected error at line {i}: {e}")
                    skip_count += 1
                    continue
    except FileNotFoundError:
        print(f"Error: File not found at {path}")
        return []
    except Exception as e:
        print(
            f"An unexpected error occurred while opening/reading the file: {e}")
        return []

    print(
        f"Total lines processed: {len(data) + skip_count}, Successfully loaded: {len(data)}, Skipped: {skip_count}")
    return data


def merge_multiple_scores(file_a_path: str, b_file_paths: List[str], output_path: str, verbose=False):
    """
    Merge multiple JSONL score files (B) into the main JSONL file (A), updating fields in Q_scores or QA_scores.

    :param file_a_path: Path to A file (main data)
    :param b_file_paths: List of B file paths (each containing id + one or more score fields)
    :param output_path: Path to the merged output JSONL file
    :param verbose: If True, prints warnings for unmatched score fields
    """

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Step 1: Load all score data from B files
    print("Step 1: Loading scores from all scorers' output files...")
    id_to_scores = {}

    for b_path in b_file_paths:
        print(f"  Reading {b_path}")
        with open(b_path, 'r', encoding='utf-8') as fb:
            for line in fb:
                data = json.loads(line)
                entry_id = data.get("id")
                if entry_id is None:
                    continue
                for key, value in data.items():
                    if key != "id":
                        id_to_scores.setdefault(entry_id, {})[key] = value

    print(f"Collected scores for {len(id_to_scores)} unique IDs.")

    # Step 2: Read A and insert scores
    print("Step 2: Processing original input file and inserting scores...")

    total = 0
    updated = 0

    with open(file_a_path, 'r', encoding='utf-8') as fa, \
            open(output_path, 'w', encoding='utf-8') as fout, \
            tqdm(desc="Progress", unit=" lines") as pbar:

        for line in fa:
            total += 1
            data = json.loads(line)
            entry_id = data.get("id")

            if entry_id in id_to_scores:
                for score_key, score_value in id_to_scores[entry_id].items():
                    inserted = False
                    if score_key in data.get("Q_scores", {}):
                        data["Q_scores"][score_key] = score_value
                        inserted = True
                    elif score_key in data.get("QA_scores", {}):
                        data["QA_scores"][score_key] = score_value
                        inserted = True
                    elif verbose:
                        print(
                            f"[Warning] id={entry_id} — field '{score_key}' not found in Q_scores or QA_scores")

                    if inserted:
                        updated += 1

            fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            pbar.update(1)


def merge_jsonl_files(input_paths, output_path):
    """
    Merge multiple JSONL files into a single JSONL file.

    Args:
        input_paths (List[str]): List of paths to the .jsonl files to be merged.
        output_path (str): Path to the output .jsonl file after merging.
    """
    with open(output_path, 'w', encoding='utf-8') as fout:
        for path in input_paths:
            with open(path, 'r', encoding='utf-8') as fin:
                for line in fin:
                    if line.strip():
                        fout.write(line)


def add_evaluation_keys_to_jsonl(input_path: str, output_path: str):

    ds = load_jsonl(input_path)
    ds_add_key = []

    for idx, item in enumerate(ds):
        assert "instruction" in item and "output" in item, f"item {idx} is not valid"
        item["id"] = idx
        item["Q_scores"] = {
            'Clarity': None,
            'Coherence': None,
            'Completeness': None,
            'Complexity': None,
            'Correctness': None,
            'Meaningfulness': None,
            'Difficulty': None,
            'Deita_Complexity': None,
            'Thinking_Prob': None,
        }
        item["QA_scores"] = {
            'Clarity': None,
            'Coherence': None,
            'Completeness': None,
            'Complexity': None,
            'Correctness': None,
            'Meaningfulness': None,
            'Relevance': None,
            'IFD': None,
            'Deita_Quality': None,
            'Reward_Model': None,
            'A_Length': None,
            'Fail_Rate': None,
        }
        ds_add_key.append(item)
    save_jsonl(ds_add_key, output_path)
