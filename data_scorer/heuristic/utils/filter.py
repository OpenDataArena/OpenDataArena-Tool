import json
import numpy as np
import argparse
import os
import collections
import math
import random
from typing import Optional


def add_labels_to_jsonl(input_jsonl_path: str, labels_npy_path: str, output_jsonl_path: str, label_key: str = "cluster_label"):
    """
    Add cluster labels sequentially to each line of a JSONL file.
    """
    print("--- Stage 1: Starting label merging ---")
    try:
        labels = np.load(labels_npy_path).astype(int).tolist()
        print(f"Successfully loaded {len(labels)} labels from '{labels_npy_path}'.")
    except FileNotFoundError:
        print(f"Error: Label file not found -> {labels_npy_path}")
        return False

    lines_processed = 0
    try:
        with open(input_jsonl_path, 'r', encoding='utf-8') as f_in, \
                open(output_jsonl_path, 'w', encoding='utf-8') as f_out:
            for i, line in enumerate(f_in):
                if i >= len(labels):
                    print(
                        f"Warning: JSONL file has more lines ({i+1}) than labels ({len(labels)}). Processing stopped early.")
                    break
                data = json.loads(line)
                data[label_key] = labels[i]
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                lines_processed = i + 1

        if lines_processed < len(labels):
            print(f"Warning: Number of labels ({len(labels)}) exceeds JSONL file lines ({lines_processed}).")
        elif lines_processed > 0:
            print(f"Merging complete! Successfully processed {lines_processed} lines to temporary file '{output_jsonl_path}'.")
        return True

    except Exception as e:
        print(f"Error occurred in Stage 1 [Label Merging]: {e}")
        return False


def filter_clusters_by_score(input_jsonl_path: str, output_jsonl_path: str, percentage_to_keep: float, seed=42):
    """
    Filter out high-score data by specified percentage based on scores within each cluster.
    Ensures that each output data entry contains 'id', 'cluster_label', and 'score' keys.

    Args:
        input_jsonl_path (str): Path to JSONL file containing 'cluster_label' and optional 'score'/'id'.
        output_jsonl_path (str): Path to final JSONL file for saving filtered data.
        percentage_to_keep (float): Percentage of data to keep in each cluster (0 to 100).
    """
    print("\n--- Stage 2: Starting score-based filtering within clusters ---")
    if not (0 <= percentage_to_keep <= 100):
        print("Error: Percentage must be between 0 and 100.")
        return

    # 1. Group all data by cluster_label into memory
    clusters = collections.defaultdict(list)
    print(f"Reading and grouping data from '{input_jsonl_path}'...")
    try:
        with open(input_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)

                # --- Modification 1: Ensure 'score' and 'id' keys exist ---
                # Use .get() method to set default value 0 for 'score'
                data['score'] = data.get('score', 0)
                # Use .get() method to set default value "0" (string type) for 'id'
                data['id'] = data.get('id', "0") # <--- New modification

                # Assume label key is 'cluster_label'
                clusters[data['cluster_label']].append(data)
    except FileNotFoundError:
        print(f"Error: Input file '{input_jsonl_path}' not found. Please ensure Stage 1 was executed successfully.")
        return
    except KeyError:
        print(f"Error: Some lines in the file are missing the 'cluster_label' key.")
        return

    print(f"Data loading complete. Found {len(clusters)} clusters.")

    # 2. Iterate through each cluster, sort, filter, and collect results
    if seed is not None:
        random.seed(seed)
    all_filtered_data = []
    print(f"Filtering top {percentage_to_keep}% highest scoring data from each cluster...")
    for label, items in sorted(clusters.items()):
        if not items:
            print(f"  - Cluster {label}: Empty cluster, skipping.")
            continue
        # Since we guarantee the 'score' key exists, we can access it directly here
        sorted_items = sorted(
            items, key=lambda x: x['score'], reverse=True)

        # b. Calculate the number of data entries to keep
        num_to_keep = math.ceil(
            len(sorted_items) * (percentage_to_keep / 100.0))

        # c. If all scores are the same, randomly sample; otherwise, truncate normally by score
        all_equal = (sorted_items[0]['score'] == sorted_items[-1]['score'])

        if all_equal:
            # All scores equal: use random sampling to avoid truncating by original order
            top_items = random.sample(sorted_items, k=num_to_keep)
        else:
            # Normal: still truncate by score
            top_items = sorted_items[:num_to_keep]

        all_filtered_data.extend(top_items)
        print(f"  - Cluster {label}: Total {len(items)} entries, filtered to {len(top_items)} entries.")

    # 3. Write all filtered data to the final output file
    print(f"\nFiltering complete. Total of {len(all_filtered_data)} data entries retained.")
    print(f"Writing to final file '{output_jsonl_path}'...")
    with open(output_jsonl_path, 'w', encoding='utf-8') as f_out:
        for item in all_filtered_data:
            # At this point, the item object must contain 'id' and 'score' keys
            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')

    print("All processing complete!")

def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Merge cluster labels into JSONL file and filter specified percentage of data based on scores within each cluster."
    )
    parser.add_argument(
        "--input_file",
        default='/mnt/hwfile/opendatalab/air/data/data-leaderboard/dolly/train.jsonl',
        help="Path to the original JSONL file (e.g., train.jsonl)"
    )
    parser.add_argument(
        "--labels_file",
        default='/mnt/petrelfs/gaoxin/DataLeaderBoardFittingAlgorithm/minibatch/cluster_labels.npy',
        help="Path to the .npy file containing cluster labels (e.g., cluster_labels.npy)"
    )
    parser.add_argument(
        "--output_file",
        default="/mnt/petrelfs/gaoxin/DataLeaderBoardFittingAlgorithm/minibatch/data.jsonl",
        help="Path to the JSONL file for saving final filtered data (e.g., train_filtered.jsonl)"
    )
    parser.add_argument(
        "--percentage",
        type=float,
        default=30.0,
        help="Percentage of data to keep in each cluster by score from high to low. Default: 20%%."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling if all scores are the same. Default: 42."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # To avoid overwriting the original file and keep the process clear, we create a temporary file for storing intermediate results
    # The temporary file name is based on the output file name
    output_dir = os.path.dirname(args.output_file)
    output_basename = os.path.basename(args.output_file)
    temp_file_path = os.path.join(output_dir, f"temp_{output_basename}")

    # Stage 1: Merge labels to temporary file
    # If merging succeeds, continue to Stage 2
    if add_labels_to_jsonl(args.input_file, args.labels_file, temp_file_path):

        # Stage 2: Read from temporary file, filter, and save to final file
        filter_clusters_by_score(
            temp_file_path, args.output_file, args.percentage, args.seed)

        # Clean up temporary file
        try:
            print(f"\nDeleting temporary file '{temp_file_path}'...")
            os.remove(temp_file_path)
            print("Temporary file deleted.")
        except OSError as e:
            print(f"Error deleting temporary file: {e}")
