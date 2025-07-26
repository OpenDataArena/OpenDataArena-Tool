import argparse
import os
import orjson as json
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
    FileSizeColumn,
    TotalFileSizeColumn,
)


def process_scores(scores_file, data_file, output_file):
    """
    Updates scores from a scores file into a data file and writes to an output file.

    :param scores_file: Path to the JSONL file containing scores.
    :param data_file: Path to the original data JSONL file with null scores.
    :param output_file: Path for the new file to save the results.
    """

    progress_columns = (
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        "•",
        FileSizeColumn(),
        "/",
        TotalFileSizeColumn(),
        "•",
        TimeRemainingColumn(),
        "•",
        TransferSpeedColumn(),
    )

    # 1. Read scores file and store in a dictionary
    print(f"📄 Processing scores file: {scores_file}")
    scores_map = {}
    file_size = os.path.getsize(scores_file)
    with Progress(*progress_columns) as progress:
        task = progress.add_task(f"[cyan]Reading scores...", total=file_size)
        with open(scores_file, "r", encoding="utf-8") as f:
            for line in f:
                score_data = json.loads(line)
                scores_map[score_data["id"]] = score_data["scores"]
                progress.update(task, advance=len(line.encode("utf-8")))

    # 2. Read original data file line by line, update scores, and write to output file
    print(f"\n📄 Processing data file and merging scores: {data_file}")
    file_size = os.path.getsize(data_file)
    with Progress(*progress_columns) as progress:
        task = progress.add_task(f"[cyan]Merging scores...", total=file_size)
        with open(data_file, "r", encoding="utf-8") as f_data, open(
            output_file, "w", encoding="utf-8"
        ) as f_out:
            for line in f_data:
                data_item = json.loads(line)
                item_id = data_item.get("id")

                if item_id in scores_map:
                    scores = scores_map[item_id]

                    # Update Q_scores
                    if "Q" in scores and "All" in scores["Q"]:
                        for key, value in scores["Q"]["All"].items():
                            if key == "Code_Difficulty" or key == "Math_Difficulty":
                                data_item["Q_scores"]["Difficulty"] = value
                            if key in data_item["Q_scores"]:
                                data_item["Q_scores"][key] = value

                    # Update QA_scores
                    if "QA" in scores and "All" in scores["QA"]:
                        for key, value in scores["QA"]["All"].items():
                            if key in data_item["QA_scores"]:
                                data_item["QA_scores"][key] = value

                # Write updated data to new file
                # orjson.dumps() returns bytes, so we decode to string.
                f_out.write(json.dumps(data_item).decode("utf-8") + "\n")
                progress.update(task, advance=len(line.encode("utf-8")))

    print(f"\n✅ Processing complete. Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Processes JSONL files to update scores into a main data file."
    )
    parser.add_argument(
        "--scores_file",
        type=str,
        required=True,
        help="Path to the JSONL file containing scores.",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="Path to the original data JSONL file with null scores.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path for the new file to save the results.",
    )

    args = parser.parse_args()

    process_scores(args.scores_file, args.data_file, args.output_file) 