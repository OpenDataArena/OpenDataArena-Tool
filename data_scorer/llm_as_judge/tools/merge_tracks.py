import typer
import orjson as json
from pathlib import Path
from typing import Set, List
from typing_extensions import Annotated
import os
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
    FileSizeColumn,
    TotalFileSizeColumn,
)


def extract_ids_from_jsonl(file_path: Path, id_set: Set[str]):
    """Extracts IDs from a .jsonl file with a progress bar."""
    print(f"📄 Processing JSONL file: {file_path}")
    count = 0
    malformed_lines = 0
    file_size = os.path.getsize(file_path)

    with Progress(
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
    ) as progress:
        task = progress.add_task(f"[cyan]Reading...", total=file_size)
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped_line = line.strip()
                if stripped_line:
                    try:
                        data = json.loads(stripped_line)
                        item_id = str(data.get("global_index", data.get("id", "")))
                        if item_id:
                            id_set.add(item_id)
                            count += 1
                    except json.JSONDecodeError:
                        malformed_lines += 1
                progress.update(task, advance=len(line.encode("utf-8")))

    print(f"  -> Found {count} IDs.")
    if malformed_lines > 0:
        print(f"  ⚠️ Warning: Skipped {malformed_lines} malformed JSON line(s).")


def extract_ids_from_txt(file_path: Path, id_set: Set[str]):
    """Extracts IDs from a plain text file with a progress bar."""
    print(f"📄 Processing text track file: {file_path}")
    count = 0
    file_size = os.path.getsize(file_path)

    with Progress(
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
    ) as progress:
        task = progress.add_task(f"[cyan]Reading...", total=file_size)
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                item_id = line.strip()
                if item_id:
                    id_set.add(item_id)
                    count += 1
                progress.update(task, advance=len(line.encode("utf-8")))

    print(f"  -> Found {count} IDs.")


def main(
    input_files: Annotated[
        List[Path],
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="A list of input files to process. Can be .jsonl or .txt track files.",
        ),
    ],
    output_file: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            writable=True,
            resolve_path=True,
            help="Path to the unified output track file.",
        ),
    ],
):
    """
    Merges multiple track ID files (.txt) or scored result files (.jsonl)
    into a single, unified track file with unique IDs.
    """
    print("🚀 Starting track file merge process...")

    unique_ids: Set[str] = set()

    for file_path in input_files:
        if not file_path.is_file():
            print(f"⚠️ Warning: Skipping non-file path: {file_path}")
            continue

        if file_path.suffix == ".jsonl":
            extract_ids_from_jsonl(file_path, unique_ids)
        else:
            extract_ids_from_txt(file_path, unique_ids)

    print(f"\n✨ Total unique IDs collected: {len(unique_ids)}")

    # Write all unique IDs to the output file
    try:
        with output_file.open("w", encoding="utf-8") as f:
            for item_id in sorted(list(unique_ids)):  # Sort for consistent output
                f.write(f"{item_id}\n")
        print(f"✅ Successfully merged all IDs into: {output_file}")
    except IOError as e:
        print(f"❌ Error: Could not write to output file {output_file}. Reason: {e}")
        raise typer.Exit(code=1)


def cli():
    """Typer entry point."""
    typer.run(main)


if __name__ == "__main__":
    cli() 