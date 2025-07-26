import asyncio
import orjson as json
from pathlib import Path
from typing import Any, Dict, List, Tuple
from itertools import islice
from datetime import timedelta
import re

import aiofiles
import aiofiles.os

from openai import APIError, AsyncOpenAI, RateLimitError
from rich import print
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    Task,
)
from rich.text import Text


from llm_as_judge.config import AppConfig
from llm_as_judge.utils import load_data
from llm_as_judge.validators import validate_model_response


class AverageTimeRemainingColumn(ProgressColumn):
    """Renders time remaining based on the average speed over the whole task."""

    def render(self, task: "Task") -> Text:
        """Show time remaining based on average speed."""
        if task.total is None or task.completed == 0 or task.elapsed is None:
            return Text("--:--:--", style="progress.remaining")

        # Calculate speed based on the entire task duration
        speed = task.completed / task.elapsed
        if speed == 0:
            return Text("??:??:??", style="progress.remaining")

        remaining = task.total - task.completed
        if remaining <= 0:
            return Text("00:00:00", style="progress.remaining")
            
        remaining_time = remaining / speed
        
        # Format as HH:MM:SS
        time_str = str(timedelta(seconds=int(remaining_time)))
        return Text(time_str, style="progress.remaining")


async def _count_lines_with_wc(file_path: Path) -> int:
    """Uses `wc -l` for a fast line count of a file."""
    command = f"wc -l < \"{file_path}\""
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        raise RuntimeError(f"wc -l failed: {stderr.decode().strip()}")

    try:
        return int(stdout.decode().strip())
    except (ValueError, IndexError):
        raise RuntimeError(f"Could not parse wc -l output: {stdout.decode()}")


class LLMEvaluator:
    def __init__(self, app_config: AppConfig):
        self.config = app_config
        self.client = AsyncOpenAI(
            api_key=self.config.openai.api_key.get_secret_value(),
            base_url=self.config.openai.base_url,
        )
        self.semaphore = asyncio.Semaphore(self.config.concurrency)
        self.prompts: Dict[str, str] = {}
        self.scored_ids = set()
        self.id_track_file_handle = None

    async def _load_scored_ids(self):
        """Loads already scored IDs from the tracking file."""
        if not self.config.id_track_file:
            return
        
        track_file = self.config.id_track_file
        if not track_file.exists():
            print(f"\nℹ️ ID tracking file not found at '{track_file}', creating a new one.")
            track_file.touch()

        async with aiofiles.open(track_file, "r", encoding="utf-8") as f:
            async for line in f:
                stripped_line = line.strip()
                if stripped_line:
                    self.scored_ids.add(stripped_line)

    async def load_prompts(self):
        """Loads all specified prompts from the prompts directory."""
        prompt_dir = self.config.prompts_dir
        for mode, metrics in self.config.metrics.dict(by_alias=True).items():
            for metric in metrics:
                prompt_key = f"{mode.upper()}_{metric}"
                prompt_path = prompt_dir / f"{prompt_key}.txt"
                if not prompt_path.exists():
                    raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
                with open(prompt_path, "r", encoding="utf-8") as f:
                    self.prompts[prompt_key] = f.read()

    async def _evaluate_item(
        self, item: Dict[str, Any], metric: str, mode: str, item_idx: int
    ) -> Tuple[int, str, str, str, Dict[str, Any]]:
        """Evaluates a single item for a given metric and mode, returning the item index, mode, and metric."""
        item_id = str(item.get("id"))
        mode_upper = mode.upper()
        prompt_key = f"{mode_upper}_{metric}"
        prompt_template = self.prompts.get(prompt_key)

        if not prompt_template:
            return item_idx, item_id, metric, mode, {"error": "Prompt not found."}

        # Extract expected keys from the prompt template
        expected_keys = re.findall(r'"(.*?)"\s*:\s*<score_integer>', prompt_template)

        instruction = item.get("instruction")

        content = ""
        if mode_upper == "QA":
            content = prompt_template.format(instruction=instruction, output=item.get("output"))
        elif mode_upper == "Q":
            content = prompt_template.format(instruction=instruction)

        for attempt in range(self.config.retry + 1):
            async with self.semaphore:
                try:
                    response = await self.client.chat.completions.create(
                        model=self.config.model,
                        messages=[{"role": "user", "content": content}],
                        temperature=min(self.config.temperature * (2 ** attempt), 1.0),
                        top_p=self.config.top_p,
                        response_format={"type": "json_object"},
                        timeout=self.config.timeout,
                    )
                    choice = response.choices[0]
                    result_text = choice.message.content
                    parsed_result = json.loads(result_text)

                    validate_model_response(
                        parsed_result, choice.finish_reason, result_text, expected_keys
                    )
                    
                    return item_idx, item_id, metric, mode, parsed_result
                except (RateLimitError, APIError, ValueError) as e:
                    if attempt < self.config.retry:
                        wait_time = 2 ** attempt
                        print(f"⚠️  Error for item #{item_idx}, metric '{metric}'. Retrying in {wait_time}s... (Attempt {attempt + 1}/{self.config.retry}) Reason: {e}")
                        await asyncio.sleep(wait_time)
                    else:
                        # Last attempt failed, return error
                        return item_idx, item_id, metric, mode, {"error": f"Error after {self.config.retry + 1} attempts: {e}"}
                except Exception as e:
                    # For non-retriable errors (e.g., JSON parsing), fail immediately
                    return item_idx, item_id, metric, mode, {"error": f"Non-retriable Evaluation Error: {e}"}
        
        # This part should be unreachable if logic is correct, but as a fallback
        return item_idx, item_id, metric, mode, {"error": "An unknown error occurred after all retries."}

    async def evaluate_file(self, file_path: Path):
        """Evaluates all items in a single data file using chunking and returns the path to the output file."""
        output_file_path = self.config.output_path / f"{file_path.stem}_scored.jsonl"
        error_file_path = self.config.output_path / f"{file_path.stem}_errors.jsonl"

        # Pre-scan to get total number of items for the progress bar
        try:
            total_items = await _count_lines_with_wc(file_path)
            print(f"🚀 Found {total_items} items in [green]{file_path.name}[/green] using 'wc -l'.")
        except (RuntimeError, FileNotFoundError) as e:
            # Fallback for non-Unix systems or if wc fails
            print(f"⚠️ 'wc -l' failed ({e}), falling back to manual line count. This may be slow for large files.")
            with open(file_path, 'r', encoding='utf-8') as f:
                total_items = sum(1 for _ in f)

        data_generator = load_data(file_path)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            AverageTimeRemainingColumn(),
            TimeElapsedColumn(),
            transient=False,  # Keep progress bar after completion
        ) as progress:
            overall_progress = progress.add_task(f"[cyan]Evaluating {file_path.name}", total=total_items)

            async with aiofiles.open(output_file_path, "ab") as f_out, aiofiles.open(error_file_path, "wb") as f_err:
                while True:
                    chunk_items = []
                    try:
                        for _ in range(self.config.chunk_size):
                            chunk_items.append(await data_generator.__anext__())
                    except StopAsyncIteration:
                        pass # Reached the end of the generator
                    
                    if not chunk_items:
                        break

                    tasks = []
                    item_task_counts = {}
                    for item_idx, item in enumerate(chunk_items):
                        item_id = str(item.get("id"))

                        # If tracking is enabled and ID is valid and already scored, skip it.
                        if self.config.id_track_file and item_id and item_id in self.scored_ids:
                            progress.update(overall_progress, advance=1)
                            continue

                        item["scores"] = item.get("scores", {})
                        count = 0
                        if self.config.metrics.q:
                            count += len(self.config.metrics.q)
                            for metric in self.config.metrics.q:
                                tasks.append(self._evaluate_item(item, metric, "Q", item_idx))
                        if self.config.metrics.qa:
                            count += len(self.config.metrics.qa)
                            for metric in self.config.metrics.qa:
                                tasks.append(self._evaluate_item(item, metric, "QA", item_idx))
                        item_task_counts[item_idx] = count

                    write_buffer = []
                    track_id_buffer = []
                    for future in asyncio.as_completed(tasks):
                        item_idx, item_id, metric, mode, result = await future

                        if "error" in result:
                            error_item = {
                                "original_item": chunk_items[item_idx],
                                "metric": metric,
                                "mode": mode,
                                "error_details": result,
                            }
                            await f_err.write(json.dumps(error_item) + b"\n")
                            # We still need to advance the progress bar for failed items
                            item_task_counts[item_idx] -= 1
                            if item_task_counts[item_idx] == 0:
                                progress.update(overall_progress, advance=1)
                            continue

                        mode_key = mode.upper()
                        if mode_key not in chunk_items[item_idx]["scores"]:
                            chunk_items[item_idx]["scores"][mode_key] = {}
                        chunk_items[item_idx]["scores"][mode_key][metric] = result

                        item_task_counts[item_idx] -= 1
                        if item_task_counts[item_idx] == 0:
                            # Item is fully evaluated, add it to the write buffer.
                            item = chunk_items[item_idx]
                            output_item = {
                                "id": item.get("global_index", item.get("id")),
                                "scores": item["scores"],
                            }
                            write_buffer.append(output_item)
                            track_id_buffer.append(item_id)
                            
                            progress.update(overall_progress, advance=1)

                    # Write any remaining items in the buffers after the chunk is processed.
                    if write_buffer:
                        await f_out.write(b"".join(json.dumps(item) + b"\n" for item in write_buffer))
                        if self.id_track_file_handle and track_id_buffer:
                            self.scored_ids.update(track_id_buffer)
                            await self.id_track_file_handle.write("".join(f"{id}\n" for id in track_id_buffer))
                    await f_out.flush()

        print(f"\n✅ Evaluation complete for [green]{file_path.name}[/green]. Results saved to [green]{output_file_path}[/green]")
        return output_file_path

    async def run(self):
        """Runs the full evaluation process."""
        print("\n🔍 Loading prompts...")
        await self.load_prompts()
        print("Prompts loaded successfully.")

        # Load scored IDs if tracking is enabled
        if self.config.id_track_file:
            await self._load_scored_ids()
            print(f"\n📄 Loaded {len(self.scored_ids)} already scored IDs from '{self.config.id_track_file}'.")

        input_path = self.config.input_path
        if not input_path.is_file():
            print(f"❌ Error: Input file not found or is not a file: '{input_path}'")
            return

        if input_path.suffix != ".jsonl":
            print(f"❌ Error: Input file must be a .jsonl file, but got '{input_path}'.")
            return

        print(f"\n🚀 Starting evaluation for file: '{input_path}'.\n")

        try:
            # Open the tracking file in append mode if enabled
            if self.config.id_track_file:
                self.id_track_file_handle = await aiofiles.open(self.config.id_track_file, "a", encoding="utf-8")

            await self.evaluate_file(input_path)

            error_file_path = self.config.output_path / f"{input_path.stem}_errors.jsonl"
            error_count = await _count_lines_with_wc(error_file_path)

            if error_count == 0:
                print("\n[bold blue]🎉 All items were processed successfully with no errors.[/bold blue]")
                # Clean up the empty error file
                await aiofiles.os.remove(error_file_path)
            else:
                print(f"\n[bold yellow]⚠️ Evaluation completed with [bold red]{error_count}[/bold red] errors. Please check '[bold cyan]{error_file_path}[/bold cyan]' for details.[/bold yellow]")
        finally:
            # Ensure the tracking file is closed
            if self.id_track_file_handle:
                await self.id_track_file_handle.close()