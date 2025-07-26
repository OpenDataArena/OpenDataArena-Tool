import asyncio
import os
import typer
from typing_extensions import Annotated
from pathlib import Path

from llm_as_judge.config import load_config
from llm_as_judge.evaluator import LLMEvaluator

from tools.add_empty_key import add_empty_key


def main(
    config_path: Annotated[
        str,
        typer.Option(
            "--config-path",
            "-c",
            help="Path to the configuration file.",
        ),
    ] = "config.yaml",
):
    """
    LLM-as-Judge: Automated evaluation using language models.
    """
    try:
        # Load configuration into a local variable
        app_config = load_config(config_path)
        
        print("🚀 Starting LLM-as-Judge evaluation...")
        print(f"Loaded configuration from: {config_path}")
        print(f"Model: {app_config.model}, Concurrency: {app_config.concurrency}\n")
        
        print("🔑 Adding empty key...")

        input_filename = os.path.basename(app_config.input_path)

        add_empty_key(app_config.input_path, os.path.join(app_config.output_path, input_filename))
        app_config.input_path = Path(os.path.join(app_config.output_path, input_filename))
        
        # Pass the configuration to the evaluator
        evaluator = LLMEvaluator(app_config)
        asyncio.run(evaluator.run())

    except FileNotFoundError:
        print(f"❌ Error: Configuration file not found at '{config_path}'")
        raise typer.Exit(code=1)
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        raise typer.Exit(code=1)


def cli():
    """Typer entry point."""
    typer.run(main)


if __name__ == "__main__":
    cli() 