import orjson as json
from pathlib import Path
from typing import Dict, Any, AsyncGenerator

import aiofiles


async def load_data(file_path: Path) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Asynchronously reads a JSONL file and yields each item.
    It expects one JSON object per line.
    """
    async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
        async for line in f:
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed line in {file_path}: {line.strip()}")
                    continue
