import yaml
from pathlib import Path
from pydantic import BaseModel, Field, SecretStr, DirectoryPath
from typing import List, Optional


class ApiConfig(BaseModel):
    api_key: SecretStr
    base_url: Optional[str] = None


class MetricsConfig(BaseModel):
    q: List[str] = Field(default_factory=list, alias="Q")
    qa: List[str] = Field(default_factory=list, alias="QA")

    class Config:
        populate_by_name = True


class AppConfig(BaseModel):
    openai: ApiConfig
    model: str
    concurrency: int = Field(5, ge=1)
    temperature: float = Field(1.0, ge=0.0, le=2.0)
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    timeout: float = Field(20.0, ge=0)
    retry: int = Field(3, ge=0)
    chunk_size: int = Field(1000, ge=1)
    input_path: Path
    output_path: Path
    prompts_dir: DirectoryPath
    id_track_file: Optional[Path] = None
    metrics: MetricsConfig

    class Config:
        # To allow both alias and field name to work for population
        populate_by_name = True


def load_config(config_path: str = "config.yaml") -> AppConfig:
    """Loads configuration from a YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found at '{config_path}'")

    with open(config_file, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)
        
    # Create directories if they don't exist
    Path(config_data.get("output_dir", "output")).mkdir(exist_ok=True)

    return AppConfig(**config_data)
