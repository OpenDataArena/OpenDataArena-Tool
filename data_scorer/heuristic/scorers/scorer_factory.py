import importlib.util
import json
from typing import Dict


class ScorerFactory:
    _scorer_classes = {}

    @classmethod
    def load_scorers(cls, config_path: str):
        with open(config_path, 'r') as f:
            config_data = json.load(f)
            for scorer in config_data['scorers']:
                module = scorer['module']
                name = scorer['name']
                try:
                    # dynamic import scorer class
                    module = importlib.import_module(module)
                    scorer_class = getattr(module, name)
                    cls._scorer_classes[name] = scorer_class
                except (ModuleNotFoundError, AttributeError) as e:
                    raise ValueError(
                        f"Failed to load scorer '{name}' from '{module}': {e}")

    @classmethod
    def create(cls, name: str, config: Dict):
        if name not in cls._scorer_classes:
            raise ValueError(f"Unsupported scorer: {name}")
        return cls._scorer_classes[name](config)
