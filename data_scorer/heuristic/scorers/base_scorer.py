from typing import Dict
from typing import Dict, List
from abc import ABC, abstractmethod
from typing import Dict, List, Any


class BaseScorer(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._validate_config()
        self._setup()

    @abstractmethod
    def _validate_config(self):
        """check the config."""
        pass

    @abstractmethod
    def _setup(self):
        """set up the model, encoder, embedder..."""
        pass

    @abstractmethod
    def score_item(self, data_item: Dict) -> float:
        """score single sample"""
        pass

    @abstractmethod
    def evaluate(self, dataset: List[Dict]) -> List[Dict]:
        """score the whole dataset"""
        pass
