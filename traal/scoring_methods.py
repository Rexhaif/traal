from cmath import log
import transformers as tr
import numpy as np
from tqdm.auto import tqdm
from rich.logging import RichHandler
import logging
from datasets import Dataset
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)

logger = logging.getLogger(__name__)

class BaseScorer:

    def __init__(self, model_path: str, task: str = 'ner') -> None:
        self.model_path = model_path
        self.task = task

    def compute_score(self, dataset: Dataset) -> np.ndarray:
        pass


class MNLPScorer(BaseScorer):

    def __init__(self, model_path: str, task: str = 'ner') -> None:
        super().__init__(model_path, task)
        logger.info("Loading tokenzier")
        self.tokenizer = tr.AutoTokenizer.from_pretrained(self.model_path)
        if self.task == "ner":
            logger.info("Loading ner model")
            self.model = tr.AutoModelForTokenClassification.from_pretrained(self.model_path)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def compute_score(self, dataset: Dataset) -> np.ndarray:
        