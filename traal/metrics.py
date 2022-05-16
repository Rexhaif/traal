from datasets import load_metric
from typing import Dict, List
import numpy as np


class BaseMetrics:

    def compute_metrics(self, p) -> Dict[str, float]:
        pass


class SeqEvalMetrics(BaseMetrics):

    def __init__(self, id2label: List[str]) -> None:
        super().__init__()
        self.id2label = id2label
        self.metric = load_metric("seqeval")


    def compute_metrics(self, p) -> Dict[str, float]:
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [self.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = self.metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }