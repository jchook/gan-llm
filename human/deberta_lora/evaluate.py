import evaluate
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import datasets

def load_metric():
    return CustomMetrics()

_DESCRIPTION = """\
Custom metric module to calculate F1, accuracy, precision, and recall for a classification task.
"""

class CustomMetrics(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("int64"),
                    "references": datasets.Value("int64"),
                }
            ),
            codebase_urls=[],
            reference_urls=[],
            format="numpy",
            citation="",
        )

    def _compute(self, predictions, references):
        accuracy = accuracy_score(references, predictions)
        f1 = f1_score(references, predictions, average='weighted')  # or 'macro' if preferred
        precision = precision_score(references, predictions, average='weighted')
        recall = recall_score(references, predictions, average='weighted')

        return {
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }

