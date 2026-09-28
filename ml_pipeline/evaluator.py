from future import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Sequence

LOGGER = logging.getLogger(name)

@dataclass(frozen=True)
class ConfusionMatrix:
  """Counts of true/false positives and negatives."""

true_positive: int
true_negative: int
false_positive: int
false_negative: int

@property
def total(self) -> int:
  """Return the total number of samples."""
  return (
  self.true_positive

  self.true_negative

  self.false_positive

  self.false_negative
  )

@dataclass(frozen=True)
class Metrics:
  """Evaluation metrics for a binary classifier."""

accuracy: float
precision: float
recall: float
f1: float
confusion: ConfusionMatrix

def to_dict(self) -> dict[str, object]:
  """Serialize the metrics to a dictionary."""
  data = asdict(self)
  data["confusion"] = asdict(self.confusion)
  return data

def build_confusion(
  predictions: Sequence[int],
  targets: Sequence[int],
) -> ConfusionMatrix:
  """Compute the confusion matrix for predictions vs. targets."""
  if len(predictions) != len(targets):
    raise ValueError("predictions and targets must have equal length")

tp = tn = fp = fn = 0
for pred, actual in zip(predictions, targets):
  if pred == 1 and actual == 1:
    tp += 1
  elif pred == 0 and actual == 0:
    tn += 1
  elif pred == 1 and actual == 0:
    fp += 1
  else:
    fn += 1

return ConfusionMatrix(tp, tn, fp, fn)

def _safe_divide(numerator: float, denominator: float) -> float:
  """Divide guarding against a zero denominator."""
  return numerator / denominator if denominator else 0.0

def evaluate(
  predictions: Sequence[int],
  targets: Sequence[int],
) -> Metrics:
  """Compute accuracy, precision, recall, and F1 for binary predictions."""
  matrix = build_confusion(predictions, targets)

accuracy = _safe_divide(
  matrix.true_positive + matrix.true_negative, matrix.total
)
precision = _safe_divide(
  matrix.true_positive, matrix.true_positive + matrix.false_positive
)
recall = _safe_divide(
  matrix.true_positive, matrix.true_positive + matrix.false_negative
)
f1 = _safe_divide(2 * precision * recall, precision + recall)

metrics = Metrics(
  accuracy=round(accuracy, 4),
  precision=round(precision, 4),
  recall=round(recall, 4),
  f1=round(f1, 4),
  confusion=matrix,
)
LOGGER.info(
  "Evaluation: accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f",
  metrics.accuracy,
  metrics.precision,
  metrics.recall,
  metrics.f1,
)
return metrics

def format_metrics(metrics: Metrics) -> str:
  """Return a human-readable report string."""
  m = metrics.confusion
  return (
  f"Accuracy : {metrics.accuracy:.4f}\n"
  f"Precision: {metrics.precision:.4f}\n"
  f"Recall : {metrics.recall:.4f}\n"
  f"F1 Score : {metrics.f1:.4f}\n"
  f"Confusion: TP={m.true_positive} TN={m.true_negative} "
  f"FP={m.false_positive} FN={m.false_negative}"
  )
  
