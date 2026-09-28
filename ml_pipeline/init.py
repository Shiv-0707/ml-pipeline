from future import annotations

"""ml_pipeline - a modular, production-style machine learning pipeline framework.

This package provides composable building blocks for reproducible ML workflows:

dataset - CSV loading and validation

features - scaling, encoding, and transformation helpers

models - estimator implementations

evaluator - binary classification metrics

pipeline - high-level orchestration

persistence - model serialization
"""

from .dataset import Dataset, DatasetError, load_csv, iter_batches
from .evaluator import ConfusionMatrix, Metrics, evaluate, format_metrics
from .features import (
MinMaxScaler,
StandardScaler,
add_bias_column,
one_hot,
select_columns,
)
from .models import (
DecisionStump,
Estimator,
LogisticRegression,
build_estimator,
)

all = [
  "Dataset",
  "DatasetError",
  "load_csv",
  "iter_batches",
  "ConfusionMatrix",
  "Metrics",
  "evaluate",
  "format_metrics",
  "MinMaxScaler",
  "StandardScaler",
  "add_bias_column",
  "one_hot",
  "select_columns",
  "DecisionStump",
  "Estimator",
  "LogisticRegression",
  "build_estimator",
]

version = "1.0.0"
