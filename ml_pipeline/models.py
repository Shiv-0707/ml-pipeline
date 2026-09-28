from future import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Protocol, Sequence

LOGGER = logging.getLogger(name)

Vector = Sequence[float]
Matrix = Sequence[Sequence[float]]

class Estimator(Protocol):
  """Protocol implemented by all estimators in this package."""

def fit(self, features: Matrix, targets: Sequence[int]) -> None:
  """Fit the estimator to the given data."""
  ...

def predict(self, features: Matrix) -> list[int]:
  """Return predicted labels for the given features."""
  ...

def dot(a: Vector, b: Vector) -> float:
  """Return the dot product of two equal-length vectors."""
  if len(a) != len(b):
    raise ValueError("vectors must have equal length")
    return sum(x * y for x, y in zip(a, b))

def sigmoid(z: float) -> float:
  """Numerically stable logistic sigmoid."""
  if z >= 0:
    return 1.0 / (1.0 + math.exp(-z))
    exp_z = math.exp(z)
    return exp_z / (1.0 + exp_z)

@dataclass
class LogisticRegression:
  """A simple binary logistic regression trained with gradient descent."""

learning_rate: float = 0.1
epochs: int = 200
weights: list[float] = field(default_factory=list)
bias: float = 0.0

def _init_params(self, n_features: int) -> None:
  """Initialise weights and bias deterministically."""
  rng = random.Random(42)
  self.weights = [rng.uniform(-0.01, 0.01) for _ in range(n_features)]
  self.bias = 0.0

def fit(self, features: Matrix, targets: Sequence[int]) -> None:
  """Train the model using batch gradient descent."""
  if not features:
    raise ValueError("no training data provided")

n_samples = len(features)
n_features = len(features[0])
self._init_params(n_features)

for epoch in range(self.epochs):
  grad_w = [0.0] * n_features
  grad_b = 0.0

for row, target in zip(features, targets):
  prediction = sigmoid(dot(self.weights, row) + self.bias)
  error = prediction - target
  for j in range(n_features):
    grad_w[j] += error * row[j]
    grad_b += error

self.weights = [
  w - self.learning_rate * (gw / n_samples)
  for w, gw in zip(self.weights, grad_w)
]
self.bias -= self.learning_rate * (grad_b / n_samples)

if epoch % 50 == 0:
  LOGGER.debug("Epoch %d: bias=%.4f", epoch, self.bias)

def predict_proba(self, features: Matrix) -> list[float]:
  """Return the predicted probability of the positive class."""
  return [sigmoid(dot(self.weights, row) + self.bias) for row in features]

def predict(self, features: Matrix) -> list[int]:
  """Return hard class predictions."""
  return [1 if p >= 0.5 else 0 for p in self.predict_proba(features)]

def to_dict(self) -> dict[str, object]:
  """Serialize the model parameters to a dictionary."""
  return {
  "type": "logistic",
  "weights": self.weights,
  "bias": self.bias,
  }

@dataclass
class DecisionStump:
  """A single-feature decision stump classifier."""

feature_index: int = 0
threshold: float = 0.0
polarity: int = 1

def fit(self, features: Matrix, targets: Sequence[int]) -> None:
  """Greedily choose the best split on a single feature."""
  best_accuracy = -1.0
  n_features = len(features[0])

for feature in range(n_features):
  values = [row[feature] for row in features]
  for threshold in sorted(set(values)):
    for polarity in (1, -1):
      predictions = [
        1 if (row[feature] - threshold) * polarity >= 0 else 0
        for row in features
      ]
      accuracy = sum(
      p == t for p, t in zip(predictions, targets)
      ) / len(targets)
      if accuracy > best_accuracy:
        best_accuracy = accuracy
        self.feature_index = feature
        self.threshold = threshold
        self.polarity = polarity

LOGGER.debug(
  "Best stump: feature=%d threshold=%.4f polarity=%d accuracy=%.4f",
  self.feature_index,
  self.threshold,
  self.polarity,
  best_accuracy,
)

def predict(self, features: Matrix) -> list[int]:
  """Return predictions for the chosen stump."""
  return [
  1 if (row[self.feature_index] - self.threshold) * self.polarity >= 0 else 0
  for row in features
  ]

def to_dict(self) -> dict[str, object]:
  """Serialize the stump to a dictionary."""
  return {
  "type": "stump",
  "feature_index": self.feature_index,
  "threshold": self.threshold,
  "polarity": self.polarity,
  }

def build_estimator(name: str) -> Estimator:
  """Factory returning an estimator by name."""
  registry = {
  "logistic": LogisticRegression,
  "stump": DecisionStump,
  "tree": DecisionStump,
  }
  try:
    return registryname
  except KeyError as exc: # pragma: no cover - defensive branch
  raise ValueError(f"unknown estimator: {name!r}") from exc
  
