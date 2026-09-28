from future import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Sequence

LOGGER = logging.getLogger(name)

Matrix = list[list[float]]

class FeatureError(ValueError):
  """Raised when feature transformation fails."""

@dataclass
class StandardScaler:
  """Standardises features by removing the mean and scaling to unit variance."""

means: list[float] = field(default_factory=list)
stds: list[float] = field(default_factory=list)

def fit(self, features: Matrix) -> None:
  """Compute per-column mean and standard deviation."""
  if not features:
    raise FeatureError("cannot fit on empty data")

n_features = len(features[0])
self.means = [0.0] * n_features
self.stds = [0.0] * n_features

for row in features:
  for j in range(n_features):
    self.means[j] += row[j]
    self.means = [m / len(features) for m in self.means]

for row in features:
  for j in range(n_features):
    diff = row[j] - self.means[j]
    self.stds[j] += diff * diff
    self.stds = [
    math.sqrt(s / len(features)) or 1.0 for s in self.stds
    ]
    LOGGER.debug("Fitted scaler on %d features", n_features)

def transform(self, features: Matrix) -> Matrix:
  """Apply the learned normalisation to the features."""
  if not self.means:
    raise FeatureError("scaler must be fitted before transform")
    return [
    [(value - self.means[j]) / self.stds[j] for j, value in enumerate(row)]
    for row in features
    ]

def fit_transform(self, features: Matrix) -> Matrix:
  """Fit the scaler and return the transformed features."""
  self.fit(features)
  return self.transform(features)

def to_dict(self) -> dict[str, object]:
  """Serialize the scaler parameters."""
  return {"means": self.means, "stds": self.stds}

@dataclass
class MinMaxScaler:
  """Scales features into the [0, 1] range."""

minimums: list[float] = field(default_factory=list)
maximums: list[float] = field(default_factory=list)

def fit(self, features: Matrix) -> None:
  """Record the minimum and maximum of each column."""
  if not features:
    raise FeatureError("cannot fit on empty data")
    n_features = len(features[0])
    self.minimums = [float("inf")] * n_features
    self.maximums = [float("-inf")] * n_features

for row in features:
  for j in range(n_features):
    self.minimums[j] = min(self.minimums[j], row[j])
    self.maximums[j] = max(self.maximums[j], row[j])

def transform(self, features: Matrix) -> Matrix:
  """Apply min-max scaling."""
  if not self.minimums:
    raise FeatureError("scaler must be fitted before transform")
    result: Matrix = []
    for row in features:
      scaled = []
      for j, value in enumerate(row):
        span = self.maximums[j] - self.minimums[j]
        scaled.append((value - self.minimums[j]) / span if span else 0.0)
        result.append(scaled)
        return result

def fit_transform(self, features: Matrix) -> Matrix:
  """Fit and transform in one step."""
  self.fit(features)
  return self.transform(features)

def one_hot(indices: Sequence[int], n_classes: int) -> Matrix:
  """Convert a sequence of class indices to one-hot vectors."""
  encoded: Matrix = []
  for index in indices:
    if not 0 <= index < n_classes:
      raise FeatureError(f"index {index} out of range for {n_classes} classes")
      vector = [0.0] * n_classes
      vector[index] = 1.0
      encoded.append(vector)
      return encoded

def add_bias_column(features: Matrix) -> Matrix:
  """Prepend a constant 1.0 column to each feature row."""
  return [[1.0, *row] for row in features]

def select_columns(features: Matrix, indices: Sequence[int]) -> Matrix:
  """Return a copy of the features with only the selected columns."""
  return [[row[i] for i in indices] for row in features]

def build_scaler(name: str):
  """Factory returning a scaler by name."""
  registry = {
  "standard": StandardScaler,
  "minmax": MinMaxScaler,
  }
  try:
    return registryname
  except KeyError as exc:
    raise FeatureError(f"unknown scaler: {name!r}") from exc
    
