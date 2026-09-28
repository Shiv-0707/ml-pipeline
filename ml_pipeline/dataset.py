from future import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

LOGGER = logging.getLogger(name)

Matrix = list[list[float]]

class DatasetError(ValueError):
  """Raised when a dataset cannot be loaded or validated."""

@dataclass(frozen=True)
class Dataset:
  """An immutable, validated tabular dataset."""

features: Matrix
targets: list[int]
feature_names: tuple[str, ...]

def len(self) -> int:
  """Return the number of rows in the dataset."""
  return len(self.features)

@property
def n_features(self) -> int:
  """Return the number of feature columns."""
  return len(self.feature_names)

def split(self, test_ratio: float = 0.2, seed: int = 42) -> tuple[Dataset, Dataset]:
  """Split the dataset into train and test partitions."""
  if not 0.0 < test_ratio < 1.0:
    raise DatasetError("test_ratio must be between 0 and 1")

import random

indices = list(range(len(self)))
random.Random(seed).shuffle(indices)
cutoff = int(len(indices) * (1 - test_ratio))

train_idx, test_idx = indices[:cutoff], indices[cutoff:]
return (
  Dataset(
    features=[self.features[i] for i in train_idx],
    targets=[self.targets[i] for i in train_idx],
    feature_names=self.feature_names,
  ),
  Dataset(
    features=[self.features[i] for i in test_idx],
    targets=[self.targets[i] for i in test_idx],
    feature_names=self.feature_names,
  ),
)

def summary(self) -> dict[str, object]:
  """Return a summary of the dataset dimensions."""
  positives = sum(self.targets)
  return {
  "rows": len(self),
  "features": self.n_features,
  "positive_labels": positives,
  "negative_labels": len(self) - positives,
  }

def _parse_row(row: Sequence[str], line_number: int) -> tuple[list[float], int]:
  """Parse a single CSV row into features and a target."""
  if len(row) < 2:
    raise DatasetError(f"line {line_number}: expected at least 2 columns")
    try:
      features = [float(value) for value in row[:-1]]
      target = int(float(row[-1]))
    except ValueError as exc:
      raise DatasetError(f"line {line_number}: invalid numeric value") from exc
      if target not in (0, 1):
        raise DatasetError(f"line {line_number}: target must be 0 or 1")
        return features, target

def load_csv(path: Path) -> Dataset:
  """Load a dataset from a CSV file with a header row."""
  if not path.exists():
    raise DatasetError(f"file not found: {path}")

with path.open(newline="", encoding="utf-8") as handle:
  reader = csv.reader(handle)
  try:
    header = next(reader)
  except StopIteration as exc:
    raise DatasetError("empty CSV file") from exc

feature_names = tuple(name.strip() for name in header[:-1])
features: Matrix = []
targets: list[int] = []

for line_number, row in enumerate(reader, start=2):
  if not row:
    continue
    row_features, target = _parse_row(row, line_number)
    features.append(row_features)
    targets.append(target)

if not features:
  raise DatasetError("no data rows found")

widths = {len(row) for row in features}
if len(widths) != 1:
  raise DatasetError("inconsistent number of columns across rows")

LOGGER.info("Loaded dataset: %d rows, %d features", len(features), len(feature_names))
return Dataset(features=features, targets=targets, feature_names=feature_names)

def iter_batches(dataset: Dataset, batch_size: int) -> Iterator[Dataset]:
  """Yield successive mini-batches from the dataset."""
  if batch_size <= 0:
    raise DatasetError("batch_size must be positive")
    for start in range(0, len(dataset), batch_size):
      stop = start + batch_size
      yield Dataset(
      features=dataset.features[start:stop],
      targets=dataset.targets[start:stop],
      feature_names=dataset.feature_names,
      )
      
