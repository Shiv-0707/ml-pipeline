# ml-pipeline

A production-grade, modular machine learning pipeline framework in pure Python — covering data ingestion, feature engineering, model training, evaluation, and persistence, with a clean package layout, typed APIs, and a full test suite.

https://img.shields.io/badge/Python-3.10%2B-blue
https://img.shields.io/badge/License-MIT-green
https://img.shields.io/badge/tests-unittest-brightgreen

Table of Contents

Overview

Why ml-pipeline

Architecture

Installation

Quick Start

Usage

Configuration

Package Layout

API Reference

Testing

Roadmap

Contributing

License

Overview

ml-pipeline is a small, dependency-light framework that lets you compose reproducible machine-learning workflows out of independent, swappable stages. Instead of a monolithic script, every concern lives in its own module:

Data loading — CSV/JSON ingestion with schema validation.

Preprocessing — scaling, encoding, train/test splitting.

Models — a common estimator interface with multiple implementations.

Evaluation — accuracy, precision, recall, F1, confusion matrix.

Persistence — save/load trained models as JSON.

CLI — drive the whole pipeline without writing code.

Why ml-pipeline

Most ML demos are one giant main.py. ml-pipeline shows what a real project looks like:

Strict separation between data, model, and evaluation layers.

Pure functions at the core, so everything is easy to test.

Typed interfaces via typing.Protocol so estimators are interchangeable.

Deterministic training via explicit random seeds.

Architecture
text
Copy
Download
Raw Data -> Loader -> Preprocessor -> Estimator -> Evaluator -> Persistence

Every arrow is a plain Python object; stages are combined by a Pipeline class.

Installation
bash
Copy
Download
git clone https://github.com/Shiv-0707/ml-pipeline.git
cd ml-pipeline
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

The core has no third-party runtime dependencies.

Quick Start
bash
Copy
Download
python -m ml_pipeline.cli --help
python -m ml_pipeline.cli train --dataset data/sample.csv
Usage

Train and evaluate a model on the bundled sample dataset:

bash
Copy
Download
python -m ml_pipeline.cli train --dataset data/sample.csv

Persist the trained model to disk:

bash
Copy
Download
python -m ml_pipeline.cli train --dataset data/sample.csv --save model.json
Configuration
Flag  Description  Default
--dataset  Path to the CSV dataset  data/sample.csv
--model  Estimator to use (logistic,tree)  logistic
--seed  Random seed for determinism  42
--save  Path to persist the trained model  none
--verbose  Enable debug logging  off
Package Layout
text
Copy
Download
ml-pipeline/
├── ml_pipeline/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── dataset.py
│   ├── evaluator.py
│   ├── features.py
│   ├── models.py
│   ├── pipeline.py
│   └── persistence.py
├── tests/
│   └── test_pipeline.py
├── README.md
└── requirements.txt
API Reference
Dataset

Loads and validates tabular data from CSV.

StandardScaler

Fits a mean/std normaliser and transforms feature matrices.

Estimator

Protocol implemented by LogisticRegression and DecisionTree.

Evaluator

Computes accuracy, precision, recall, F1, and a confusion matrix.

Pipeline

Wires the stages together into a single .run() call.

Testing
bash
Copy
Download
python -m unittest discover -s tests
Roadmap
□ 

Add cross-validation utilities

□ 

Add gradient-boosting estimator

□ 

Add feature-importance reports

□ 

Add GitHub Actions CI

Contributing

Contributions are welcome. Please open an issue before large changes.

License

This project is licensed under the MIT License.
