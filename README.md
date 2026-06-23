# TFT Round Prediction

A machine learning project designed to predict the outcomes of combat rounds in Teamfight Tactics (TFT). This repository contains the full pipeline for raw data processing, feature engineering, and model training. It evaluates multiple machine learning architectures, ranging from a baseline tree-based model (XGBoost) to deep learning models (Convolutional Neural Networks and Vision Transformers). The models process comprehensive game states, including unit placements, item distributions, and active traits.

## Setup

This project uses uv for dependency management.

1. Install dependencies:
   ```bash
   uv sync
   ```
2. Activate the virtual environment:
   ```bash
   # On Windows
   .venv\Scripts\activate

   # On macOS/Linux
   source .venv/bin/activate
   ```

## Baseline

The baseline model take the list of units, traits and the total board costs as input

Extract features from raw data:

```bash
trp extract-baseline-features --raw-path path/to/raw/data --feature-path path/to/features
```

Train & Evaluate baseline XGBoost model:

```bash
trp train-baseline --feature-path path/to/features
```

## CNN

The CNN model takes units and items placement and traits as input

Extract features from raw data:

```bash
trp extract-cnn-features --raw-path path/to/raw/data --feature-path path/to/features
```

Train & Evaluate CNN model:

```bash
trp train-cnn --feature-path path/to/features --batch-size 512 --model-kv dropout=0.2
```

Run hyperparameter optimization:

```bash
trp hpo-cnn --feature-path path/to/features --n-trials 50
```

## ViT

The ViT model takes units and items placement and traits as input

Extract features from raw data:

```bash
trp extract-vit-features --raw-path path/to/raw/data --feature-path path/to/features
```

Train & Evaluate ViT model:

```bash
trp train-vit --feature-path path/to/features --batch-size 512 --model-kv dropout=0.2
```

Run hyperparameter optimization:

```bash
trp hpo-vit --feature-path path/to/features --n-trials 50
```

## Results

Here is a comparison of the different models evaluated on the test set.

| Model | Accuracy |
| :--- | :---: |
| **XGBoost** | 73.0% |
| **CNN** | 79.1% |
| **ViT** | 80.4% |

## References

Ran Cao (Riot Games) "Machine Learning Summit: Simulating Teamfight Tactics Using Deep Learning for Fast Reinforcement Learning AI Training" ([Slides](https://gdcvault.com/play/1028851/Machine-Learning-Summit-Simulating-Teamfight)) ([Video](https://gdcvault.com/play/1029228/Machine-Learning-Summit-Simulating-Teamfight))

Wesley Kerr (Riot Games) "Large-scale deep learning to augment production RL workloads" ([Video](https://www.youtube.com/watch?v=8EsQkFxWYhU))
