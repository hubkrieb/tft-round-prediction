# TFT Round Prediction

Project to build and train ML models that can predict TFT round outcomes.

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

## Results



## References

Ran Cao (Riot Games) "Machine Learning Summit: Simulating Teamfight Tactics Using Deep Learning for Fast Reinforcement Learning AI Training" ([Slides](https://gdcvault.com/play/1028851/Machine-Learning-Summit-Simulating-Teamfight)) ([Video](https://gdcvault.com/play/1029228/Machine-Learning-Summit-Simulating-Teamfight))
