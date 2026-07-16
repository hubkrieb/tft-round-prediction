# TFT Round Prediction

A machine learning project designed to predict the outcomes of combat rounds in Teamfight Tactics (TFT). This repository contains the full pipeline for raw data processing, feature engineering, and model training. It evaluates multiple machine learning architectures, ranging from a baseline tree-based model (XGBoost) to deep learning models (Convolutional Neural Networks and Vision Transformers). The models process comprehensive game states, including unit placements, item distributions, and active traits.

### Live demo available at: [tft-round-prediction.com](https://tft-round-prediction.com)

![Board builder with a full matchup and the ViT-predicted win probability](assets/app_screen.png)

## Project structure

The code is split into three clearly separated concerns:

```
src/
├── training/     # Model training (pre-existing): feature extraction, datasets,
│   ├── baseline/ #   models and trainers for the XGBoost / CNN / ViT pipelines.
│   ├── cnn/
│   ├── vit/
│   └── utils/    #   static data + vocabularies shared by the extractors.
├── api/          # Inference backend (FastAPI): featurize a board and predict.
│   ├── config.py, schema.py, featurize.py, predictor.py, app.py, fetch_assets.py
├── web/          # Frontend: the board-builder single-page app + downloaded icons.
│   └── index.html, style.css, app.js, assets/, catalog.json, data/
└── cli.py        # `trp` command-line entrypoint for every stage.
```

`api` depends on `training` (it reuses the exact feature transforms) and serves
`web`; `training` has no knowledge of the API or frontend.

## Setup

This project uses uv for dependency management.

1. Install dependencies, choosing a torch backend via an extra — `cu130`
   (CUDA 13.0, for GPU training machines) or `cpu` (CI and machines without
   CUDA). The two extras are mutually exclusive, and a bare `uv sync` installs
   an unpinned PyPI torch, so always pass one of them:
   ```bash
   # GPU (CUDA 13.0) training machine
   uv sync --extra cu130

   # CPU-only (CI, no CUDA)
   uv sync --extra cpu
   ```
2. Activate the virtual environment:
   ```bash
   # On Windows
   .venv\Scripts\activate

   # On macOS/Linux
   source .venv/bin/activate
   ```

## Training

All code under `src/training`. Each model follows the same two-step pattern:
extract features from the raw parquet into a feature directory, then train and
evaluate on a chronological (oldest → train, newest → test) split.

### Baseline (XGBoost)

Takes the list of units, traits and the total board cost as input.

```bash
# Extract features
trp extract-baseline-features --raw-path path/to/raw/data --feature-path path/to/features

# Train & evaluate (optionally save the model for the app with -m)
trp train-baseline --feature-path path/to/features -m models/xgboost/xgboost.json
```

### CNN

Takes unit and item placement and traits as input.

```bash
# Extract features
trp extract-cnn-features --raw-path path/to/raw/data --feature-path path/to/features

# Train & evaluate
trp train-cnn --feature-path path/to/features --batch-size 512 --model-kw dropout=0.2

# Hyperparameter optimization
trp hpo-cnn --feature-path path/to/features --n-trials 50
```

### ViT

Takes unit and item placement and traits as input.

```bash
# Extract features
trp extract-vit-features --raw-path path/to/raw/data --feature-path path/to/features

# Train & evaluate
trp train-vit --feature-path path/to/features --batch-size 512 --model-kw dropout=0.2

# Hyperparameter optimization
trp hpo-vit --feature-path path/to/features --n-trials 50
```

### Data

Each sample is one PVP combat round. both boards at round
start (units, star levels, items, positions) plus the
round outcome as a binary label. PVE and stage-1
rounds, and rounds with missing boards or outcomes, are filtered out.

- **Scale**: ~4.8M usable rounds from ~258k matches and ~1,500 tracked players.
- **Coverage**: December 2025 – February 2026, patches 16.1 → 16.4. All regions.
- **Split**: chronological 80/10/10 (oldest rounds train, newest 10% test) so
  models are always evaluated on rounds played *after* everything they trained
  on, never on shuffled future data.

### Results

Comparison of the different models, all evaluated on the same chronological
test split (the newest 10% of rounds).

| Model | ROC AUC | Accuracy | Log-loss | Brier |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost** | 0.802 | 0.722 | 0.537 | 0.181 |
| **CNN** | 0.882 | 0.791 | 0.434 | 0.142 |
| **ViT** | **0.907** | **0.819** | **0.381** | **0.124** |

## App

An interactive board builder (`src/web`) backed by a FastAPI inference service
(`src/api`). You build a matchup (units, items and star levels for both sides)
and get the predicted win probability from any trained model. Board encoding
reuses the exact training feature transforms, so a board built in the UI is
featurized identically to the data the models were trained on.

### 1. Save a model

- **XGBoost** `train-baseline` saves the fitted model and its feature order:

  ```bash
  trp train-baseline -f data/set16/feature/baseline/set16.parquet -m models/xgboost/xgboost.json
  ```

- **CNN / ViT** `train-cnn` / `train-vit` automatically export the best model to
  ONNX at the serving defaults (`models/cnn/cnn.onnx`, `models/vit/vit.onnx`),
  with a Lightning `.ckpt` copy alongside for resuming / re-export. Use
  `--model-path` to save elsewhere.

### 2. Download the set 16 assets

The UI needs champion / item / trait icons. Fetch them from Community Dragon
(one-off; writes icons + `catalog.json` into the `src/web` frontend):

```bash
trp fetch-assets
```

### 3. Extract sample boards (optional)

The **🎲 random board** button loads a pre-saved board from real games so the
models can be tested on realistic positions. A sample set is committed at
`src/web/data/sample_boards.json`; regenerate it from the raw data with:

```bash
trp extract-sample-boards --raw-path data/set16/raw/merged_data.parquet
```

### 4. Serve the UI + API

```bash
trp serve            # http://127.0.0.1:8000
```

Open the page, drag champions onto the hex grids for both sides, drop items onto
units, hover a placed unit to set its star level, pick a model, and hit
**Predict outcome** or hit **🎲 random board** to load a real board instead of
building one.

Inference runs on ONNX Runtime, so serving needs neither torch nor a GPU. It
defaults to CPU; setting `TRP_DEVICE=cuda` requests the CUDA execution provider
instead (requires the `onnxruntime-gpu` package).

### API

- `GET  /api/models` which model backends are available on disk.
- `POST /api/predict` body: `{"model": "vit|cnn|xgboost", "player": [...], "opponent": [...]}`
  where each unit is `{"unit": "TFT16_Tristana", "tier": 2, "items": [...], "row": 0, "col": 0}`.
  Returns `{"model", "win_probability", "prediction"}`.

## Limitations

- **Board-only inputs, no match context.** The models see units, star levels,
  items, positions and traits, nothing else. Augments are invisible,
  even though they can decide a fight.
- **Combat RNG caps accuracy.** Crits, target selection and ability variance
  mean identical boards can produce different outcomes, so no model can reach
  100%.
- **No damage information.** The raw data records only a binary win/loss per
  round, not the damage inflicted. The models therefore cannot learn the richer
  target used in the Riot references below, where combat is modeled as a
  distribution over damage dealt, a formulation that captures how decisively
  a board wins and from which win probability can be derived.
- **Out-of-distribution boards.** The board builder accepts positions that
  never occur in real games (illegal item stacks, unit counts no player level
  allows). The models still return a confident probability for them, so treat
  predictions on unrealistic hand-built boards with skepticism.

## References

Ran Cao (Riot Games) "Machine Learning Summit: Simulating Teamfight Tactics Using Deep Learning for Fast Reinforcement Learning AI Training" ([Slides](https://gdcvault.com/play/1028851/Machine-Learning-Summit-Simulating-Teamfight)) ([Video](https://gdcvault.com/play/1029228/Machine-Learning-Summit-Simulating-Teamfight))

Wesley Kerr (Riot Games) "Large-scale deep learning to augment production RL workloads" ([Video](https://www.youtube.com/watch?v=8EsQkFxWYhU))
