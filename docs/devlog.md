# Dev Log: DL for Arrhythmia Heartbeat Detection

First-person log of decisions and changes made throughout the project. Short entries, honest language, no cleanup after the fact.

---

## Phase A: Research & Ideation

### Day 0 - Project Start

Started with the MIT-BIH Arrhythmia Database and two rough instincts: either a neural network (Transformer or LSTM) to classify raw ECG beats, or a simpler baseline where I fit a "normal" line to a patient's heartbeat and flag deviations with linear regression. No concrete plan yet.

First thing I did was read what the dataset actually contains. Found out arrhythmia is not the same as a heart attack. MIT-BIH labels irregular heart rhythms (AAMI classes N/S/V/F/Q), not myocardial infarction. That's a different dataset (PTB-XL). Fixed the scope immediately before touching any code.

### Day 0 - Compute Budget

Checked what hardware is actually needed. MIT-BIH is small by DL standards: 48 half-hour recordings, ~110,000 labeled beats. State-of-the-art CNN-Transformer hybrids on this dataset train on a single GPU. Free-tier compute (Colab, Kaggle) is enough. The real constraint isn't hardware, it's methodology.

### Day 0 - Architecture Survey

Surveyed the landscape instead of committing to one architecture. Settled on three legitimate framings for the project:

- Multi-class classification: predict which of the 5 AAMI classes a beat belongs to.
- Anomaly detection: learn what "normal" looks like, flag deviations. Closest to my original linear baseline idea.
- Cascade of both: anomaly gate first, heavier classifier only on flagged beats. More clinically realistic.

The cascade approach absorbs both original ideas. That became the framing.

Architectures worth testing: classical ML baselines (Logistic Regression, RF, XGBoost), 1D-CNN (learns morphology directly from raw signal), LSTM/GRU (cross-beat temporal dependencies), CNN-Transformer hybrids (best in 2025-2026 literature), FFT 2D-CNN (frequency-domain representation), Autoencoder (anomaly detection track), fusion and ensemble variants.

Decided early this is a comparison study, not a "pick the best one" exercise.

### Day 0 - Critical Methodology Decision

The most important decision: how to split the data.

Most tutorials split individual beats randomly into train/test. That inflates accuracy past 99% because beats from the same patient appear on both sides. The model memorizes patient-specific quirks, not actual arrhythmia patterns.

Decided to use the inter-patient paradigm: patient-grouped split where no patient's beats appear in both train and test. Reported accuracy will be lower and honest.

Also decided: report Macro-F1 and per-class sensitivity, not just accuracy. ~90% of beats are Normal. A model that always predicts Normal scores 90% and is worthless. The rare classes are where the diagnostic value is.

---

## Phase B: Data Loading & EDA

### Entry 2 - Validation Method Upgrade

Switched from the fixed de Chazal DS1/DS2 split to patient-grouped k-fold cross-validation (`GroupKFold`, patient ID as the group key). With 48 records this gives ~10-12 folds of 4-5 whole patients each.

Why: a fixed split gives one point estimate. k-fold gives mean ± std across folds, which is what makes a proper statistical comparison between models possible downstream. The patient-level grouping is non-negotiable either way.

One subtlety I locked in: normalization statistics (mean/std per beat) are computed inside each fold, from that fold's training patients only. Computing them over the whole dataset before splitting is a quieter version of the same data leak I'm already trying to prevent.

### Entry 2 - Model Lineup Curation: 22 to 12

Trimmed the candidate list from 22 to 12. Cut rule: each model stays only if it tests something no other model in the set already covers.

Cuts and why:
- Naive Bayes: weaker than Logistic Regression, adds no distinct comparison point.
- K-Nearest Neighbors: SVM already covers the non-tree classical geometry.
- Plain Decision Tree: subsumed by Random Forest.
- Standalone LSTM, GRU: Bidirectional LSTM dominates both on this task.
- Standalone Transformer: CNN-Transformer hybrids beat pure attention in every paper I reviewed.
- Second anomaly route (K-Means): Autoencoder covers the anomaly track directly.

Final lineup: 12 trained models across 5 families, plus PCA as a utility step (preprocessing and cluster visualization only, not scored as a detector).

---

## Phase B+: Repo Restructure

### Entry 3 - Decision to Restructure

The original repo was one monolithic 765KB notebook plus raw dataset binaries committed to Git. This worked for early exploration but became a problem as the scope expanded to 12 models across 10+ folds. Problems:

- All model definitions, preprocessing, data loading, and EDA in the same notebook. Changing the window length for beat segmentation meant hunting through thousands of lines.
- ~80MB of raw binary files committed to Git. Slow clones, no separation between raw and processed data.
- No way to share preprocessing code between notebooks without copy-pasting.
- No automated tests. Data leakage could silently pass through.
- No package structure. `import ecg_arrhythmia` throws `ModuleNotFoundError`.

Decided to refactor before Phase C. Restructuring now costs a few days. Discovering a silent data leakage bug after running 12 models across 10 folds costs far more.

### Entry 3 - What Changed

**Directory structure.** Created `src/ecg_arrhythmia/` as an installable Python package (`pip install -e .` via `pyproject.toml`). Organized into `data/`, `models/`, `evaluation/`, `inference/`, `tracking/`, `utils/` submodules. Any notebook or script can now do `from ecg_arrhythmia.data import extract_all_beats` without path hacks.

**Data pipeline extracted.** `loader.py` handles WFDB record loading. `preprocessor.py` holds the AAMI EC57 annotation map (the exact symbol-to-class mapping is now a testable object, not an inline dict buried in a notebook cell). `dataset.py` holds `GroupKFoldSplitter` with a built-in `assert_no_leakage()` check that raises immediately if any patient ID appears in both train and test.

**12 model architectures extracted.** Moved out of notebook cells into `models/classical.py`, `models/deep_learning.py`, `models/ensemble.py`. Added `registry.py` with a `get_model("1d_cnn")` factory so any notebook or training script instantiates models the same way.

**Model registry approach.** Adding a 13th model means: write the class in the right file, register it in `registry.py`. Nothing else changes in any notebook or script.

**5 phase notebooks.** Replaced the single master notebook with 5 lean notebooks:
- `01_eda_and_waveform_analysis.ipynb`
- `02_groupkfold_leakage_safe_splits.ipynb`
- `03_model_benchmarking.ipynb`
- `04_statistical_significance_testing.ipynb`
- `05_interpretability_and_reporting.ipynb`

Each imports from `ecg_arrhythmia` rather than defining everything inline. The master notebook moved to `notebooks/archive/` and is frozen.

**Raw data out of Git.** 202 raw MIT-BIH binary files moved from `dataset/` to `data/raw/mitdb/`. The folder is gitignored. `scripts/download_data.py` downloads and verifies records on demand via `wfdb`. The repo clone is now tiny.

**Experiment tracking.** Added `src/ecg_arrhythmia/tracking/experiment_logger.py`, a thin MLflow wrapper. Every training run logs config, fold metrics, and artifact paths to a local file store under `runs/`. The significance testing notebook can query this instead of reading hardcoded numbers.

**FastAPI inference service.** Added `api/app.py` with `GET /health` and `POST /predict`. The `/predict` endpoint accepts a raw 200-sample ECG beat array and returns the predicted AAMI class and probability distribution. Uses `Predictor` from `src/ecg_arrhythmia/inference/predictor.py`, which is the same class used in the notebooks.

**Automated tests.** Added 5 pytest modules, all 21 pass:
- `test_data_integrity.py`: AAMI EC57 symbol-to-class mapping matches the spec exactly. Normalization produces correct mean/std.
- `test_data_leakage.py`: `GroupKFoldSplitter` produces zero patient overlap across all folds. `assert_no_leakage()` raises on a manufactured overlap.
- `test_models.py`: Forward pass and output shape `(batch, 5)` for all 7 PyTorch models. Fit/predict/predict_proba for all 4 classical models. Heterogeneous ensemble with mixed member types.
- `test_pipeline_smoke.py`: Full end-to-end run (data extraction, 1-epoch training, evaluation, predictor inference) without errors.
- `test_api.py`: `/health` returns 200. `/predict` returns correct schema. Empty signal returns 400.

**CI.** Added `.github/workflows/ci.yml`. On every push or PR: installs the package, runs `ruff check`, runs full `pytest tests/`. Matrix across Python 3.10, 3.11, 3.12.

**Pre-commit.** Added `.pre-commit-config.yaml` running Ruff before every local commit.

**Configs.** `configs/dataset_config.yaml` holds AAMI mapping, window sizes, and GroupKFold settings. `configs/model_config.yaml` holds hyperparameters for all 12 models. No magic numbers buried in code.

---

## Phase C: Splits, Balancing, Baseline (Upcoming)

Patient-grouped 10-fold CV split is implemented in `src/ecg_arrhythmia/data/dataset.py`. Next step is running it across the full 48-record dataset, applying per-fold normalization and class balancing (class weights or SMOTE on training portion only), computing the majority-class baseline per fold, and feeding the resulting data splits into each of the 12 model notebooks via shared imports.

---

## Phase D: Training (Upcoming)

Train all 12 models across all folds. Log config, fold metrics, and artifacts to `runs/` via `ExperimentLogger`. Time every run.

---

## Phase E: Evaluation (Upcoming)

Aggregate confusion matrices across folds. Report accuracy as mean ± std per model. Run paired Wilcoxon signed-rank test between top 2-3 performers. Interpretability pass (SHAP for RF, attention weights for CNN-Transformer).

---

## Phase F: Final Report (Upcoming)

Confusion matrix comparison grid, accuracy/F1 bar charts with error bars, training time table, stated limitations, significance test results, hyperparameter log.
