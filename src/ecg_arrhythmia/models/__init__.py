"""Models module exports."""

from ecg_arrhythmia.models.classical import (
    ClassicalModelWrapper,
    build_gradient_boosting,
    build_logistic_regression,
    build_random_forest,
    build_svm,
)
from ecg_arrhythmia.models.deep_learning import (
    CNN1D,
    FFT2DCNN,
    MLP,
    AutoencoderClassifier,
    BiLSTM,
    CNNTransformerHybrid,
)
from ecg_arrhythmia.models.ensemble import (
    HandcraftedDeepFusionModel,
    HeterogeneousEnsemble,
    extract_morphological_features,
)
from ecg_arrhythmia.models.registry import MODEL_REGISTRY, get_model, list_models

__all__ = [
    "ClassicalModelWrapper",
    "build_logistic_regression",
    "build_random_forest",
    "build_gradient_boosting",
    "build_svm",
    "MLP",
    "CNN1D",
    "FFT2DCNN",
    "BiLSTM",
    "CNNTransformerHybrid",
    "AutoencoderClassifier",
    "HandcraftedDeepFusionModel",
    "HeterogeneousEnsemble",
    "extract_morphological_features",
    "MODEL_REGISTRY",
    "get_model",
    "list_models",
]
