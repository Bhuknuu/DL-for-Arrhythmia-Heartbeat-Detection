"""Model factory and central registry for all 12 benchmark architectures."""

from typing import Any, Callable, Dict, List

from ecg_arrhythmia.models.classical import (
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
from ecg_arrhythmia.models.ensemble import HandcraftedDeepFusionModel, HeterogeneousEnsemble

MODEL_REGISTRY: Dict[str, Callable[..., Any]] = {
    # 1. Linear Baseline
    "logistic_regression": build_logistic_regression,
    # 2. Tree Ensembles
    "random_forest": build_random_forest,
    "gradient_boosting": build_gradient_boosting,
    # 3. Distance-based
    "svm": build_svm,
    # 4. Feedforward NN
    "mlp": lambda **kw: MLP(**kw),
    # 5. Spatial / Frequency
    "1d_cnn": lambda **kw: CNN1D(**kw),
    "fft_2d_cnn": lambda **kw: FFT2DCNN(**kw),
    # 6. Sequence
    "bilstm": lambda **kw: BiLSTM(**kw),
    # 7. Attention
    "cnn_transformer": lambda **kw: CNNTransformerHybrid(**kw),
    # 8. Fusion
    "fusion_model": lambda **kw: HandcraftedDeepFusionModel(**kw),
    # 9. Anomaly / Unsupervised Pretraining
    "autoencoder": lambda **kw: AutoencoderClassifier(**kw),
    # 10. Synthesis
    "heterogeneous_ensemble": lambda **kw: HeterogeneousEnsemble(**kw),
}


def get_model(name: str, **kwargs: Any) -> Any:
    """Instantiates a model by its registry name.

    Args:
        name: Name of the model architecture (e.g. '1d_cnn', 'bilstm')
        **kwargs: Additional hyperparameters passed to the model constructor

    Returns:
        Instantiated model (PyTorch nn.Module or Scikit-Learn wrapper)
    """
    key = name.lower().replace("-", "_").replace(" ", "_")
    if key not in MODEL_REGISTRY:
        raise KeyError(
            f"Model '{name}' not found in registry. Available models: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[key](**kwargs)


def list_models() -> List[str]:
    """Returns a list of all registered model architecture names."""
    return list(MODEL_REGISTRY.keys())
