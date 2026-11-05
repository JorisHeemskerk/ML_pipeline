"""
The training engine: a `Trainer` that does split training, k-fold cross
validation and evaluation, selectable metrics and visualisers, and
callbacks for everything around the training loop.
"""

from training.callbacks import (
    CALLBACKS,
    Callback, 
    CheckpointCallback, 
    EarlyStoppingCallback, 
    PruningCallback,
    build_callbacks
)
from training.early_stopper import EarlyStopper
from training.feature_resampling import DeadFeatureResamplingCallback
from training.feature_visualisation import FeatureVisualiser
from training.metrics import METRICS, Metric
from training.results import (
    EpochResult, 
    EvaluationResult, 
    FoldResult, 
    TrainingResult
)
from training.tensors import ModelOutput
from training.trainer import Trainer, TrainerConfig
from training.visualisation import VISUALISERS, Visualiser
