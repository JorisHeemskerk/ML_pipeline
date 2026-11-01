"""
The training engine: a `Trainer` that does split training, k-fold cross
validation and evaluation, selectable metrics and visualisers, and
callbacks for everything around the training loop.
"""

from training.callbacks import (
    Callback,
    CheckpointCallback,
    EarlyStoppingCallback,
    PruningCallback
)
from training.early_stopper import EarlyStopper
from training.metrics import METRICS, Metric
from training.results import (
    EpochResult,
    EvaluationResult,
    FoldResult,
    TrainingResult
)
from training.trainer import Trainer, TrainerConfig
from training.visualisation import VISUALISERS, Visualiser
