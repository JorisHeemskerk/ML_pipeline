"""
Dataclasses that hold everything the trainer produces.
"""

from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import Dataset

from training.tensors import ModelOutput


def _pad_to_longest(arrays: list[list[float]]) -> np.ndarray:
    """
    Pad arrays to equal length with NaN so that folds stopped early do
    not corrupt the mean/std calculation.

    :param arrays: Arrays of different sizes.
    :type arrays: list[list[float]]
    :return: Padded arrays, shape (len(arrays), longest).
    :rtype: np.ndarray
    """
    max_len = max(len(a) for a in arrays)
    return np.array([
        np.pad(
            np.array(a, dtype=float), 
            (0, max_len - len(a)), 
            constant_values=np.nan
        )
        for a in arrays
    ])


@dataclass
class StepOutput:
    """
    Output of a single training or validation step.

    :param loss: Loss of the batch (still attached to the graph in the
        training step).
    :type loss: torch.Tensor
    :param loss_components: Named parts of the loss, empty if the loss
        function does not report any.
    :type loss_components: dict[str, float]
    :param y_hat: Model output, detached. A tensor, or a tuple or dict of
        tensors if the model has multiple outputs.
    :type y_hat: ModelOutput
    :param y: Targets.
    :type y: torch.Tensor
    """
    loss: torch.Tensor
    loss_components: dict[str, float]
    y_hat: ModelOutput
    y: torch.Tensor


@dataclass
class EpochResult:
    """
    Average loss and metrics over one epoch.

    :param loss: Average total loss.
    :type loss: float
    :param loss_components: Average of each named loss component.
    :type loss_components: dict[str, float]
    :param metrics: Average of each metric.
    :type metrics: dict[str, float]
    """
    loss: float
    loss_components: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)

    def get(self, key: str) -> float:
        """
        Look up a value by name: "loss", a metric or a loss component.

        :param key: Name to look up.
        :type key: str
        :return: The value belonging to `key`.
        :rtype: float
        """
        if key == "loss":
            return self.loss
        if key in self.metrics:
            return self.metrics[key]
        if key in self.loss_components:
            return self.loss_components[key]
        raise KeyError(
            f"'{key}' is not 'loss', a metric ({list(self.metrics)}) or "
            f"a loss component ({list(self.loss_components)})."
        )


@dataclass
class FoldResult:
    """
    Everything that happened during the training of one fold.

    :param train: Training result of each epoch.
    :type train: list[EpochResult]
    :param val: Validation result of each epoch.
    :type val: list[EpochResult]
    :param best_epoch: Epoch (0-indexed) with the best monitored 
        validation score, -1 if no epoch has finished yet.
    :type best_epoch: int
    :param best_score: Best monitored validation score.
    :type best_score: float
    :param stopped_early: True if a callback stopped the training before
        the last epoch.
    :type stopped_early: bool
    """
    train: list[EpochResult] = field(default_factory=list)
    val: list[EpochResult] = field(default_factory=list)
    best_epoch: int = -1
    best_score: float = float("nan")
    stopped_early: bool = False

    @property
    def n_epochs(self) -> int:
        """Number of epochs that were trained."""
        return len(self.train)

    @property
    def keys(self) -> list[str]:
        """All names that `curve` accepts."""
        if not self.train:
            return ["loss"]
        return [
            "loss", 
            *self.train[0].loss_components, 
            *self.train[0].metrics
        ]

    @property
    def metric_names(self) -> list[str]:
        """Names of all metrics."""
        return list(self.train[0].metrics) if self.train else []

    def curve(self, split: str, key: str) -> list[float]:
        """
        Get the value of `key` for every epoch.

        :param split: Either "train" or "val".
        :type split: str
        :param key: "loss", a metric or a loss component.
        :type key: str
        :return: One value per epoch.
        :rtype: list[float]
        """
        if split not in ("train", "val"):
            raise ValueError(f"Split must be 'train' or 'val', not {split}.")
        return [epoch.get(key) for epoch in getattr(self, split)]


@dataclass
class TrainingResult:
    """
    Results of a full training run (one fold for a normal split, `k`
    folds for cross validation).

    :param folds: Result of each fold.
    :type folds: list[FoldResult]
    :param best_fold: Index of the fold with the best monitored score,
        -1 if no fold has finished yet.
    :type best_fold: int
    """
    folds: list[FoldResult] = field(default_factory=list)
    best_fold: int = -1

    @property
    def mean_best_score(self) -> float:
        """Mean over the folds of the best monitored score."""
        return float(np.nanmean([fold.best_score for fold in self.folds]))

    def curve(self, split: str, key: str) -> np.ndarray:
        """
        Get the value of `key` for every fold and epoch.

        :param split: Either "train" or "val".
        :type split: str
        :param key: "loss", a metric or a loss component.
        :type key: str
        :return: Array of shape (k_folds, n_epochs), padded with NaN
            for folds that stopped early.
        :rtype: np.ndarray
        """
        return _pad_to_longest([fold.curve(split, key) for fold in self.folds])

    def mean_curve(
        self, 
        split: str, 
        key: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the mean and standard deviation over the folds, per epoch.

        :param split: Either "train" or "val".
        :type split: str
        :param key: "loss", a metric or a loss component.
        :type key: str
        :return: Mean and standard deviation, both of shape (n_epochs,).
        :rtype: tuple[np.ndarray, np.ndarray]
        """
        curve = self.curve(split, key)
        return np.nanmean(curve, axis=0), np.nanstd(curve, axis=0)


@dataclass
class EvaluationResult:
    """
    Result of evaluating a model on a dataset (e.g. the test set).

    :param loss: Average total loss over the batches.
    :type loss: float
    :param loss_components: Average of each named loss component.
    :type loss_components: dict[str, float]
    :param metrics: Each metric, computed over the entire dataset at 
        once.
    :type metrics: dict[str, float]
    :param predictions: All model outputs, concatenated, on the CPU. A
        tensor, or a tuple or dict of tensors if the model has multiple
        outputs.
    :type predictions: ModelOutput
    :param targets: All targets, concatenated, on the CPU.
    :type targets: torch.Tensor
    :param sample_inputs: Inputs of the first batch, on the CPU. Handy
        for visualisers that draw the model input.
    :type sample_inputs: torch.Tensor | None
    :param dataset: Dataset that was evaluated, so visualisers can look
        up the inputs of any datapoint: item `i` of `predictions` belongs
        to `dataset[i]`. None if the dataloader shuffled the data, as
        that order is not kept. (DEFAULT=None)
    :type dataset: Dataset | None
    """
    loss: float
    loss_components: dict[str, float]
    metrics: dict[str, float]
    predictions: ModelOutput
    targets: torch.Tensor
    sample_inputs: torch.Tensor | None = None
    dataset: Dataset | None = None
