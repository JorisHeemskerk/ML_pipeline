"""
The `Trainer`: one object that does normal (split) training, k-fold 
cross validation and evaluation.

The training loop is split into small methods. To change one aspect of
the training, subclass `Trainer` and override only that method:

- `forward`: how a batch goes through the model (e.g. reshaping the
  output of a detection model). The output may be a tensor, or a tuple 
  or dict of tensors for models with multiple outputs, the loss and the
  metrics then receive that structure.
- `compute_loss`: how the loss is computed from the predictions.
- `compute_metrics`: how the metrics are computed from the predictions.
- `training_step` / `validation_step`: what happens for one batch.
- `train_epoch` / `val_epoch`: what happens for one epoch.
- `get_fold_indices`: how the dataset is divided into folds.
- `prepare_fold`: what happens to the dataset before a fold starts 
  (e.g. fitting normalisation on the training indices).
"""

import copy
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, RandomSampler, Subset
from tqdm import tqdm

from training.callbacks import Callback
from training.metrics import Metric, build_metrics
from training.results import (
    EpochResult, 
    EvaluationResult, 
    FoldResult, 
    StepOutput, 
    TrainingResult
)
from training.tensors import ModelOutput, concatenate, detach, to_cpu
from training.visualisation import Visualiser, build_visualisers


@dataclass
class TrainerConfig:
    """
    Settings of the `Trainer`, simple enough to be read from a config
    file.

    :param n_epochs: Number of epochs to train for.
    :type n_epochs: int
    :param metrics: Metrics to track, registered names or (registered 
        name, kwargs), see `metrics.py`.
    :type metrics: list[str | tuple[str, dict]]
    :param monitor: Validation value that decides which epoch is the
        best: "loss", a metric or a loss component. (DEFAULT="loss")
    :type monitor: str
    :param mode: "min" if a lower monitored value is better, "max" if a
        higher one is. (DEFAULT="min")
    :type mode: str
    :param mixed_precision: True to run the forward pass in bfloat16.
        (DEFAULT=False)
    :type mixed_precision: bool
    """
    n_epochs: int
    metrics: list[str | tuple[str, dict]] = field(default_factory=list)
    monitor: str = "loss"
    mode: str = "min"
    mixed_precision: bool = False

    def __post_init__(self):
        if self.mode not in ("min", "max"):
            raise ValueError(f"Mode must be 'min' or 'max', not {self.mode}.")


class Trainer:
    """
    Trains, validates and evaluates a model.

    The loss function can be any module that returns either the loss
    tensor, or a tuple of the loss tensor and a dict of named loss 
    components (e.g. {"xy": ..., "wh": ...}), which are then tracked
    per epoch as well.

    :param model: Model to train.
    :type model: nn.Module
    :param loss_fn: Loss function to update gradients with.
    :type loss_fn: nn.Module
    :param optimiser: Optimiser used for backpropagation.
    :type optimiser: torch.optim.Optimizer
    :param device: Device to move data to.
    :type device: str
    :param logger: Logger to log to.
    :type logger: logging.Logger
    :param config: Settings of the trainer.
    :type config: TrainerConfig
    :param scheduler: Scheduler to change how the learning rate adapts.
        (DEFAULT=None)
    :type scheduler: torch.optim.lr_scheduler.LRScheduler | None
    :param callbacks: Callbacks to hook into the training. (DEFAULT=None)
    :type callbacks: list[Callback] | None
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimiser: torch.optim.Optimizer,
        device: str,
        logger: logging.Logger,
        config: TrainerConfig,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None=None,
        callbacks: list[Callback] | None=None
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimiser = optimiser
        self.device = device
        self.logger = logger
        self.config = config
        self.scheduler = scheduler
        self.callbacks = callbacks or []
        self.metrics: dict[str, Metric] = build_metrics(config.metrics)

        # State that callbacks can read (and `should_stop` can write).
        self.result = TrainingResult()
        self.current_fold = FoldResult()
        self.train_dataloader: DataLoader | None = None
        self.val_dataloader: DataLoader | None = None
        self.is_best_epoch = False
        self.should_stop = False

        self._best_score = float("nan")
        self._best_state: dict | None = None
        self._initial_model_state: dict = {}
        self._initial_optimiser_state: dict = {}
        self._initial_scheduler_state: dict | None = None

    # ------------------------------------------------------------------
    # Public entry points.
    # ------------------------------------------------------------------

    def fit_split(
        self, 
        train_dataloader: DataLoader, 
        val_dataloader: DataLoader
    ) -> TrainingResult:
        """
        Train on a fixed train/validation split. The weights of the best
        epoch are loaded into the model afterwards.

        :param train_dataloader: Data to train with.
        :type train_dataloader: DataLoader
        :param val_dataloader: Data to validate with.
        :type val_dataloader: DataLoader
        :return: Results, containing a single fold.
        :rtype: TrainingResult
        """
        self._start_fit()
        self._run_fold(0, 1, train_dataloader, val_dataloader)
        return self._finish_fit()

    def fit_cross_validation(
        self, 
        dataset: Dataset, 
        k_folds: int, 
        dataset_to_dataloader_function: Callable[[Dataset], DataLoader]
    ) -> TrainingResult:
        """
        Train using k-fold cross validation. Every fold starts from the
        same initial weights. The weights of the best epoch of the best
        fold are loaded into the model afterwards.

        :param dataset: Dataset to divide into folds.
        :type dataset: Dataset
        :param k_folds: The number of folds to use.
        :type k_folds: int
        :param dataset_to_dataloader_function: Function that converts a 
            Dataset into a DataLoader (or a tuple with the DataLoader as
            first item).
        :type dataset_to_dataloader_function: Callable
        :return: Results, containing a fold for every `k_folds`.
        :rtype: TrainingResult
        """
        self._start_fit()
        for k in range(k_folds):
            self.logger.info(
                f"-----=====##### Fold {k + 1}/{k_folds} #####=====-----"
            )
            train_idx, val_idx = self.get_fold_indices(dataset, k, k_folds)
            self.prepare_fold(dataset, train_idx, val_idx)

            train_dataloader = self._as_dataloader(
                dataset_to_dataloader_function, Subset(dataset, train_idx)
            )
            val_dataloader = self._as_dataloader(
                dataset_to_dataloader_function, Subset(dataset, val_idx)
            )
            self._run_fold(k, k_folds, train_dataloader, val_dataloader)
        return self._finish_fit()

    def evaluate(
        self,
        dataloader: DataLoader,
        visualisers: list[str | tuple[str, dict]] | None=None,
        output_dir: str | Path | None=None
    ) -> EvaluationResult:
        """
        Evaluate the model on a dataset, such as the test set. Metrics 
        are computed over the entire dataset at once, the loss is the
        average over the batches.

        :param dataloader: Data to evaluate with.
        :type dataloader: DataLoader
        :param visualisers: Visualisers to run on the result, registered
            names or (registered name, kwargs), see `visualisation.py`.
            (DEFAULT=None)
        :type visualisers: list[str | tuple[str, dict]] | None
        :param output_dir: Where to save the visualisations, required if
            `visualisers` is given. (DEFAULT=None)
        :type output_dir: str | Path | None
        :return: Loss, metrics, predictions and targets.
        :rtype: EvaluationResult
        """
        if visualisers and output_dir is None:
            raise ValueError("`output_dir` is required to use visualisers.")

        shuffled = isinstance(dataloader.sampler, RandomSampler)
        if shuffled:
            self.logger.warning(
                "The dataloader shuffles, so the order of the predictions "
                "does not match the dataset. `EvaluationResult.dataset` "
                "will not be set."
            )

        self.model.eval()
        losses: list[float] = []
        components: dict[str, list[float]] = defaultdict(list)
        predictions, targets = [], []
        sample_inputs = None

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating batches"):
                with self._autocast():
                    y_hat, y = self.forward(batch)
                    loss, loss_components = self.compute_loss(y_hat, y)

                losses.append(loss.item())
                for name, value in loss_components.items():
                    components[name].append(value)
                predictions.append(to_cpu(y_hat))
                targets.append(y.cpu())
                if sample_inputs is None:
                    sample_inputs = batch[0].cpu()

        predictions = concatenate(predictions)
        targets = torch.cat(targets)
        result = EvaluationResult(
            loss=float(np.mean(losses)),
            loss_components=self._mean_per_key(components),
            metrics=self.compute_metrics(predictions, targets),
            predictions=predictions,
            targets=targets,
            sample_inputs=sample_inputs,
            dataset=None if shuffled else dataloader.dataset
        )
        self._log_result("test", result.loss, result.metrics)

        if visualisers:
            self.visualise(result, visualisers, output_dir)
        return result

    def visualise(
        self,
        result: EvaluationResult,
        visualisers: list[str | tuple[str, dict]],
        output_dir: str | Path
    ) -> list[Path]:
        """
        Run visualisers on the result of an evaluation.

        :param result: Result of `evaluate`.
        :type result: EvaluationResult
        :param visualisers: Registered names or (registered name, 
            kwargs), see `visualisation.py`.
        :type visualisers: list[str | tuple[str, dict]]
        :param output_dir: Where to save the visualisations.
        :type output_dir: str | Path
        :return: Paths of all images that were created.
        :rtype: list[Path]
        """
        built: dict[str, Visualiser] = build_visualisers(visualisers)
        paths = []
        for name, visualiser in built.items():
            self.logger.info(f"Running visualiser '{name}'")
            saved = visualiser(result, Path(output_dir))
            if not saved:
                self.logger.warning(f"Visualiser '{name}' created no files.")
            paths.extend(saved)
        return paths

    # ------------------------------------------------------------------
    # Overridable parts of the training loop.
    # ------------------------------------------------------------------

    def get_fold_indices(
        self, 
        dataset: Dataset, 
        fold: int, 
        k_folds: int
    ) -> tuple[list[int], list[int]]:
        """
        Decide which items of the dataset are used for training and for
        validation in a fold. By default the dataset is divided into
        `k_folds` consecutive blocks of (almost) equal size.

        :param dataset: Dataset to divide.
        :type dataset: Dataset
        :param fold: Index of the fold (0-indexed).
        :type fold: int
        :param k_folds: The number of folds.
        :type k_folds: int
        :return: Training indices and validation indices.
        :rtype: tuple[list[int], list[int]]
        """
        blocks = np.array_split(np.arange(len(dataset)), k_folds)
        val_idx = blocks[fold].tolist()
        train_idx = np.concatenate(
            [block for i, block in enumerate(blocks) if i != fold]
        ).tolist()
        return train_idx, val_idx

    def prepare_fold(
        self, 
        dataset: Dataset, 
        train_idx: list[int], 
        val_idx: list[int]
    ) -> None:
        """
        Called before the dataloaders of a fold are created. Does
        nothing by default, override to e.g. fit normalisation on the 
        training indices only.

        :param dataset: The full dataset.
        :type dataset: Dataset
        :param train_idx: Training indices of this fold.
        :type train_idx: list[int]
        :param val_idx: Validation indices of this fold.
        :type val_idx: list[int]
        """

    def forward(
        self, 
        batch: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[ModelOutput, torch.Tensor]:
        """
        Move a batch to the device and run it through the model.

        :param batch: Inputs and targets.
        :type batch: tuple[torch.Tensor, torch.Tensor]
        :return: Model output (a tensor, or a tuple or dict of tensors if
            the model has multiple outputs) and targets, on the device.
        :rtype: tuple[ModelOutput, torch.Tensor]
        """
        X, y = batch
        X = X.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        return self.model(X), y

    def compute_loss(
        self, 
        y_hat: ModelOutput, 
        y: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute the loss with the loss function.

        :param y_hat: Model output.
        :type y_hat: ModelOutput
        :param y: Targets.
        :type y: torch.Tensor
        :return: Total loss and the named loss components (empty if the
            loss function returns no components).
        :rtype: tuple[torch.Tensor, dict[str, float]]
        """
        output = self.loss_fn(y_hat, y)
        if isinstance(output, tuple):
            loss, components = output
            return loss, {name: float(v) for name, v in components.items()}
        return output, {}

    def compute_metrics(
        self, 
        y_hat: ModelOutput, 
        y: torch.Tensor
    ) -> dict[str, float]:
        """
        Compute all configured metrics.

        :param y_hat: Model output.
        :type y_hat: ModelOutput
        :param y: Targets.
        :type y: torch.Tensor
        :return: Dict mapping metric names to values.
        :rtype: dict[str, float]
        """
        return {
            name: metric(y_hat, y) for name, metric in self.metrics.items()
        }

    def training_step(
        self, 
        batch: tuple[torch.Tensor, torch.Tensor]
    ) -> StepOutput:
        """
        Train on a single batch.

        :param batch: Inputs and targets.
        :type batch: tuple[torch.Tensor, torch.Tensor]
        :return: Loss, loss components, predictions and targets.
        :rtype: StepOutput
        """
        self.optimiser.zero_grad(set_to_none=True)
        with self._autocast():
            y_hat, y = self.forward(batch)
            loss, components = self.compute_loss(y_hat, y)
        loss.backward()
        self.optimiser.step()
        return StepOutput(loss, components, detach(y_hat), y)

    def validation_step(
        self, 
        batch: tuple[torch.Tensor, torch.Tensor]
    ) -> StepOutput:
        """
        Validate on a single batch, without updating the model.

        :param batch: Inputs and targets.
        :type batch: tuple[torch.Tensor, torch.Tensor]
        :return: Loss, loss components, predictions and targets.
        :rtype: StepOutput
        """
        with self._autocast():
            y_hat, y = self.forward(batch)
            loss, components = self.compute_loss(y_hat, y)
        return StepOutput(loss, components, y_hat, y)

    def train_epoch(self, dataloader: DataLoader) -> EpochResult:
        """
        Train the model for 1 epoch.

        :param dataloader: Data to train with.
        :type dataloader: DataLoader
        :return: Average training loss and metrics over the epoch.
        :rtype: EpochResult
        """
        self.model.train()
        result = self._run_epoch(dataloader, self.training_step)
        self._log_result("train", result.loss, result.metrics)
        return result

    def val_epoch(self, dataloader: DataLoader) -> EpochResult:
        """
        Validate the model for 1 epoch.

        :param dataloader: Data to validate with.
        :type dataloader: DataLoader
        :return: Average validation loss and metrics over the epoch.
        :rtype: EpochResult
        """
        self.model.eval()
        with torch.no_grad():
            result = self._run_epoch(dataloader, self.validation_step)
        self._log_result("validation", result.loss, result.metrics)
        return result

    # ------------------------------------------------------------------
    # Internals.
    # ------------------------------------------------------------------

    def _start_fit(self) -> None:
        """
        Reset the results and remember the initial states, so every fold
        starts from the same weights.
        """
        self.result = TrainingResult()
        self._best_score = float("nan")
        self._best_state = None
        self._initial_model_state = copy.deepcopy(self.model.state_dict())
        self._initial_optimiser_state = copy.deepcopy(
            self.optimiser.state_dict()
        )
        self._initial_scheduler_state = copy.deepcopy(
            self.scheduler.state_dict()
        ) if self.scheduler is not None else None
        self._call_callbacks("on_fit_start")

    def _finish_fit(self) -> TrainingResult:
        """
        Load the best weights across all folds into the model.

        :return: The results of the training.
        :rtype: TrainingResult
        """
        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)
        self.logger.info("Done training")
        self._call_callbacks("on_fit_end")
        return self.result

    def _reset_training_state(self) -> None:
        """
        Reset the model, optimiser and scheduler to their initial state.
        """
        self.model.load_state_dict(copy.deepcopy(self._initial_model_state))
        self.optimiser.load_state_dict(
            copy.deepcopy(self._initial_optimiser_state)
        )
        if self.scheduler is not None:
            self.scheduler.load_state_dict(
                copy.deepcopy(self._initial_scheduler_state)
            )

    def _run_fold(
        self,
        fold: int,
        k_folds: int,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader
    ) -> None:
        """
        Train one fold, and restore the best weights of that fold.

        :param fold: Index of the fold (0-indexed).
        :type fold: int
        :param k_folds: The number of folds.
        :type k_folds: int
        :param train_dataloader: Data to train with.
        :type train_dataloader: DataLoader
        :param val_dataloader: Data to validate with.
        :type val_dataloader: DataLoader
        """
        self._reset_training_state()
        self.current_fold = FoldResult()
        self.result.folds.append(self.current_fold)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.should_stop = False
        self.is_best_epoch = False
        fold_best_state = None
        self._call_callbacks("on_fold_start", fold)

        preamble = f"Fold {fold + 1}/{k_folds} - " if k_folds > 1 else ""
        for epoch in tqdm(range(self.config.n_epochs), f"{preamble}Epoch"):
            self.logger.info(f"-----===== Epoch {epoch} (training) =====-----")
            train_result = self.train_epoch(train_dataloader)
            self.logger.info(
                f"-----===== Epoch {epoch} (validation) =====-----"
            )
            val_result = self.val_epoch(val_dataloader)
            self.current_fold.train.append(train_result)
            self.current_fold.val.append(val_result)

            score = val_result.get(self.config.monitor)
            self.is_best_epoch = self._is_better(
                score, self.current_fold.best_score
            )
            if self.is_best_epoch:
                self.current_fold.best_epoch = epoch
                self.current_fold.best_score = score
                fold_best_state = copy.deepcopy(self.model.state_dict())

            self._step_scheduler(val_result)
            self._call_callbacks("on_epoch_end", fold, epoch)
            if self.should_stop:
                self.current_fold.stopped_early = True
                break

        if fold_best_state is not None:
            self.model.load_state_dict(fold_best_state)
            if self._is_better(self.current_fold.best_score, self._best_score):
                self._best_score = self.current_fold.best_score
                self._best_state = fold_best_state
                self.result.best_fold = fold
        self._call_callbacks("on_fold_end", fold)

    def _run_epoch(
        self, 
        dataloader: DataLoader, 
        step_function: Callable[[tuple], StepOutput]
    ) -> EpochResult:
        """
        Run `step_function` on every batch and average the results.

        :param dataloader: Data to run on.
        :type dataloader: DataLoader
        :param step_function: Either `training_step` or `validation_step`.
        :type step_function: Callable[[tuple], StepOutput]
        :return: Average loss, loss components and metrics.
        :rtype: EpochResult
        """
        losses: list[float] = []
        components: dict[str, list[float]] = defaultdict(list)
        metrics: dict[str, list[float]] = defaultdict(list)

        for batch in dataloader:
            output = step_function(batch)
            losses.append(output.loss.item())
            for name, value in output.loss_components.items():
                components[name].append(value)
            for name, value in self.compute_metrics(
                output.y_hat, output.y
            ).items():
                metrics[name].append(value)

        return EpochResult(
            loss=float(np.mean(losses)),
            loss_components=self._mean_per_key(components),
            metrics=self._mean_per_key(metrics)
        )

    def _step_scheduler(self, val_result: EpochResult) -> None:
        """
        Step the scheduler, if there is one.

        :param val_result: Validation result of the epoch that just 
            finished.
        :type val_result: EpochResult
        """
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step(val_result.loss)
        else:
            self.scheduler.step()

    def _is_better(self, score: float, best: float) -> bool:
        """
        Check if `score` is better than `best`, given the mode.

        :param score: Score to check.
        :type score: float
        :param best: Best score so far, NaN if there is none yet.
        :type best: float
        :return: True if `score` is better.
        :rtype: bool
        """
        if math.isnan(best):
            return True
        if self.config.mode == "min":
            return score < best
        return score > best

    def _autocast(self) -> torch.autocast:
        """
        :return: Autocast context, only active if `mixed_precision` is 
            set.
        :rtype: torch.autocast
        """
        return torch.autocast(
            device_type=torch.device(self.device).type,
            dtype=torch.bfloat16,
            enabled=self.config.mixed_precision
        )

    def _call_callbacks(self, hook: str, *args) -> None:
        """
        Call a hook on every callback.

        :param hook: Name of the hook, e.g. "on_epoch_end".
        :type hook: str
        :param args: Arguments to pass after the trainer.
        """
        for callback in self.callbacks:
            getattr(callback, hook)(self, *args)

    def _log_result(
        self, 
        split: str, 
        loss: float, 
        metrics: dict[str, float]
    ) -> None:
        """
        Log the loss and metrics of a split.

        :param split: Name of the split, e.g. "train".
        :type split: str
        :param loss: Loss to log.
        :type loss: float
        :param metrics: Metrics to log.
        :type metrics: dict[str, float]
        """
        metrics_string = ", ".join(
            f"{name}: {value:>2f}" for name, value in metrics.items()
        )
        self.logger.debug(f"{split} loss: {loss:>7f} | {metrics_string} |")

    @staticmethod
    def _mean_per_key(values: dict[str, list[float]]) -> dict[str, float]:
        """
        :param values: Dict mapping names to lists of values.
        :type values: dict[str, list[float]]
        :return: Dict mapping the same names to the mean of the lists.
        :rtype: dict[str, float]
        """
        return {name: float(np.mean(v)) for name, v in values.items()}

    @staticmethod
    def _as_dataloader(
        dataset_to_dataloader_function: Callable, 
        dataset: Dataset
    ) -> DataLoader:
        """
        Convert a dataset into a dataloader. The function may return a 
        tuple with the dataloader as first item.

        :param dataset_to_dataloader_function: Function that converts a
            Dataset into a DataLoader.
        :type dataset_to_dataloader_function: Callable
        :param dataset: Dataset to convert.
        :type dataset: Dataset
        :return: The dataloader.
        :rtype: DataLoader
        """
        dataloader = dataset_to_dataloader_function(dataset)
        return dataloader[0] if isinstance(dataloader, tuple) else dataloader
