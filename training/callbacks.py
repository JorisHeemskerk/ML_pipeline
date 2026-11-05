"""
Callbacks that hook into the `Trainer`. 

Everything that is not the core training loop (early stopping, pruning,
checkpointing, ...) lives here, so the trainer itself stays small. To 
add behaviour, subclass `Callback` and override the hooks you need.

Callbacks that only need keyword arguments can also be selected by name
from a config. To add one, register it:

    @CALLBACKS.register("my_callback")
    class MyCallback(Callback):
        ...

It can then be requested with "my_callback", or with 
("my_callback", {"some_kwarg": 1}) to pass arguments.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Protocol

import torch

from training.registry import Registry
from training.visualisation import (
    plot_cross_validation_curves, 
    plot_training_curves
)

if TYPE_CHECKING:
    from training.trainer import Trainer

CALLBACKS = Registry("callback")


class Callback:
    """
    Base class for all callbacks, every hook does nothing by default.
    The trainer exposes what a callback needs: `trainer.model`,
    `trainer.loss_fn`, `trainer.optimiser`, `trainer.logger`, 
    `trainer.result`, `trainer.current_fold`, `trainer.is_best_epoch`, 
    `trainer.should_stop` and the dataloaders of the current fold: 
    `trainer.train_dataloader` and `trainer.val_dataloader`.
    """

    def on_fit_start(self, trainer: "Trainer") -> None:
        """
        Called once before the first fold starts.

        :param trainer: The trainer that calls this hook.
        :type trainer: Trainer
        """

    def on_fold_start(self, trainer: "Trainer", fold: int) -> None:
        """
        Called at the start of every fold, after the model, optimiser
        and scheduler have been reset.

        :param trainer: The trainer that calls this hook.
        :type trainer: Trainer
        :param fold: Index of the fold (0-indexed).
        :type fold: int
        """

    def on_epoch_end(self, trainer: "Trainer", fold: int, epoch: int) -> None:
        """
        Called after every epoch, once the results of that epoch are
        added to `trainer.current_fold`. Set `trainer.should_stop` to 
        end the fold early.

        :param trainer: The trainer that calls this hook.
        :type trainer: Trainer
        :param fold: Index of the fold (0-indexed).
        :type fold: int
        :param epoch: Index of the epoch (0-indexed).
        :type epoch: int
        """

    def on_fold_end(self, trainer: "Trainer", fold: int) -> None:
        """
        Called at the end of every fold, after the best weights of the
        fold have been restored.

        :param trainer: The trainer that calls this hook.
        :type trainer: Trainer
        :param fold: Index of the fold (0-indexed).
        :type fold: int
        """

    def on_fit_end(self, trainer: "Trainer") -> None:
        """
        Called once after the last fold, after the best weights across
        all folds have been restored.

        :param trainer: The trainer that calls this hook.
        :type trainer: Trainer
        """


class EarlyStopperLike(Protocol):
    """
    Anything with the interface of `EarlyStopper`.
    """

    def should_stop(self, value: float) -> bool: ...

    def reset(self) -> None: ...


class EarlyStoppingCallback(Callback):
    """
    Stop a fold when the early stopper decides the monitored validation
    value no longer improves.

    :param early_stopper: Object with `should_stop(value)` and
        `reset()`, such as the `EarlyStopper`.
    :type early_stopper: EarlyStopperLike
    :param monitor: Validation value to give to the early stopper: 
        "loss", a metric or a loss component. (DEFAULT="loss")
    :type monitor: str
    """

    def __init__(self, early_stopper: EarlyStopperLike, monitor: str="loss"):
        self.early_stopper = early_stopper
        self.monitor = monitor

    def on_fold_start(self, trainer: "Trainer", fold: int) -> None:
        self.early_stopper.reset()

    def on_epoch_end(self, trainer: "Trainer", fold: int, epoch: int) -> None:
        value = trainer.current_fold.val[-1].get(self.monitor)
        if self.early_stopper.should_stop(value):
            trainer.logger.warning(f"Decided to stop early, at epoch {epoch}")
            trainer.should_stop = True


class PruningCallback(Callback):
    """
    Report the validation value after every epoch to a pruner, for
    example the `OptunaPruningCallback`. The pruner is free to raise an
    exception (like `optuna.TrialPruned`) to abort the training.

    :param report: Function that takes the validation value.
    :type report: Callable[[float], None]
    :param monitor: Validation value to report: "loss", a metric or a
        loss component. (DEFAULT="loss")
    :type monitor: str
    """

    def __init__(self, report: Callable[[float], None], monitor: str="loss"):
        self.report = report
        self.monitor = monitor

    def on_epoch_end(self, trainer: "Trainer", fold: int, epoch: int) -> None:
        self.report(trainer.current_fold.val[-1].get(self.monitor))


class CheckpointCallback(Callback):
    """
    Save the model weights and the loss/metric curves up to that point.

    Files are saved to `output_dir/fold_<k>/`: `epoch_<n>.pt` and 
    `curves_epoch_<n>.png` every `every_n_epochs` epochs, `best.pt` 
    whenever the monitored validation score improves and
    `curves_final.png` at the end of the fold. After a cross validation,
    `cross_validation_curves.png` is saved to `output_dir`.

    :param output_dir: Directory to save everything in.
    :type output_dir: str | Path
    :param every_n_epochs: Number of epochs between checkpoints.
        (DEFAULT=5)
    :type every_n_epochs: int
    :param save_best: True to also save the best weights as `best.pt`.
        (DEFAULT=True)
    :type save_best: bool
    :param plot_curves: True to plot the curves at every checkpoint.
        (DEFAULT=True)
    :type plot_curves: bool
    """

    def __init__(
        self, 
        output_dir: str | Path, 
        every_n_epochs: int=5, 
        save_best: bool=True, 
        plot_curves: bool=True
    ):
        self.output_dir = Path(output_dir)
        self.every_n_epochs = every_n_epochs
        self.save_best = save_best
        self.plot_curves = plot_curves

    def _fold_dir(self, fold: int) -> Path:
        fold_dir = self.output_dir / f"fold_{fold + 1}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        return fold_dir

    def on_epoch_end(self, trainer: "Trainer", fold: int, epoch: int) -> None:
        fold_dir = self._fold_dir(fold)

        if self.save_best and trainer.is_best_epoch:
            torch.save(trainer.model.state_dict(), fold_dir / "best.pt")

        if (epoch + 1) % self.every_n_epochs != 0:
            return
        name = f"epoch_{epoch + 1:04d}"
        torch.save(trainer.model.state_dict(), fold_dir / f"{name}.pt")
        if self.plot_curves:
            plot_training_curves(
                trainer.current_fold, 
                fold_dir / f"curves_{name}.png",
                f"Fold {fold + 1}, epoch {epoch + 1}"
            )
        trainer.logger.info(f"Saved checkpoint {name} to {fold_dir}")

    def on_fold_end(self, trainer: "Trainer", fold: int) -> None:
        if self.plot_curves and trainer.current_fold.n_epochs > 0:
            plot_training_curves(
                trainer.current_fold, 
                self._fold_dir(fold) / "curves_final.png",
                f"Fold {fold + 1}"
            )

    def on_fit_end(self, trainer: "Trainer") -> None:
        if self.plot_curves and len(trainer.result.folds) > 1:
            plot_cross_validation_curves(
                trainer.result,
                self.output_dir / "cross_validation_curves.png",
                "Cross validation"
            )


def build_callbacks(
    specs: list[str | tuple[str, dict]]
) -> list[Callback]:
    """
    Build callbacks from their registered names.

    :param specs: Registered names, or (registered name, kwargs).
    :type specs: list[str | tuple[str, dict]]
    :return: The callbacks, in the order of `specs`.
    :rtype: list[Callback]
    """
    return list(CALLBACKS.build_all(specs).values())
