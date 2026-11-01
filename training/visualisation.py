"""
Plotting for the trainer. 

There are two kinds of plots:

- Training curves, built from the results of the training. These are
  plain functions as they work for every model.
- Evaluation visualisers, built from the result of evaluating on a
  dataset. Every model needs different plots, so these are classes that
  are selected by name. 

To add a visualiser, subclass `Visualiser` and register it:

    @VISUALISERS.register("my_visualiser")
    class MyVisualiser(Visualiser):
        def __call__(self, result, output_dir) -> list[Path]:
            ...

It can then be requested with "my_visualiser", or with 
("my_visualiser", {"some_kwarg": 1}) to pass arguments.
"""

import math
from abc import ABC, abstractmethod
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from training.registry import Registry
from training.results import EvaluationResult, FoldResult, TrainingResult

VISUALISERS = Registry("visualiser")


def _subplot_grid(n_plots: int, n_cols: int = 3):
    """
    Create a grid of subplots with at least `n_plots` axes. Unused axes
    are hidden.

    :param n_plots: Number of plots needed.
    :type n_plots: int
    :param n_cols: Maximum number of columns. (DEFAULT=3)
    :type n_cols: int
    :return: The figure and a flat list of axes.
    :rtype: tuple[plt.Figure, list[plt.Axes]]
    """
    n_cols = min(n_plots, n_cols)
    n_rows = math.ceil(n_plots / n_cols)
    fig, axes = plt.subplots(
        n_rows, 
        n_cols, 
        figsize=(5 * n_cols, 4 * n_rows), 
        squeeze=False
    )
    axes = list(axes.flat)
    for ax in axes[n_plots:]:
        ax.set_visible(False)
    return fig, axes


def plot_training_curves(
    fold: FoldResult, 
    path: Path, 
    title: str = ""
) -> Path:
    """
    Plot the train and validation curve of the loss, every loss 
    component and every metric of a fold, one subplot each.

    :param fold: Results of the fold to plot, may be unfinished.
    :type fold: FoldResult
    :param path: Where to save the image.
    :type path: Path
    :param title: Title of the figure. (DEFAULT="")
    :type title: str
    :return: The path the image was saved to.
    :rtype: Path
    """
    fig, axes = _subplot_grid(len(fold.keys))
    for ax, key in zip(axes, fold.keys):
        ax.plot(fold.curve("train", key), label="train")
        ax.plot(fold.curve("val", key), label="validation")
        ax.set_title(key)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_cross_validation_curves(
    result: TrainingResult, 
    path: Path, 
    title: str = ""
) -> Path:
    """
    Plot the mean and standard deviation over the folds of the train and
    validation curve of the loss and every metric, one subplot each.

    :param result: Results of all folds.
    :type result: TrainingResult
    :param path: Where to save the image.
    :type path: Path
    :param title: Title of the figure. (DEFAULT="")
    :type title: str
    :return: The path the image was saved to.
    :rtype: Path
    """
    keys = ["loss", *result.folds[0].metric_names]
    fig, axes = _subplot_grid(len(keys))
    for ax, key in zip(axes, keys):
        for split in ("train", "val"):
            mean, std = result.mean_curve(split, key)
            ax.plot(mean, label=split)
            ax.fill_between(
                range(len(mean)), mean - std, mean + std, alpha=0.2
            )
        ax.set_title(key)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


class Visualiser(ABC):
    """
    Base class for all evaluation visualisers.
    """

    @abstractmethod
    def __call__(
        self, 
        result: EvaluationResult, 
        output_dir: Path
    ) -> list[Path]:
        """
        Create visualisations of an evaluation.

        :param result: Result of evaluating a model on a dataset.
        :type result: EvaluationResult
        :param output_dir: Directory to save the images in.
        :type output_dir: Path
        :return: Paths of all images that were created.
        :rtype: list[Path]
        """


@VISUALISERS.register("confusion_matrix")
class ConfusionMatrix(Visualiser):
    """
    Confusion matrix for classification, expects predictions of shape 
    (N, n_classes) and integer targets of shape (N,).

    :param class_names: Names of the classes, defaults to the indices.
    :type class_names: list[str] | None
    :param normalise: True to normalise every row to sum to 1.
    :type normalise: bool
    """

    def __init__(
        self, 
        class_names: list[str] | None = None, 
        normalise: bool = False
    ):
        self.class_names = class_names
        self.normalise = normalise

    def __call__(
        self, 
        result: EvaluationResult, 
        output_dir: Path
    ) -> list[Path]:
        predicted = result.predictions.argmax(dim=-1).numpy()
        actual = result.targets.numpy()
        n_classes = result.predictions.shape[-1]

        matrix = np.zeros((n_classes, n_classes))
        np.add.at(matrix, (actual, predicted), 1)
        if self.normalise:
            matrix = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)

        names = self.class_names or [str(i) for i in range(n_classes)]
        fig, ax = plt.subplots(figsize=(8, 7))
        image = ax.imshow(matrix, cmap="Blues")
        fig.colorbar(image, ax=ax)
        ax.set_xticks(range(n_classes), names, rotation=45, ha="right")
        ax.set_yticks(range(n_classes), names)
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
        ax.set_title("Confusion matrix")
        if n_classes <= 20:
            for i in range(n_classes):
                for j in range(n_classes):
                    ax.text(
                        j, 
                        i, 
                        f"{matrix[i, j]:.2f}" if self.normalise 
                        else f"{int(matrix[i, j])}",
                        ha="center", 
                        va="center",
                        color="white" if matrix[i, j] > matrix.max() / 2 
                        else "black"
                    )
        fig.tight_layout()

        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "confusion_matrix.png"
        fig.savefig(path)
        plt.close(fig)
        return [path]


def build_visualisers(
    specs: list[str | tuple[str, dict]]
) -> dict[str, Visualiser]:
    """
    Build visualisers from their registered names.

    :param specs: Registered names, or (registered name, kwargs).
    :type specs: list[str | tuple[str, dict]]
    :return: Dict mapping keys to visualisers.
    :rtype: dict[str, Visualiser]
    """
    return VISUALISERS.build_all(specs)
