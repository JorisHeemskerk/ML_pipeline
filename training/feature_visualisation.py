"""
DISCLAIMER:
This code was previously part of Joris Heemskerk's work at Technolution
BV, Gouda NL (see the publication "Privacy-Aware Traffic
Re-Identification with Interpretable Sparse Autoencoders", BNAIC 2025),
and is being re-used here. All rights are reserved to Joris Heemskerk,
and Technolution BV, Gouda NL.

Visualisers for the features of a sparse autoencoder, to be used as
`test: visualisers` of the trainer config. They show which datapoints
activate which feature, and are grouped by class (e.g. the class of an
image) instead of by track.

All of them expect the model output to be a dict with the key
"activations", of shape (n_datapoints, n_features), such as the output of
the `SparseAutoencoder`. The visualisers that draw datapoints also need
the evaluated dataset to return images of shape (channels, height,
width) with values in [0, 1], and integer class targets.

Not every feature is interesting: some never activate, others only on a
few datapoints. The visualisers that draw single features therefore only
draw the `top_n` most relevant features, see `FeatureData.relevance`.

- `feature_statistics`: Range and non-zero count of every feature.
- `feature_trigger_histogram`: Number of features triggered per datapoint.
- `feature_gallery`: Per relevant feature, its top activating images
    grouped by class, above the distribution of its activations.
- `feature_summary`: All relevant features side by side, with one image
    per class and the distribution of the activations.
- `feature_contrast`: Per relevant feature, its top activating images
    next to random images that do not activate it.
"""

from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import ConnectionPatch
from torch.utils.data import Dataset

from training.results import EvaluationResult
from training.visualisation import VISUALISERS, Visualiser


def _min_max(values: np.ndarray) -> np.ndarray:
    """
    Scale values to the range [0, 1].

    :param values: Values to scale.
    :type values: np.ndarray
    :return: Scaled values, all 0 if the values are all the same.
    :rtype: np.ndarray
    """
    spread = values.max() - values.min()
    if spread == 0:
        return np.zeros(len(values))
    return (values - values.min()) / spread


def _save(fig: plt.Figure, path: Path) -> Path:
    """
    Save a figure and close it.

    :param fig: Figure to save.
    :type fig: plt.Figure
    :param path: Where to save the image, folders are created.
    :type path: Path
    :return: The path the image was saved to.
    :rtype: Path
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _tile(images: list[np.ndarray]) -> np.ndarray:
    """
    Tile images of identical size in a grid, as square as possible.

    :param images: Images of shape (height, width, channels).
    :type images: list[np.ndarray]
    :return: The tiled image, with a white background.
    :rtype: np.ndarray
    """
    columns = int(np.ceil(np.sqrt(len(images))))
    rows = int(np.ceil(len(images) / columns))
    height, width, channels = images[0].shape
    canvas = np.ones((rows * height, columns * width, channels))
    for i, image in enumerate(images):
        row, column = divmod(i, columns)
        canvas[
            row * height:(row + 1) * height,
            column * width:(column + 1) * width
        ] = image
    return canvas


@dataclass
class FeatureData:
    """
    Everything the feature visualisers need to know, in numpy.

    :param activations: Activation of every feature for every datapoint.
    :type activations: np.ndarray
    :param labels: Class of every datapoint.
    :type labels: np.ndarray
    :param class_names: Name of every class.
    :type class_names: list[str]
    :param dataset: Dataset the datapoints came from, None if unknown.
    :type dataset: Dataset | None
    """
    activations: np.ndarray
    labels: np.ndarray
    class_names: list[str]
    dataset: Dataset | None
    _images: dict[int, np.ndarray] = field(default_factory=dict, repr=False)

    @classmethod
    def from_result(
        cls,
        result: EvaluationResult,
        class_names: list[str] | None=None
    ) -> "FeatureData":
        """
        Extract the feature data from the result of an evaluation.

        :param result: Result of evaluating a model on a dataset.
        :type result: EvaluationResult
        :param class_names: Names of the classes. If None, the `classes`
            of the dataset are used, or else the class indices.
            (DEFAULT=None)
        :type class_names: list[str] | None
        :return: The feature data.
        :rtype: FeatureData
        """
        if not (
            isinstance(result.predictions, dict) and
            "activations" in result.predictions
        ):
            raise TypeError(
                "Feature visualisers require the model output to be a "
                "dict with the key 'activations'."
            )
        labels = result.targets.numpy()
        if class_names is None:
            class_names = getattr(result.dataset, "classes", None)
        if class_names is None:
            class_names = [str(i) for i in range(int(labels.max()) + 1)]
        return cls(
            activations=result.predictions["activations"].float().numpy(),
            labels=labels,
            class_names=list(class_names),
            dataset=result.dataset
        )

    @property
    def n_features(self) -> int:
        """Number of features."""
        return self.activations.shape[1]

    def class_name(self, label: int) -> str:
        """
        :param label: Class index.
        :type label: int
        :return: Name of the class.
        :rtype: str
        """
        label = int(label)
        if label < len(self.class_names):
            return self.class_names[label]
        return str(label)

    def image(self, index: int) -> np.ndarray:
        """
        Get the image of a datapoint.

        :param index: Index of the datapoint in the dataset.
        :type index: int
        :return: RGB image of shape (height, width, 3), values in [0, 1].
        :rtype: np.ndarray
        """
        if self.dataset is None:
            raise ValueError(
                "This visualiser draws images, but the evaluation result "
                "has no dataset (the dataloader of the evaluation may "
                "have shuffled)."
            )
        index = int(index)
        if index not in self._images:
            image = np.asarray(self.dataset[index][0], dtype=float)
            if image.shape[0] == 1:
                image = np.repeat(image, 3, axis=0)
            self._images[index] = np.clip(image.transpose(1, 2, 0), 0, 1)
        return self._images[index]

    def relevance(self) -> np.ndarray:
        """
        Score of how relevant every feature is: the normalised number of
        datapoints it activates on, plus its normalised largest
        activation (its range), so between 0 and 2.

        :return: Relevance per feature.
        :rtype: np.ndarray
        """
        non_zero_counts = np.count_nonzero(self.activations, axis=0)
        return _min_max(non_zero_counts) + _min_max(self.activations.max(0))

    def relevant_features(
        self,
        top_n: int,
        features: list[int] | None=None
    ) -> list[int]:
        """
        Select the features to visualise: the `top_n` most relevant
        features, never the ones that do not activate at all.

        :param top_n: Number of features to select.
        :type top_n: int
        :param features: Select exactly these features instead.
            (DEFAULT=None)
        :type features: list[int] | None
        :return: Indices of the features, most relevant first.
        :rtype: list[int]
        """
        alive = np.flatnonzero(np.count_nonzero(self.activations, axis=0))
        if features is not None:
            return [int(f) for f in features if int(f) in alive]
        ranked = alive[np.argsort(self.relevance()[alive])[::-1]]
        return ranked[:top_n].tolist()

    def top_datapoints(self, feature: int, n: int) -> np.ndarray:
        """
        Get the datapoints that activate a feature the most.

        :param feature: Index of the feature.
        :type feature: int
        :param n: Maximum number of datapoints.
        :type n: int
        :return: Indices of the datapoints, strongest activation first,
            only datapoints that activate the feature.
        :rtype: np.ndarray
        """
        activation = self.activations[:, feature]
        active = np.flatnonzero(activation > 0)
        return active[np.argsort(activation[active])[::-1][:n]]


class FeatureVisualiser(Visualiser):
    """
    Base class of the feature visualisers. To create a new one,
    implement `draw`.

    :param class_names: Names of the classes. If None, the `classes` of
        the dataset are used, or else the class indices. (DEFAULT=None)
    :type class_names: list[str] | None
    """
    # Folder in the output directory to save the images in.
    folder = ""

    def __init__(self, class_names: list[str] | None=None):
        self.class_names = class_names

    def __call__(
        self,
        result: EvaluationResult,
        output_dir: Path
    ) -> list[Path]:
        data = FeatureData.from_result(result, self.class_names)
        return self.draw(data, Path(output_dir) / self.folder)

    @abstractmethod
    def draw(self, data: FeatureData, output_dir: Path) -> list[Path]:
        """
        Create the visualisations.

        :param data: The feature data to visualise.
        :type data: FeatureData
        :param output_dir: Directory to save the images in.
        :type output_dir: Path
        :return: Paths of all images that were created.
        :rtype: list[Path]
        """


class RelevantFeatureVisualiser(FeatureVisualiser):
    """
    Base class of the feature visualisers that only look at the most
    relevant features.

    :param top_n: Number of most relevant features to visualise.
        (DEFAULT=16)
    :type top_n: int
    :param features: Visualise exactly these features instead of the
        most relevant ones. (DEFAULT=None)
    :type features: list[int] | None
    :param class_names: Names of the classes. (DEFAULT=None)
    :type class_names: list[str] | None
    """

    def __init__(
        self,
        top_n: int=16,
        features: list[int] | None=None,
        class_names: list[str] | None=None
    ):
        super().__init__(class_names)
        self.top_n = top_n
        self.features = features

    def select(self, data: FeatureData) -> list[int]:
        """
        :param data: The feature data.
        :type data: FeatureData
        :return: The features to visualise.
        :rtype: list[int]
        """
        return data.relevant_features(self.top_n, self.features)


@VISUALISERS.register("feature_statistics")
class FeatureStatistics(RelevantFeatureVisualiser):
    """
    Bar charts of the range (largest activation), the non-zero count
    (number of datapoints with an activation) and the relevance of every
    feature. The `top_n` of each chart is highlighted in red.
    """
    # Bars of more features than this are not labelled with their value.
    MAX_LABELLED = 256

    def draw(self, data: FeatureData, output_dir: Path) -> list[Path]:
        panels = [
            (
                data.activations.max(axis=0),
                "Feature ranges",
                "Maximum positive activation (i.e., range)"
            ),
            (
                np.count_nonzero(data.activations, axis=0),
                "Non-zero count per feature",
                "Number of non-zero results"
            ),
            (
                data.relevance(),
                "Summed normalised ranges and non-zero counts per feature",
                "Summed normalised ranges and non-zero counts"
            ),
        ]
        width = min(100, max(12, 0.25 * data.n_features))
        fig, axs = plt.subplots(nrows=3, figsize=(width, 12))
        for ax, (heights, title, ylabel) in zip(axs, panels):
            self._plot_bars(ax, heights, title, ylabel)
        fig.tight_layout()
        return [_save(fig, output_dir / "feature_statistics.png")]

    def _plot_bars(
        self,
        ax: plt.Axes,
        heights: np.ndarray,
        title: str,
        ylabel: str
    ) -> None:
        """
        Plot a bar for every feature, highlighting the highest `top_n`.

        :param ax: Axis to plot on.
        :type ax: plt.Axes
        :param heights: Height of every bar.
        :type heights: np.ndarray
        :param title: Title of the chart.
        :type title: str
        :param ylabel: Label of the y-axis.
        :type ylabel: str
        """
        highlighted = np.argsort(heights)[::-1][:self.top_n].tolist()
        container = ax.bar(
            range(len(heights)),
            heights,
            color=[
                "#d62728" if i in highlighted else "#1f77b4"
                for i in range(len(heights))
            ]
        )
        if len(heights) <= self.MAX_LABELLED:
            ax.bar_label(container, labels=np.round(heights, 2), fontsize=5)
        ax.set_title(title)
        ax.set_xlabel("Feature index")
        ax.set_ylabel(ylabel)
        ax.set_xlim(-1, len(heights))
        ax.set_xticks(range(0, len(heights), max(1, len(heights) // 100)))
        ax.tick_params(axis="x", labelsize=5)


@VISUALISERS.register("feature_trigger_histogram")
class FeatureTriggerHistogram(FeatureVisualiser):
    """
    Histogram of the number of features that are triggered per datapoint.

    :param max_xticks: Maximum number of values along the x-axis, which
        keeps the graph readable. (DEFAULT=10)
    :type max_xticks: int
    :param class_names: Names of the classes. (DEFAULT=None)
    :type class_names: list[str] | None
    """

    def __init__(
        self,
        max_xticks: int=10,
        class_names: list[str] | None=None
    ):
        super().__init__(class_names)
        self.max_xticks = max_xticks

    def draw(self, data: FeatureData, output_dir: Path) -> list[Path]:
        n_triggered = np.count_nonzero(data.activations, axis=1)
        labels, counts = np.unique(n_triggered, return_counts=True)

        fig, ax = plt.subplots(figsize=(10, 5))
        # A bar chart gives tighter control over the bins than a histogram.
        ax.bar(labels, counts, align="center")
        ax.set_title(
            "Histogram of number of features triggered per datapoint "
            f"(mean: {n_triggered.mean():.1f})"
        )
        ax.set_xlabel("Number of features triggered")
        ax.set_ylabel("Number of datapoints")

        if len(labels) > self.max_xticks:
            # Always include the min and max, snapped to existing labels.
            tick_values = np.linspace(
                labels.min(), labels.max(), self.max_xticks, dtype=int
            )
            ticks = np.unique(
                [labels[np.abs(labels - value).argmin()]
                for value in tick_values]
            )
        else:
            ticks = labels
        ax.set_xticks(ticks)
        return [_save(fig, output_dir / "feature_trigger_histogram.png")]


@VISUALISERS.register("feature_gallery")
class FeatureGallery(RelevantFeatureVisualiser):
    """
    One image per relevant feature, in 2 parts:

    - Top: The `n_images` datapoints with the highest activation, placed
        at their activation. Grouped by class, the most frequent class at
        the bottom.
    - Bottom: The distribution of all non-zero activations, stacked by
        density. The red lines mark the range of the top part.

    :param top_n: Number of most relevant features to visualise.
        (DEFAULT=16)
    :type top_n: int
    :param features: Visualise exactly these features instead.
        (DEFAULT=None)
    :type features: list[int] | None
    :param class_names: Names of the classes. (DEFAULT=None)
    :type class_names: list[str] | None
    :param n_images: Number of top images to show per feature.
        (DEFAULT=50)
    :type n_images: int
    :param zoom: Zoom factor of the images. (DEFAULT=1.5)
    :type zoom: float
    :param bin_interval: The activation interval for which points are
        stacked in the distribution. None for 1% of the largest
        activation of the feature. (DEFAULT=None)
    :type bin_interval: float | None
    """
    folder = "feature_gallery"

    def __init__(
        self,
        top_n: int=16,
        features: list[int] | None=None,
        class_names: list[str] | None=None,
        n_images: int=50,
        zoom: float=1.5,
        bin_interval: float | None=None
    ):
        super().__init__(top_n, features, class_names)
        self.n_images = n_images
        self.zoom = zoom
        self.bin_interval = bin_interval

    def draw(self, data: FeatureData, output_dir: Path) -> list[Path]:
        return [
            self._draw_feature(
                data, feature, output_dir / f"feature_{feature:04d}.png"
            )
            for feature in self.select(data)
        ]

    def _draw_feature(
        self,
        data: FeatureData,
        feature: int,
        path: Path
    ) -> Path:
        """
        Create the image of a single feature.

        :param data: The feature data.
        :type data: FeatureData
        :param feature: Index of the feature.
        :type feature: int
        :param path: Where to save the image.
        :type path: Path
        :return: The path the image was saved to.
        :rtype: Path
        """
        activation = data.activations[:, feature]
        top = data.top_datapoints(feature, self.n_images)

        # One row per class, the most frequent class at the bottom.
        classes, counts = np.unique(data.labels[top], return_counts=True)
        order = np.argsort(-counts, kind="stable")
        rows = {int(classes[i]): row for row, i in enumerate(order)}
        n_rows = len(rows)

        fig, (ax_top, ax_bottom) = plt.subplots(
            nrows=2,
            figsize=(12, 1.5 * n_rows + 4),
            gridspec_kw={"height_ratios": [1.5 * n_rows, 3]}
        )

        # Top: the most activating images, at their activation.
        for index in top:
            ax_top.add_artist(AnnotationBbox(
                OffsetImage(data.image(index), zoom=self.zoom),
                (activation[index], rows[int(data.labels[index])]),
                frameon=False,
                pad=0
            ))
        for row in range(n_rows + 1):
            ax_top.axhline(row - 0.5, color="gray", linestyle="--", lw=1)
        left, right = activation[top].min(), activation[top].max()
        padding = 0.1 * (right - left if right > left else right)
        ax_top.set_xlim(left - padding, right + padding)
        ax_top.set_ylim(-0.5, n_rows - 0.5)
        ax_top.set_yticks(range(n_rows))
        ax_top.set_yticklabels([
            f"{data.class_name(classes[i])} ({counts[i]})" for i in order
        ])
        ax_top.set_title(
            f"Feature {feature} of {data.n_features}: top {len(top)} images"
        )
        ax_top.set_xlabel("Activation")
        ax_top.set_ylabel("Class (number of images)")

        # Bottom: the distribution of all non-zero activations.
        values = activation[activation > 0]
        interval = self.bin_interval or values.max() / 100
        stacked = defaultdict(int)
        heights = []
        for value in values:
            bin_index = round(value / interval)
            heights.append(stacked[bin_index])
            stacked[bin_index] += 1
        ax_bottom.scatter(values, heights, s=4)
        ax_bottom.set_xlim(0, max(values.max(), ax_top.get_xlim()[1]) * 1.02)
        ax_bottom.set_ylim(bottom=-1)
        ax_bottom.set_yticks([])
        ax_bottom.set_title("Distribution of the non-zero activations")
        ax_bottom.set_xlabel("Activation")
        ax_bottom.set_ylabel("Approximate density")

        # Mark the range of the top part in the distribution.
        for x in ax_top.get_xlim():
            ax_bottom.axvline(max(0, x), color="red", linestyle="--")
            ax_bottom.add_artist(ConnectionPatch(
                xyA=(x, ax_top.get_ylim()[0]),
                xyB=(max(0, x), ax_bottom.get_ylim()[1]),
                coordsA="data",
                coordsB="data",
                axesA=ax_top,
                axesB=ax_bottom,
                color="red",
                linestyle="--",
                zorder=100
            ))
        return _save(fig, path)


@VISUALISERS.register("feature_summary")
class FeatureSummary(RelevantFeatureVisualiser):
    """
    A single image with a column for every relevant feature, to compare
    them at a glance. Every column shows the highest activating image of
    each of the (at most `max_groups`) most frequent classes among the
    `n_images` top images, with the number of those images, and the
    distribution of the non-zero activations below.

    :param top_n: Number of most relevant features to visualise.
        (DEFAULT=16)
    :type top_n: int
    :param features: Visualise exactly these features instead.
        (DEFAULT=None)
    :type features: list[int] | None
    :param class_names: Names of the classes. (DEFAULT=None)
    :type class_names: list[str] | None
    :param n_images: Number of top images to look at per feature.
        (DEFAULT=50)
    :type n_images: int
    :param max_groups: Maximum number of classes to show per feature.
        (DEFAULT=6)
    :type max_groups: int
    """

    def __init__(
        self,
        top_n: int=16,
        features: list[int] | None=None,
        class_names: list[str] | None=None,
        n_images: int=50,
        max_groups: int=6
    ):
        super().__init__(top_n, features, class_names)
        self.n_images = n_images
        self.max_groups = max_groups

    def draw(self, data: FeatureData, output_dir: Path) -> list[Path]:
        selected = self.select(data)
        if len(selected) == 0:
            return []

        fig, axs = plt.subplots(
            nrows=self.max_groups + 1,
            ncols=len(selected),
            figsize=(1.6 * len(selected), 1.9 * self.max_groups + 1.5),
            squeeze=False,
            gridspec_kw={"height_ratios": [1] * self.max_groups + [0.8]}
        )
        for column, feature in enumerate(selected):
            self._draw_column(data, feature, axs[:, column])
        fig.tight_layout()
        return [_save(fig, output_dir / "feature_summary.png")]

    def _draw_column(
        self,
        data: FeatureData,
        feature: int,
        axs: np.ndarray
    ) -> None:
        """
        Fill the column of a single feature.

        :param data: The feature data.
        :type data: FeatureData
        :param feature: Index of the feature.
        :type feature: int
        :param axs: The axes of the column: 1 for every group, and 1 for
            the distribution at the bottom.
        :type axs: np.ndarray
        """
        top = data.top_datapoints(feature, self.n_images)
        classes, counts = np.unique(data.labels[top], return_counts=True)
        order = np.argsort(-counts, kind="stable")[:self.max_groups]

        for row, ax in enumerate(axs[:-1]):
            ax.set_xticks([])
            ax.set_yticks([])
            if row >= len(order):
                ax.axis("off")
                continue
            label, count = classes[order[row]], counts[order[row]]
            # `top` is sorted, so the first match activates the most.
            ax.imshow(data.image(top[data.labels[top] == label][0]))
            ax.set_xlabel(f"{data.class_name(label)} ({count})", fontsize=7)
        axs[0].set_title(f"Feature {feature}", fontsize=9)

        activation = data.activations[:, feature]
        axs[-1].hist(activation[activation > 0], bins=20, color="#1f77b4")
        axs[-1].set_yticks([])
        axs[-1].set_xlabel("Activation", fontsize=7)
        axs[-1].tick_params(axis="x", labelsize=6)


@VISUALISERS.register("feature_contrast")
class FeatureContrast(RelevantFeatureVisualiser):
    """
    One image per relevant feature, with the `n_images` top activating
    images tiled on the left, and the same number of random images that
    do not activate the feature (a control) on the right. Whatever the
    left images share and the right ones do not, is what the feature
    encodes.

    :param top_n: Number of most relevant features to visualise.
        (DEFAULT=16)
    :type top_n: int
    :param features: Visualise exactly these features instead.
        (DEFAULT=None)
    :type features: list[int] | None
    :param class_names: Names of the classes. (DEFAULT=None)
    :type class_names: list[str] | None
    :param n_images: Number of images to tile per side. (DEFAULT=16)
    :type n_images: int
    :param seed: Seed to draw the control images with. (DEFAULT=0)
    :type seed: int
    """
    folder = "feature_contrast"

    def __init__(
        self,
        top_n: int=16,
        features: list[int] | None=None,
        class_names: list[str] | None=None,
        n_images: int=16,
        seed: int=0
    ):
        super().__init__(top_n, features, class_names)
        self.n_images = n_images
        self.seed = seed

    def draw(self, data: FeatureData, output_dir: Path) -> list[Path]:
        rng = np.random.default_rng(self.seed)
        paths = []
        for feature in self.select(data):
            silent = np.flatnonzero(data.activations[:, feature] == 0)
            control = rng.choice(
                silent, size=min(self.n_images, len(silent)), replace=False
            )
            top = data.top_datapoints(feature, self.n_images)

            fig, axs = plt.subplots(ncols=2, figsize=(10, 5.5))
            for ax, indices, title in zip(
                axs,
                (top, control),
                (
                    "Top activating images",
                    "Random images without activation (control)"
                )
            ):
                ax.axis("off")
                ax.set_title(title)
                if len(indices) > 0:
                    ax.imshow(_tile([data.image(i) for i in indices]))
            fig.suptitle(f"Feature {feature} of {data.n_features}")
            paths.append(_save(fig, output_dir / f"feature_{feature:04d}.png"))
        return paths
