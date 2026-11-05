"""
Metrics that can be selected by name. 

To add a metric, subclass `Metric` and register it:

    @METRICS.register("my_metric")
    class MyMetric(Metric):
        def __call__(self, y_hat, y) -> float:
            ...

It can then be used by passing "my_metric" to the trainer config.
"""

from abc import ABC, abstractmethod

import torch

from training.registry import Registry
from training.tensors import ModelOutput

METRICS = Registry("metric")


class Metric(ABC):
    """
    Base class for all metrics. A metric is evaluated per batch, the
    trainer averages the results over the batches of an epoch.
    """

    @abstractmethod
    def __call__(self, y_hat: ModelOutput, y: torch.Tensor) -> float:
        """
        Compute the metric for one batch.

        :param y_hat: Model output. A tensor, or a tuple or dict of 
            tensors if the model has multiple outputs.
        :type y_hat: ModelOutput
        :param y: Targets.
        :type y: torch.Tensor
        :return: Value of the metric.
        :rtype: float
        """


@METRICS.register("accuracy")
class Accuracy(Metric):
    """
    Fraction of samples for which the arg max of the predictions equals 
    the target class.
    """

    def __call__(self, y_hat: torch.Tensor, y: torch.Tensor) -> float:
        return (y_hat.argmax(dim=-1) == y).float().mean().item()


@METRICS.register("l0")
class L0Sparsity(Metric):
    """
    Average number of hidden features that are active (larger than 0) per
    datapoint. Expects the model output to be a dict with the key 
    "activations", of shape (batch_size, n_features), such as the output
    of the `SparseAutoencoder`.
    """

    def __call__(
        self, 
        y_hat: dict[str, torch.Tensor], 
        y: torch.Tensor
    ) -> float:
        return (y_hat["activations"] > 0).float().sum(dim=-1).mean().item()


@METRICS.register("explained_variance")
class ExplainedVariance(Metric):
    """
    Fraction of the variance of the input that is explained by the 
    reconstruction (1 is perfect). Expects the model output to be a dict
    with the keys "input" and "reconstruction", both of shape 
    (batch_size, n_dimensions), such as the output of the 
    `SparseAutoencoder`.
    """

    def __call__(
        self, 
        y_hat: dict[str, torch.Tensor], 
        y: torch.Tensor
    ) -> float:
        x, x_hat = y_hat["input"], y_hat["reconstruction"]
        residual = ((x - x_hat) ** 2).sum()
        total = ((x - x.mean(dim=0)) ** 2).sum()
        return (1 - residual / total.clamp_min(1e-8)).item()


def build_metrics(specs: list[str | tuple[str, dict]]) -> dict[str, Metric]:
    """
    Build metrics from their registered names.

    :param specs: Registered names, or (registered name, kwargs). A
        "name" entry in the kwargs renames the metric, which allows the
        same metric to be used multiple times.
    :type specs: list[str | tuple[str, dict]]
    :return: Dict mapping metric names to metrics.
    :rtype: dict[str, Metric]
    """
    return METRICS.build_all(specs)
