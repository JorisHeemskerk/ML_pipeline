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

METRICS = Registry("metric")


class Metric(ABC):
    """
    Base class for all metrics. A metric is evaluated per batch, the
    trainer averages the results over the batches of an epoch.
    """

    @abstractmethod
    def __call__(self, y_hat: torch.Tensor, y: torch.Tensor) -> float:
        """
        Compute the metric for one batch.

        :param y_hat: Model predictions.
        :type y_hat: torch.Tensor
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
