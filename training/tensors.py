"""
Helpers to handle the output of a model, which is either a tensor, or a
tuple or dict of (nested) tensors. This allows models with multiple
outputs (e.g. an autoencoder that returns the reconstruction and the
hidden activations) to be trained by the same `Trainer`.
"""

from typing import Callable

import torch

ModelOutput = torch.Tensor | tuple | dict


def apply(
    output: ModelOutput,
    function: Callable[[torch.Tensor], torch.Tensor]
) -> ModelOutput:
    """
    Apply a function to every tensor of a model output.

    :param output: Tensor, or tuple or dict of (nested) tensors.
    :type output: ModelOutput
    :param function: Function to apply to every tensor.
    :type function: Callable[[torch.Tensor], torch.Tensor]
    :return: Output with the same structure, with the function applied.
    :rtype: ModelOutput
    """
    if isinstance(output, torch.Tensor):
        return function(output)
    if isinstance(output, dict):
        return {key: apply(value, function) for key, value in output.items()}
    return tuple(apply(value, function) for value in output)


def detach(output: ModelOutput) -> ModelOutput:
    """
    Detach every tensor of a model output from the graph.

    :param output: Tensor, or tuple or dict of (nested) tensors.
    :type output: ModelOutput
    :return: Output with the same structure, detached.
    :rtype: ModelOutput
    """
    return apply(output, lambda tensor: tensor.detach())


def to_cpu(output: ModelOutput) -> ModelOutput:
    """
    Move every tensor of a model output to the CPU.

    :param output: Tensor, or tuple or dict of (nested) tensors.
    :type output: ModelOutput
    :return: Output with the same structure, on the CPU.
    :rtype: ModelOutput
    """
    return apply(output, lambda tensor: tensor.cpu())


def concatenate(outputs: list[ModelOutput]) -> ModelOutput:
    """
    Concatenate model outputs of multiple batches along the batch
    dimension, keeping their structure.

    :param outputs: Outputs of all batches, all with the same structure.
    :type outputs: list[ModelOutput]
    :return: A single output with the same structure.
    :rtype: ModelOutput
    """
    first = outputs[0]
    if isinstance(first, torch.Tensor):
        return torch.cat(outputs)
    if isinstance(first, dict):
        return {
            key: concatenate([output[key] for output in outputs])
            for key in first
        }
    return tuple(concatenate(list(parts)) for parts in zip(*outputs))
