"""
DISCLAIMER:
This code was previously part of Joris Heemskerk's work at Technolution
BV, Gouda NL (see the publication "Privacy-Aware Traffic
Re-Identification with Interpretable Sparse Autoencoders", BNAIC 2025),
and is being re-used here. All rights are reserved to Joris Heemskerk,
and Technolution BV, Gouda NL.
"""

import logging
import torch
import torch.nn.functional as F
from torch import nn

from base import Base


class SparseAutoencoderLoss(Base, nn.Module):
    """
    Loss of a sparse autoencoder: the mean squared error of the
    reconstruction, plus an L1 penalty on the hidden activations to make
    them sparse.

    The penalty is the sum of the absolute activations of a datapoint,
    averaged over the batch, so `l1_coeff` does not depend on the batch
    size. (The original implementation summed over the batch too, so to
    reuse an old coefficient, multiply it by the old batch size.)

    Expects the output of the `SparseAutoencoder`. The mean squared error
    and the L1 penalty are tracked separately, as the loss components
    "mse" and "l1".
    """

    def __init__(
        self,
        device: str,
        label: str,
        logger: logging.Logger,
        l1_coeff: float
    ) -> None:
        """
        :param l1_coeff: Weight of the L1 sparsity penalty. A higher
            value gives fewer active features, but a worse
            reconstruction.
        :type l1_coeff: float
        """
        nn.Module.__init__(self)
        Base.__init__(self, device=device, label=label, logger=logger)
        self.l1_coeff = l1_coeff

    def forward(
        self,
        output: dict[str, torch.Tensor],
        y: torch.Tensor | None=None
    )-> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute the loss of a batch.

        :param output: Output of the `SparseAutoencoder`.
        :type output: dict[str, torch.Tensor]
        :param y: Targets of the batch, not used as the autoencoder
            recreates its own input. (DEFAULT=None)
        :type y: torch.Tensor | None
        :return: The total loss and the loss components "mse" and "l1".
        :rtype: tuple[torch.Tensor, dict[str, torch.Tensor]]
        """
        mse = F.mse_loss(output["reconstruction"], output["input"])
        l1 = self.l1_coeff * output["activations"].abs().sum(dim=-1).mean()
        return mse + l1, {"mse": mse.detach(), "l1": l1.detach()}

    def per_sample_loss(self, output: dict[str, torch.Tensor])-> torch.Tensor:
        """
        Compute the loss of every datapoint separately: the squared error
        summed over the dimensions, plus the L1 penalty. Used to find the
        datapoints that are reconstructed worst.

        :param output: Output of the `SparseAutoencoder`.
        :type output: dict[str, torch.Tensor]
        :return: Loss per datapoint, of shape (batch_size,).
        :rtype: torch.Tensor
        """
        squared_error = (output["reconstruction"] - output["input"]) ** 2
        l1 = self.l1_coeff * output["activations"].abs().sum(dim=-1)
        return squared_error.sum(dim=-1) + l1
