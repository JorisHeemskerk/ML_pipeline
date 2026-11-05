"""
DISCLAIMER:
This code was previously part of Joris Heemskerk's work at Technolution
BV, Gouda NL (see the publication "Privacy-Aware Traffic
Re-Identification with Interpretable Sparse Autoencoders", BNAIC 2025),
and is being re-used here. All rights are reserved to Joris Heemskerk,
and Technolution BV, Gouda NL.

Callback that revives dead features of a sparse autoencoder, as
described in
https://transformer-circuits.pub/2023/monosemantic-features#appendix-autoencoder.
"""

from typing import TYPE_CHECKING

import torch
from torch import nn

from training.callbacks import CALLBACKS, Callback

if TYPE_CHECKING:
    from training.trainer import Trainer


@CALLBACKS.register("dead_feature_resampling")
class DeadFeatureResamplingCallback(Callback):
    """
    Every `every_n_epochs` epochs, find the features of a sparse
    autoencoder that never activate on the training data (dead features)
    and reinitialise them, so they can start learning again.

    A random sample of the training data is run through the model. The
    features that do not activate on any of it are dead. Every dead
    feature is reinitialised towards an input from that sample, drawn
    with a probability proportional to the square of the loss of that
    input, so the features that are reconstructed worst get the most
    attention. The Adam state of the reinitialised features is reset as
    well.

    This callback requires:

    - A module somewhere in the model with a `reinitialise_features` and
      a `feature_parameters` method, such as the `SparseAutoencoder`.
    - A model output that is a dict with the keys "input" and
      "activations".
    - A loss function with a `per_sample_loss(output)` method, such as
      the `SparseAutoencoderLoss`.
    - An optimiser that keeps Adam state (`exp_avg` and `exp_avg_sq`).

    :param every_n_epochs: Number of epochs between resamplings.
        (DEFAULT=10)
    :type every_n_epochs: int
    :param until_epoch: Stop resampling after this epoch (1-indexed), so
        the features can settle at the end of the training. None to keep
        resampling. (DEFAULT=None)
    :type until_epoch: int | None
    :param n_samples: Number of training datapoints to look for dead
        features and to draw inputs from. (DEFAULT=20000)
    :type n_samples: int
    """

    def __init__(
        self,
        every_n_epochs: int=10,
        until_epoch: int | None=None,
        n_samples: int=20000
    ):
        self.every_n_epochs = every_n_epochs
        self.until_epoch = until_epoch
        self.n_samples = n_samples

    def on_epoch_end(self, trainer: "Trainer", fold: int, epoch: int) -> None:
        epochs_done = epoch + 1
        if epochs_done % self.every_n_epochs != 0:
            return
        if self.until_epoch is not None and epochs_done > self.until_epoch:
            return

        autoencoder = self._find_autoencoder(trainer.model)
        inputs, losses, fired = self._sample(trainer)
        dead = ~fired
        n_dead = int(dead.sum())
        if n_dead == 0:
            trainer.logger.info(
                f"No dead features to resample after epoch {epochs_done}."
            )
            return

        # Draw an input for every dead feature, with a probability
        # proportional to the square of the loss of that input.
        weights = losses ** 2
        if weights.sum() <= 0:
            weights = torch.ones_like(weights)
        chosen = torch.multinomial(weights, n_dead, replacement=True)

        autoencoder.reinitialise_features(dead, inputs[chosen])
        self._reset_optimiser_state(
            trainer.optimiser,
            autoencoder.feature_parameters(),
            dead.nonzero(as_tuple=True)[0]
        )
        trainer.logger.info(
            f"Resampled {n_dead}/{len(dead)} dead features after epoch "
            f"{epochs_done}."
        )

    @staticmethod
    def _find_autoencoder(model: nn.Module) -> nn.Module:
        """
        Find the module that can resample its features, anywhere in the
        model.

        :param model: The model that is trained.
        :type model: nn.Module
        :return: The module with the features.
        :rtype: nn.Module
        """
        found = [
            module for module in model.modules()
            if hasattr(module, "reinitialise_features")
        ]
        if len(found) != 1:
            raise ValueError(
                "Resampling features requires exactly 1 module with a "
                f"`reinitialise_features` method in the model, found "
                f"{len(found)}."
            )
        return found[0]

    def _sample(
        self,
        trainer: "Trainer"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run training data through the model until `n_samples` datapoints
        are seen.

        :param trainer: The trainer that calls this callback.
        :type trainer: Trainer
        :return: The inputs of the autoencoder, the loss of each of them
            and a boolean mask of the features that fired at least once.
        :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        was_training = trainer.model.training
        trainer.model.eval()

        inputs, losses = [], []
        fired = None
        n_seen = 0
        with torch.no_grad():
            for batch in trainer.train_dataloader:
                output, _ = trainer.forward(batch)
                inputs.append(output["input"])
                losses.append(trainer.loss_fn.per_sample_loss(output))
                batch_fired = (output["activations"] > 0).any(dim=0)
                fired = batch_fired if fired is None else fired | batch_fired
                n_seen += len(output["input"])
                if n_seen >= self.n_samples:
                    break

        trainer.model.train(was_training)
        return torch.cat(inputs), torch.cat(losses), fired

    @staticmethod
    def _reset_optimiser_state(
        optimiser: torch.optim.Optimizer,
        parameters: list[tuple[nn.Parameter, int]],
        indices: torch.Tensor
    ) -> None:
        """
        Reset the Adam state of the reinitialised features, as the
        momentum of the old features is meaningless for the new ones.

        :param optimiser: The optimiser of the model.
        :type optimiser: torch.optim.Optimizer
        :param parameters: The parameters that hold the features, with
            the dimension along which each feature is stored.
        :type parameters: list[tuple[nn.Parameter, int]]
        :param indices: Indices of the reinitialised features.
        :type indices: torch.Tensor
        """
        for parameter, dim in parameters:
            state = optimiser.state.get(parameter, {})
            for key in ("exp_avg", "exp_avg_sq"):
                if key in state:
                    state[key].index_fill_(dim, indices, 0.0)
