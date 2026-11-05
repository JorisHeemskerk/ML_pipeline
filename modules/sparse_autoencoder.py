"""
DISCLAIMER:
This code was previously part of Joris Heemskerk's work at Technolution
BV, Gouda NL (see the publication "Privacy-Aware Traffic
Re-Identification with Interpretable Sparse Autoencoders", BNAIC 2025),
and is being re-used here. All rights are reserved to Joris Heemskerk,
and Technolution BV, Gouda NL.

Sparse autoencoder to find interpretable features in embeddings. The
architecture is based on
https://transformer-circuits.pub/2023/monosemantic-features#appendix-autoencoder.
"""

import logging
import torch
from torch import nn

import modules.types as T

from base_model import BaseModel


class SparseAutoencoder(BaseModel):
    """
    One-hidden-layer sparse autoencoder from Bricken, et al.

    The encoder uses weights W_e (e=encoder) and bias b_e, and the
    decoder uses weights W_d (d=decoder) (with unit-norm columns) and
    bias b_d. The autoencoder subtracts the decoder bias (b_d) from the
    inputs (pre-encoder bias). Every column of W_d is the vector of one
    learned feature.

    The output is a dict, so a loss can be computed from the output
    alone, no matter which modules came before this one:

    - "input": The embeddings that were given to the autoencoder.
    - "reconstruction": The recreated embeddings.
    - "activations": The hidden activations, shape (batch_size,
        hidden_dim). The sparse features.
    """

    def __init__(
        self,
        device: str,
        label: str,
        logger: logging.Logger,
        input_dim: int,
        hidden_dim: int
    ) -> None:
        """
        Create a sparse autoencoder with shape
        [input_dim, hidden_dim, input_dim].

        :param input_dim: Input dimensions (should be size of the
            embeddings).
        :type input_dim: int
        :param hidden_dim: Hidden dimensions (should be the same number
            as the desired amount of features).
        :type hidden_dim: int
        """
        super().__init__(device, label, logger)
        self.logger.info(
            f"Creating SparseAutoencoder with shape "
            f"[{input_dim}, {hidden_dim}, {input_dim}]."
        )
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Encoder parameters: W_e (size: hidden_dim x input_dim) and b_e
        # (size: hidden_dim).
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)

        # Decoder parameters: W_d (size: input_dim x hidden_dim) and b_d
        # (size: input_dim). The weights of the encoder and decoder are
        # not tied.
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)

        nn.init.kaiming_uniform_(self.encoder.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.decoder.weight, nonlinearity="linear")
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)
        self.normalise_decoder_weights()

        self.input_type = T.TensorSpecific(
            dtype=torch.float32,
            shape=(input_dim,),
        )
        # The output is a dict of tensors, which has no type yet.
        self.output_type = None
        self.logger.info("Done creating SparseAutoencoder.")

    @property
    def feature_vectors(self)-> torch.Tensor:
        """
        The learned feature vectors: the columns of the decoder weights.

        :return: Tensor of shape (hidden_dim, input_dim).
        :rtype: torch.Tensor
        """
        return self.decoder.weight.detach().T

    def feature_parameters(self)-> list[tuple[nn.Parameter, int]]:
        """
        The parameters that hold the features, with the dimension along
        which each feature is stored. Needed to reset the optimiser
        state of resampled features.

        :return: List of (parameter, feature dimension).
        :rtype: list[tuple[nn.Parameter, int]]
        """
        return [
            (self.encoder.weight, 0),
            (self.encoder.bias, 0),
            (self.decoder.weight, 1)
        ]

    def normalise_decoder_weights(self)-> None:
        """
        Normalise the decoder weights, so every learned feature vector
        has unit norm. Feature activations can then be compared, as they
        are the projection along these vectors.
        """
        with torch.no_grad():
            weight = self.decoder.weight  # shape: (input_dim, hidden_dim)
            # L2 norm of every column (dim=0), avoiding division by zero.
            norm = weight.norm(p=2, dim=0, keepdim=True) + 1e-8
            self.decoder.weight.copy_(weight / norm)

    def encode(self, x: torch.Tensor)-> torch.Tensor:
        """
        Encode embeddings into feature activations: subtract the
        decoder bias and apply the encoder with a ReLU.

        :param x: Embeddings of shape (batch_size, input_dim).
        :type x: torch.Tensor
        :return: Activations of shape (batch_size, hidden_dim).
        :rtype: torch.Tensor
        """
        x_bar = x - self.decoder.bias
        return torch.relu(self.encoder(x_bar))

    def forward(self, x: torch.Tensor)-> dict[str, torch.Tensor]:
        """
        Encode embeddings and decode them again.

        The decoder weights are normalised at the start of every pass,
        which is the same as normalising them after every optimiser
        step, so no hook is needed in the training loop.

        :param x: Embeddings of shape (batch_size, input_dim).
        :type x: torch.Tensor
        :return: The "input", its "reconstruction" and the "activations",
            see the class description.
        :rtype: dict[str, torch.Tensor]
        """
        self.normalise_decoder_weights()
        activations = self.encode(x)
        return {
            "input": x,
            "reconstruction": self.decoder(activations),
            "activations": activations
        }

    def reinitialise_features(
        self,
        dead: torch.Tensor,
        inputs: torch.Tensor
    )-> None:
        """
        Reinitialise dead features (features that never activate) to
        point towards given inputs, so they can start learning again.

        The decoder column of a feature becomes its input, normalised to
        unit length. The encoder row becomes that same direction, scaled
        to 20% of the average norm of the encoder rows of the features
        that are alive. Encoder biases are reset to 0.

        :param dead: Boolean mask of shape (hidden_dim,), True for the
            features to reinitialise.
        :type dead: torch.Tensor
        :param inputs: One input for every dead feature, of shape
            (number of dead features, input_dim).
        :type inputs: torch.Tensor
        """
        with torch.no_grad():
            alive_weights = self.encoder.weight[~dead]
            alive_norm = alive_weights.norm(p=2, dim=1).mean() \
                if alive_weights.shape[0] > 0 else 1.0

            norms = inputs.norm(p=2, dim=1, keepdim=True)
            directions = inputs / (norms + 1e-8)
            directions = directions.to(self.decoder.weight.dtype)

            self.decoder.weight[:, dead] = directions.T
            self.encoder.weight[dead] = directions * (alive_norm * 0.2)
            self.encoder.bias[dead] = 0.0
