import logging
import torch
from torch import nn

from base_model import BaseModel


class NeuralNetwork(BaseModel):
    """
    A generic NN class for any shape NN.
    """
    def __init__(
        self,
        device: int,
        label: int,
        logger: logging.Logger,
        network_shape: list[dict[str, dict]],
        *args,
        **kwargs
    ) -> None:
        """
        Define a simple neural network.

        :param logger: Logger to log to.
        :type logger: logging.Logger
        """
        super().__init__(device, label, logger, *args, **kwargs)

        layers = []
        for item in network_shape:
            name = next(iter(item))
            config = item[name]
            if config is None:
                config = {}
            layers.append(getattr(nn, name)(**config))
 
        self.layers = nn.Sequential(*layers)
        self.logger.debug(f"Created NeuralNetwork: {self}")
        
        self._initialise_weights()

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        """
        Perform a forward pass on the network.

        :param x: Input tensor of shape (batch_size, window_size, 
            input_size).
        :type x: torch.Tensor
        :return: Output tensor of shape (batch_size, 1).
        :rtype: torch.Tensor
        """
        return self.layers(x)
