import logging
import torch
from torch import nn

from base import Base


class BaseModel(Base, nn.Module):
    """
    A base model class that can be used to build other specific models 
    on top of.
    """
    def __init__(
        self, 
        device: str, 
        label: str, 
        logger: logging.Logger, 
        *args, 
        **kwargs
    )-> None:
        Base.__init__(self, device=device, label=label, logger=logger)
        nn.Module.__init__(self, *args, **kwargs)

    def _initialise_weights(self) -> None:
        """
        Apply appropriate weight initialisation across all layer types:
        - Linear & Conv layers: Xavier normal for weights, 
            zeros for biases.
        - Recurrent layers (LSTM, GRU, RNN): Xavier normal for weights, 
            zeros for biases.
        - Normalisation layers (BatchNorm, LayerNorm, etc.): ones for 
            weights, zeros for biases.
        - Embedding layers: Standard normal distribution.
        - MultiheadAttention: Xavier normal for weights, 
            zeros for biases.
        """
        self.logger.debug("Initialising weights")

        for module in self.modules():
            # Linear & Convolutional 
            if isinstance(module, (
                nn.Linear,
                nn.Conv1d, 
                nn.Conv2d, 
                nn.Conv3d,
                nn.ConvTranspose1d, 
                nn.ConvTranspose2d, 
                nn.ConvTranspose3d,
            )):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            # Recurrent
            elif isinstance(module, (nn.LSTM, nn.GRU, nn.RNN)):
                for name, param in module.named_parameters():
                    if "weight" in name:
                        nn.init.xavier_normal_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)

            # Normalisation
            elif isinstance(module, (
                nn.BatchNorm1d, 
                nn.BatchNorm2d, 
                nn.BatchNorm3d,
                nn.LayerNorm, 
                nn.GroupNorm,
                nn.InstanceNorm1d, 
                nn.InstanceNorm2d, 
                nn.InstanceNorm3d,
            )):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            # Embedding
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight)
                if module.padding_idx is not None:
                    nn.init.zeros_(module.weight[module.padding_idx])

            # MultiheadAttention
            elif isinstance(module, nn.MultiheadAttention):
                for name, param in module.named_parameters():
                    if "weight" in name:
                        nn.init.xavier_normal_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        raise NotImplementedError

    def save(self, destination: str)-> None:
        """
        Save internal state to file.

        :param destination: File to output model to.
        :type destination: str
        :param prefix: prefix for the filename.
        :type prefix: str
        """
        self.logger.info(f"Saving model to {destination}...")
        torch.save(self, destination)

    @classmethod
    def load(
        cls, 
        source: str, 
        logger: logging.Logger, 
        weights_only: bool=False
    )-> "BaseModel":
        """
        Load a model from a file.

        :param source: Source path.
        :type source: str
        :param logger: Logger to assign to the loaded model, as it would
            otherwise load the old logger.
        :type logger: logging.Logger
        :param weights_only: Whether to only load weights with PyTorch.
        :type weights_only: bool (DEFAULT=False)
        :return: The loaded model instance.
        :rtype: "BaseModel"
        """
        logger.info(f"Loading model from {source}...")
        model = torch.load(source, weights_only=weights_only)
        model.logger = logger
        return model
