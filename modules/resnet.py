import logging
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

import modules.types as T

from base_model import BaseModel


class ResNet(BaseModel):
    """
    A ResNet that turns images into embeddings, by removing the
    classification layer of the network. Any of the `ARCHITECTURES` can
    be used, with or without pretrained (ImageNet) weights, and the
    weights can be frozen so the ResNet only acts as a feature extractor.

    Images are expected as RGB tensors with values in [0, 1], the
    ImageNet normalisation is done inside this module. This way it works
    with any dataset, and the dataset can be plotted as is.
    """
    ARCHITECTURES = (
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152"
    )
    # ImageNet statistics, which the pretrained weights expect.
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        device: str,
        label: str,
        logger: logging.Logger,
        architecture: str = "resnet18",
        pretrained: bool = True,
        frozen: bool = True,
        input_size: int | None = None
    ) -> None:
        """
        Create a ResNet that outputs embeddings of size `embedding_dim`
        (512 for resnet18 and resnet34, 2048 for the others).

        :param architecture: Which ResNet to use, one of `ARCHITECTURES`.
            (DEFAULT="resnet18")
        :type architecture: str
        :param pretrained: True to start from the pretrained ImageNet
            weights, False for random weights. (DEFAULT=True)
        :type pretrained: bool
        :param frozen: True to freeze the weights. The network is then
            also kept in evaluation mode while training, so the batch
            normalisation statistics do not change. (DEFAULT=True)
        :type frozen: bool
        :param input_size: Resize the (square) images to this size before
            the network, None to keep the size of the images. The
            pretrained weights work best around 224, but that is slow.
            (DEFAULT=None)
        :type input_size: int | None
        """
        if architecture not in self.ARCHITECTURES:
            raise ValueError(
                f"Unknown architecture '{architecture}', choose from: "
                f"{', '.join(self.ARCHITECTURES)}."
            )
        super().__init__(device, label, logger)

        self.architecture = architecture
        self.pretrained = pretrained
        self.frozen = frozen
        self.input_size = input_size

        self.backbone = models.get_model(
            architecture,
            weights="DEFAULT" if pretrained else None
        )
        self.embedding_dim: int = self.backbone.fc.in_features
        # Remove the classification layer, to get the embeddings.
        self.backbone.fc = nn.Identity()

        self.register_buffer(
            "mean",
            torch.tensor(self.MEAN).view(1, 3, 1, 1),
            persistent=False
        )
        self.register_buffer(
            "std",
            torch.tensor(self.STD).view(1, 3, 1, 1),
            persistent=False
        )

        if frozen:
            self.requires_grad_(False)
            self.eval()

        self.input_type = T.TensorGeneric(dtype=torch.float32)
        self.output_type = T.TensorSpecific(
            dtype=torch.float32,
            shape=(self.embedding_dim,),
        )
        self.logger.debug(
            f"Created ResNet: {architecture = }, {pretrained = }, "
            f"{frozen = }, {input_size = }, "
            f"embedding_dim = {self.embedding_dim}"
        )

    def train(self, mode: bool=True)-> "ResNet":
        """
        Set the training mode. A frozen ResNet stays in evaluation mode,
        otherwise the batch normalisation layers would keep updating
        their statistics.

        :param mode: True for training mode, False for evaluation mode.
            (DEFAULT=True)
        :type mode: bool
        :return: This module.
        :rtype: ResNet
        """
        super().train(mode and not self.frozen)
        return self

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        """
        Embed a batch of images.

        :param x: RGB images with values in [0, 1], of shape
            (batch_size, 3, height, width).
        :type x: torch.Tensor
        :return: Embeddings of shape (batch_size, embedding_dim).
        :rtype: torch.Tensor
        """
        if self.input_size is not None:
            x = F.interpolate(
                x,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False
            )
        return self.backbone((x - self.mean) / self.std)
