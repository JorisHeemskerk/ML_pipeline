from typing import Tuple

import logging
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
 
import modules.types as T

from base import Base


class MNISTData(Base, Dataset):
    """MNIST as a torch Dataset."""
 
    MEAN = 0.1307
    STD = 0.3081
 
    def __init__(
        self,
        device: str,
        label: str,
        root: str,
        train: bool,
        normalise: bool,
        logger: logging.Logger,
        download: bool = True,
    ) -> None:
        Base.__init__(self, device=device, label=label, logger=logger)

        # Shapes/dtypes of what this dataset yields
        self.input_type = T.TensorSpecific(
            dtype=torch.float32,
            shape=(1, 28, 28),
        )
        self.output_type = T.TensorGeneric(dtype=torch.int64)
 
        self.train = train
        self.normalise = normalise

        tfms = [transforms.ToTensor()]
        if normalise:
            tfms.append(transforms.Normalize((self.MEAN), (self.STD)))

        self.logger.debug("Loading/downloading MNIST data")
        self.data = datasets.MNIST(
            root=root,
            train=train,
            transform=transforms.Compose(tfms),
            download=download,
        )
        self.logger.debug(f"Finished creating MNISTData object: {self}")
 
    def __len__(self) -> int:
        return len(self.data)
 
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Stays on CPU so it is safe to use with num_workers > 0
        return self.data[idx]

    def __repr__(self)-> str:
        return (
            f"MNISTData("
            f"label={self.label}, "
            f"n_datapoints={len(self)}, "
            f"train={self.train}, "
            f"normalise={self.normalise}, "
            f"input_type={self.input_type}, "
            f"output_type={self.output_type})"
        )