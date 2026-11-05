import logging
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

import modules.types as T

from base import Base


class CIFAR10Data(Base, Dataset):
    """
    CIFAR-10 as a torch Dataset. Images are RGB tensors with values in
    the range [0, 1], targets are the class indices. The names of the
    classes are available as `classes`, so visualisers can use them.
    """

    def __init__(
        self,
        device: str,
        label: str,
        root: str,
        train: bool,
        logger: logging.Logger,
        download: bool = True,
    ) -> None:
        """
        Load (and download if necessary) CIFAR-10.

        :param root: Folder to store the data in.
        :type root: str
        :param train: True for the training set, False for the test set.
        :type train: bool
        :param download: True to download the data if it is not present.
            (DEFAULT=True)
        :type download: bool
        """
        Base.__init__(self, device=device, label=label, logger=logger)

        # Shapes/dtypes of what this dataset yields.
        self.input_type = T.TensorSpecific(
            dtype=torch.float32,
            shape=(3, 32, 32),
        )
        self.output_type = T.TensorGeneric(dtype=torch.int64)

        self.train = train

        self.logger.debug("Loading/downloading CIFAR-10 data")
        self.data = datasets.CIFAR10(
            root=root,
            train=train,
            transform=transforms.ToTensor(),
            download=download,
        )
        self.classes: list[str] = self.data.classes
        self.logger.debug(f"Finished creating CIFAR10Data object: {self}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        # Stays on CPU so it is safe to use with num_workers > 0
        return self.data[idx]

    def __repr__(self)-> str:
        return (
            f"CIFAR10Data("
            f"label={self.label}, "
            f"n_datapoints={len(self)}, "
            f"train={self.train}, "
            f"input_type={self.input_type}, "
            f"output_type={self.output_type})"
        )
