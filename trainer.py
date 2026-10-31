import logging

from torch.utils.data import Dataset, DataLoader

import modules as M

from job import Job


class Base_and_Dataset(Dataset, M.Base):
    pass

class Trainer:
    def __init__(self, job: Job, config: dict)-> None:
        self.job = job
        self.logger = self.job.logger
        self.CONFIG = config

        self.train_dataset: Base_and_Dataset = job.modules[
            self.CONFIG["train"]["data"]
        ]
        self.val_dataset: Base_and_Dataset = job.modules[
            self.CONFIG["val"]["data"]
        ]
        self.test_dataset: Base_and_Dataset = job.modules[
            self.CONFIG["test"]["data"]
        ]

        self.train_dataloader, self.val_dataloader, self.test_dataloader = \
            self._init_dataloaders()

    def _init_dataloaders(self)-> list[DataLoader]:
        dataloaders = []
        datasets = [self.train_dataset, self.val_dataset, self.test_dataset]
        labels = ["train", "val", "test"]
        for dataset, label in zip(datasets, labels):
            kwargs = {
                **self.CONFIG["loader_defaults"],
                # Override defaults if provided
                **(
                    self.CONFIG[label]["loader"] if 
                    "loader" in self.CONFIG[label] else {}
                )
            }
            self.logger.debug(f"Creating {label} dataloader with {kwargs}")
            dataloaders.append(DataLoader(
                dataset=dataset,
                **kwargs
            ))
        return dataloaders

    def validate_components(self)-> bool:
        """
        Invalid if IO of subsequent modules mismatch or are None.
        """
        if not (
            self.train_dataset.output_type ==
            self.val_dataset.output_type ==
            self.test_dataset.output_type
        ):
            self.logger.error(
                "Mismatch in output types of different dataset objects: "
                f"({self.train_dataset.output_type = }) == "
                f"({self.val_dataset.output_type = }) == "
                f"({self.test_dataset.output_type = })"
            )
            return False
            
        return True
        
        # for i in range(len(self.modules) - 1):
        #     if self.modules[i].output_type != self.modules[i+1].input_type:
        #         self.logger.error(
        #             f"Mismatch in input-output types of modules at index {i} "
        #             f"& {i+1} ({self.modules[i].__class__.__name__} & "
        #             f"{self.modules[i+1].__class__.__name__}). "
        #             f"{self.modules[i].output_type} != "
        #             f"{self.modules[i+1].input_type}"
        #         )
        #         return False
        #     elif self.modules[i].output_type == None:
        #         self.logger.error(
        #             f"Output type of Module {i} ("
        #             f"{self.modules[i].__class__.__name__}) is None"
        #         )
        #         return False
        # return True

    def __str__(self)-> str:
        return str(self.CONFIG)