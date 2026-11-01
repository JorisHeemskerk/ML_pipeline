"""
Config driven trainer of a job.

Reads the `trainer` section of the job config and builds everything the
training needs: the datasets and the model come from the modules of the
job, the loss, optimiser and scheduler from torch (or from a module of
the job, for custom ones). Everything is then handed to the `Trainer`
engine of the `training` package, which does the actual training.
"""

from functools import partial
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

import modules as M

from job import Job
from training import (
    Callback,
    CheckpointCallback,
    EarlyStopper,
    EarlyStoppingCallback,
    Trainer as Engine,
    TrainerConfig,
    TrainingResult
)


class Base_and_Dataset(Dataset, M.Base):
    pass

class Trainer:
    def __init__(self, job: Job, config: dict)-> None:
        """
        Read the trainer config and create the dataloaders.

        The `mode` of the config decides how the model is trained:

        - `holdout`: train on the `train` data, validate on the `val`
            data. If `val` points to the same module as `train`, a
            `fraction` of it is held out for validation.
        - `kfold`: k-fold cross validation on the `train` data, with `k`
            from the `kfold` section. No `val` section is needed.

        :param job: Job this trainer belongs to.
        :type job: Job
        :param config: The `trainer` section of the job config.
        :type config: dict
        """
        self.job = job
        self.logger = self.job.logger
        self.CONFIG = config
        self.output_dir = Path(self.job.output_folder)

        self.mode = self.CONFIG.get("mode", "holdout")
        if self.mode not in ("holdout", "kfold"):
            raise ValueError(
                f"Trainer mode must be 'holdout' or 'kfold', not "
                f"'{self.mode}'."
            )

        self.train_dataset: Base_and_Dataset = job.modules[
            self.CONFIG["train"]["data"]
        ]
        self.test_dataset: Base_and_Dataset = job.modules[
            self.CONFIG["test"]["data"]
        ]
        self.val_dataset: Base_and_Dataset | None = None
        if self.mode == "holdout":
            if "val" not in self.CONFIG:
                raise ValueError("Mode 'holdout' requires a `val` section.")
            self.val_dataset = job.modules[self.CONFIG["val"]["data"]]

        self._init_dataloaders()

    def _loader_kwargs(self, label: str)-> dict:
        """
        Get the dataloader kwargs of a split: the `loader_defaults`,
        overridden by the `loader` of the split itself.

        :param label: Name of the split ("train", "val" or "test").
        :type label: str
        :return: Keyword arguments for the DataLoader.
        :rtype: dict
        """
        return {
            **(self.CONFIG.get("loader_defaults") or {}),
            **((self.CONFIG.get(label) or {}).get("loader") or {})
        }

    def _to_dataloader(self, dataset: Dataset, label: str)-> DataLoader:
        """
        Convert a dataset into a dataloader using the config of a split.

        :param dataset: Dataset to convert.
        :type dataset: Dataset
        :param label: Name of the split ("train", "val" or "test").
        :type label: str
        :return: The dataloader.
        :rtype: DataLoader
        """
        kwargs = self._loader_kwargs(label)
        self.logger.debug(f"Creating {label} dataloader with {kwargs}")
        return DataLoader(dataset=dataset, **kwargs)

    def _split_holdout(self)-> tuple[Dataset, Dataset]:
        """
        Get the datasets to train and validate on in holdout mode.

        :return: Training dataset and validation dataset.
        :rtype: tuple[Dataset, Dataset]
        """
        if self.val_dataset is not self.train_dataset:
            return self.train_dataset, self.val_dataset

        val_config = self.CONFIG["val"]
        n_val = int(len(self.train_dataset) * val_config["fraction"])
        generator = torch.Generator().manual_seed(val_config.get("seed", 42))
        self.logger.info(
            f"Holding out {n_val} of {len(self.train_dataset)} training "
            "datapoints for validation."
        )
        train_data, val_data = random_split(
            self.train_dataset,
            [len(self.train_dataset) - n_val, n_val],
            generator=generator
        )
        return train_data, val_data

    def _init_dataloaders(self)-> None:
        """
        Create the dataloaders. In kfold mode the train and validation
        loaders are created per fold by the engine instead.
        """
        self.test_dataloader = self._to_dataloader(self.test_dataset, "test")
        if self.mode == "kfold":
            return

        train_data, val_data = self._split_holdout()
        self.train_dataloader = self._to_dataloader(train_data, "train")
        self.val_dataloader = self._to_dataloader(val_data, "val")

    def validate_components(self)-> bool:
        """
        Invalid if IO of subsequent modules mismatch or are None, or if
        the config refers to something that is missing.
        """
        datasets = {"train": self.train_dataset, "test": self.test_dataset}
        if self.val_dataset is not None:
            datasets["val"] = self.val_dataset
        output_types = {
            label: dataset.output_type for label, dataset in datasets.items()
        }
        if len(set(output_types.values())) != 1:
            self.logger.error(
                "Mismatch in output types of different dataset objects: "
                f"{output_types}"
            )
            return False

        if self.CONFIG["model"] not in self.job.modules:
            self.logger.error(
                f"Model `{self.CONFIG['model']}` is not a module of this job."
            )
            return False

        if (
            self.val_dataset is self.train_dataset and
            "fraction" not in self.CONFIG["val"]
        ):
            self.logger.error(
                "`val` uses the same data as `train`, so it requires a "
                "`fraction` to hold out."
            )
            return False

        if self.mode == "kfold" and self.CONFIG["kfold"]["k"] < 2:
            self.logger.error("Mode 'kfold' requires `k` to be at least 2.")
            return False

        return True

    def _build_component(self, spec: dict, namespace, *args):
        """
        Build a loss, optimiser or scheduler from its config. A spec
        with `module` refers to a module of the job (for custom
        components), a spec with `name` to a class in `namespace`.

        :param spec: Config of the component.
        :type spec: dict
        :param namespace: Where to find the class by its name,
            e.g. `torch.optim`.
        :param args: Positional arguments for the class.
        :return: The component.
        """
        if "module" in spec:
            return self.job.modules[spec["module"]]
        return getattr(namespace, spec["name"])(
            *args,
            **(spec.get("params") or {})
        )

    def _build_callbacks(self, config: TrainerConfig)-> list[Callback]:
        """
        Build the callbacks that are enabled in the config.

        :param config: Config of the engine, to monitor the same value
            as the engine does.
        :type config: TrainerConfig
        :return: The callbacks.
        :rtype: list[Callback]
        """
        callbacks = []

        early_stopping = self.CONFIG.get("early_stopping")
        if early_stopping:
            callbacks.append(EarlyStoppingCallback(
                EarlyStopper(mode=config.mode, **early_stopping),
                monitor=config.monitor
            ))

        checkpoint = self.CONFIG.get("checkpoint")
        if checkpoint:
            if checkpoint.get("resume"):
                self.logger.warning(
                    "Resuming from a checkpoint is not supported yet, "
                    "ignoring `checkpoint: resume`."
                )
            callbacks.append(CheckpointCallback(
                self.output_dir / "checkpoints",
                every_n_epochs=checkpoint.get("every", 5),
                save_best=checkpoint.get("save_best", True),
                plot_curves=checkpoint.get("plot_curves", True)
            ))

        return callbacks

    def _build_engine(self)-> Engine:
        """
        Build the training engine from the config.

        :return: The engine, ready to train.
        :rtype: Engine
        """
        model = self.job.modules[self.CONFIG["model"]]
        device = model.device
        model.to(device)

        loss_fn = self._build_component(self.CONFIG["loss"], nn).to(device)
        optimiser = self._build_component(
            self.CONFIG["optimiser"],
            torch.optim,
            model.parameters()
        )
        scheduler = None
        if self.CONFIG.get("scheduler"):
            scheduler = self._build_component(
                self.CONFIG["scheduler"],
                torch.optim.lr_scheduler,
                optimiser
            )

        monitor = self.CONFIG.get("monitor") or {}
        config = TrainerConfig(
            n_epochs=self.CONFIG["epochs"],
            metrics=self.CONFIG.get("metrics", []),
            monitor=monitor.get("value", "loss"),
            mode=monitor.get("goal", "min"),
            mixed_precision=self.CONFIG.get("mixed_precision", False)
        )
        return Engine(
            model=model,
            loss_fn=loss_fn,
            optimiser=optimiser,
            device=device,
            logger=self.logger,
            config=config,
            scheduler=scheduler,
            callbacks=self._build_callbacks(config)
        )

    def _log_training_result(self, result: TrainingResult)-> None:
        """
        Log the best score of every fold.

        :param result: Result of the training.
        :type result: TrainingResult
        """
        for i, fold in enumerate(result.folds):
            self.logger.info(
                f"Fold {i + 1}/{len(result.folds)}: best "
                f"{self.CONFIG.get('monitor', {}).get('value', 'loss')} of "
                f"{fold.best_score:.4f} at epoch {fold.best_epoch}"
                f"{' (stopped early)' if fold.stopped_early else ''}"
            )

    def run(self)-> None:
        """
        Train the model, evaluate it on the test data and save the
        results to the output folder of the job.
        """
        engine = self._build_engine()

        if self.mode == "kfold":
            result = engine.fit_cross_validation(
                self.train_dataset,
                self.CONFIG["kfold"]["k"],
                partial(self._to_dataloader, label="train")
            )
        else:
            result = engine.fit_split(
                self.train_dataloader,
                self.val_dataloader
            )
        self._log_training_result(result)
        torch.save(
            engine.model.state_dict(), 
            self.output_dir / "best_model.pt"
        )

        test_result = engine.evaluate(
            self.test_dataloader,
            visualisers=(self.CONFIG["test"] or {}).get("visualisers"),
            output_dir=self.output_dir / "evaluation"
        )
        self.logger.info(
            f"Test loss: {test_result.loss:.4f}, "
            f"metrics: {test_result.metrics}"
        )

        if (self.CONFIG.get("export") or {}).get("predictions"):
            torch.save(
                {
                    "predictions": test_result.predictions,
                    "targets": test_result.targets
                },
                self.output_dir / "predictions.pt"
            )

    def __str__(self)-> str:
        return str(self.CONFIG)
