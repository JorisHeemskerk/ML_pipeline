import os

import modules as M

from create_logger import create_logger


class Job:
    def __init__(self, job_config, logger_dir) -> None:
        self.CONFIG = job_config

        # Create output folder.
        self.output_folder = logger_dir + f"job_{self.CONFIG['name']}/"
        try:
            os.mkdir(self.output_folder)
        # In case of duplicate jobs, increase suffix.
        except FileExistsError:
            # In case of duplicate, keep adding suffix until available.
            i = 1
            done = False
            while not done:
                try:
                    attempt = self.output_folder[:-1] + f"_{i}/"
                    os.mkdir(attempt)
                    done = True
                    self.output_folder = attempt
                except FileExistsError:
                    i += 1

        # Get name from newly suffixed folder.
        self.name = self.output_folder[:-1].split('/')[-1][4:]

        # Initialise job-specific logger.
        self.logger = create_logger(
            name=self.name, 
            output_log_file_name=f"{self.output_folder}{self.name}.log"
        )
        self.logger.info(f"{self.name} logger initalised.")
        self.logger.debug(f"Provided configuration: {self.CONFIG}")

        self.modules: dict[str, M.Base] = {}

    def add_module(self, module: M.Base)-> None:
        if module.label in self.modules:
            self.logger.error(
                f"Module `{module.label}` already exists, entry skipped."
            )
        else:
            self.modules[module.label] = module

    def set_trainer(self, trainer_config: dict):
        from trainer import Trainer
        self.trainer = Trainer(self, trainer_config)

    def run(self)-> None:
        self.logger.info(f"Running job {self.name}")
        self.trainer.run()

    def __str__(self)-> str:
        return str(self.CONFIG)
