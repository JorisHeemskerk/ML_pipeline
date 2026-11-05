import logging
import numpy as np
import os
import random
import shutil
import torch
import traceback
import yaml

from jsonschema import validate, ValidationError

import modules as M

from config.config_validation_template import CONFIG_TEMPLATE
from create_logger import create_logger
from job import Job
from timestamp import START_TIMESTAMP


def seed_everything(seed: int, deterministic: bool=False)-> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        # Faster for fixed input sizes.
        torch.backends.cudnn.benchmark = True 
    

class JobFactory:
    def __init__(self, config_path: str)-> None:
        """
        Initialise parser object and required modules.

        Will read and validate config file with config template.
        Will initialise all general sub-objects.

        :param config_path: Path to config yaml file.
        """

        # Validate the provided config file.
        with open(config_path, 'r') as stream:
            CONFIG = yaml.safe_load(stream)
        try:
            validate(CONFIG, CONFIG_TEMPLATE)
        except ValidationError as e:
            location = " -> ".join(str(part) for part in e.absolute_path)
            raise ValidationError(
                "\x1b[31;1mA validation error occurred in the config file "
                f"(at `{location}`): {e.message}\x1b[0m"
            ) from e

        self.CONFIG = CONFIG

        # Initialise relevant general objects.
        self._init_general(config_path)

    def _init_general(self, config_path)-> None:
        # Initialise general logger.
        self.output_folder = self.CONFIG["general"]["output_folder"].replace(
            "{datetime}", 
            START_TIMESTAMP
        )
        os.makedirs(self.output_folder, exist_ok=True)
        shutil.copy(config_path, self.output_folder + "config.yml")

        self.logger = create_logger(
            name="general_logger", 
            output_log_file_name=f"{self.output_folder}general_logger.log"
        )
        self.logger.info(f"Logger initalised.")
        self.logger.info(f"Provided configuration: {self.CONFIG}")

        # Initialise Device.
        try:
            self.device = self.CONFIG["general"]["device"]
        except KeyError:
            self.logger.warning(
                "Device type not provided, attempting to load fastest."
                "HINT: set `device: 'value'` in config under 'general'."
            )
            self.device = torch.accelerator.current_accelerator().type if \
                torch.accelerator.is_available() else "cpu"
        self.logger.info(f"Using {self.device} device")

    
    def _init_module(
        self, 
        module_config: dict, 
        logger: logging.Logger
    )-> M.Base:
        module_name = next(iter(module_config))
        self.logger.debug(
            f"Initialising module `{module_name}` with kwargs: {module_config}"
        )

        module_config = module_config[module_name]
        module = getattr(M, module_name)(
            device=self.device, 
            logger=logger, 
            **module_config
        )
        
        return module

    def parse(self)-> None:
        """
        Main executing loop for parser.
        """
        self.logger.info("Starting initialising of jobs")
        self.jobs = []
        for job_config in self.CONFIG["jobs"]:
            new_job = Job(
                job_config["job"], 
                self.output_folder
            )
            self.logger.debug(f'Done initialising job {new_job.name}')

            self.logger.debug(f'Adding modules to job {new_job.name}')
            for module_config in job_config["job"]["modules"]:
                new_job.add_module(
                    self._init_module(module_config, new_job.logger), 
                )
            new_job.link_modules()

            new_job.set_trainer(job_config["job"]["trainer"])
            self.logger.debug(f"Validating job with trainer")
            valid = new_job.trainer.validate_components()
            if not valid:
                self.logger.error(
                    f"Job {new_job.name} has an invalid trainer and will be "
                    "skipped"
                )
            else:
                self.logger.debug("Job trainer validated successfully")
                self.logger.debug(f"Done with creating job {new_job.name}")
                self.jobs.append(new_job)

            
    
    def run(self)-> None:
        for job in self.jobs:
            seed_everything(self.CONFIG["general"]["seed"])
            try:
                job.run()
            except KeyboardInterrupt as e:
                self.logger.critical(
                    "PROGRAM MANUALLY HALTED BY KEYBOARD INTERRUPT "
                    "(inside job execution loop)."
                )
                raise KeyboardInterrupt(
                    "Keyboard interupt detected, halting program."
                ) from e
            except Exception as e:
                trace = ''.join(
                    traceback.format_exception(type(e), e, e.__traceback__)
                )
                self.logger.error(
                    f"Error during handling of job {job.name} ({job = })\n\t"
                    f"Traceback:\n\t{trace}\n\t'''{type(e)}: {e}'''\n"
                    "Skipping this job, attempting to execute next job."
                )
        
    def __str__(self)-> str:
            return str(self.CONFIG)
