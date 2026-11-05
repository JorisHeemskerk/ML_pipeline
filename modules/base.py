import logging

class Base:
    def __init__(self, device: str, label: str, logger: logging.Logger) -> None:
        self.device = device
        self.label = label
        self.logger = logger
        self.input_type = None
        self.output_type = None

    def link(self, modules: dict[str, "Base"])-> None:
        """
        Called once by the job, after all of its modules are created.
        Does nothing by default, override to refer to other modules of 
        the job by their label (e.g. to chain them together).

        :param modules: All modules of the job, by label.
        :type modules: dict[str, Base]
        """
