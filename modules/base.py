import logging

class Base:
    def __init__(self, device: str, label: str, logger: logging.Logger) -> None:
        self.device = device
        self.label = label
        self.logger = logger
        self.input_type = None
        self.output_type = None
