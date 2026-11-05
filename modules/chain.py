import logging
import torch
from torch import nn

from base import Base
from base_model import BaseModel


class Chain(BaseModel):
    """
    Runs other modules of the job one after the other, the output of
    each module is the input of the next one. This allows a model to be
    built from separate modules, for example an embedding model followed
    by an autoencoder.

    The stages are referred to by their label, and are linked once all
    modules of the job exist (see `Base.link`). The parameters of all
    stages are parameters of the chain, so the chain can be given to the
    trainer as the model. Frozen stages (parameters without gradients)
    are not updated.
    """

    def __init__(
        self,
        device: str,
        label: str,
        logger: logging.Logger,
        stages: list[str],
        *args,
        **kwargs
    ) -> None:
        """
        :param stages: Labels of the modules to run, in order.
        :type stages: list[str]
        """
        super().__init__(device, label, logger, *args, **kwargs)
        if len(stages) == 0 or len(set(stages)) != len(stages):
            raise ValueError(
                f"Chain `{label}` needs at least 1 stage, and every stage "
                f"may only be used once, but got: {stages}."
            )
        self.stage_labels = list(stages)
        self.stages = nn.ModuleDict()

    def link(self, modules: dict[str, Base])-> None:
        """
        Find the stages among the modules of the job, and check that the
        output type of every stage matches the input type of the next.

        :param modules: All modules of the job, by label.
        :type modules: dict[str, Base]
        """
        for stage_label in self.stage_labels:
            if stage_label not in modules:
                raise KeyError(
                    f"Chain `{self.label}` refers to stage `{stage_label}`"
                    f", which is not a module of this job."
                )
            stage = modules[stage_label]
            if stage is self or not (
                isinstance(stage, Base) and isinstance(stage, nn.Module)
            ):
                raise TypeError(
                    f"Stage `{stage_label}` of chain `{self.label}` is not"
                    " a torch model module of this job."
                )
            self.stages[stage_label] = stage

        stages = list(self.stages.values())
        for previous, following in zip(stages, stages[1:]):
            if (
                previous.output_type is not None and
                following.input_type is not None and
                previous.output_type != following.input_type
            ):
                raise ValueError(
                    f"Mismatch in chain `{self.label}`: the output of "
                    f"`{previous.label}` ({previous.output_type}) does "
                    f"not match the input of `{following.label}` "
                    f"({following.input_type})."
                )

        self.input_type = stages[0].input_type
        self.output_type = stages[-1].output_type
        self.logger.debug(f"Linked {self}")

    def forward(
        self,
        x: torch.Tensor
    )-> torch.Tensor | dict[str, torch.Tensor]:
        """
        Run the input through all the stages.

        :param x: Input of the first stage.
        :type x: torch.Tensor
        :return: Output of the last stage.
        :rtype: torch.Tensor | dict[str, torch.Tensor]
        """
        for stage in self.stages.values():
            x = stage(x)
        return x
