"""
Early stopper, to be used with the `EarlyStoppingCallback`.
"""


class EarlyStopper:
    """
    Decides to stop when a monitored value has not improved for
    `patience` consecutive checks.

    :param patience: Number of checks without improvement to wait before
        stopping. (DEFAULT=5)
    :type patience: int
    :param min_delta: Minimal change that counts as an improvement.
        (DEFAULT=0.0)
    :type min_delta: float
    :param mode: "min" if a lower value is better, "max" if a higher one
        is. (DEFAULT="min")
    :type mode: str
    """

    def __init__(
        self,
        patience: int=5,
        min_delta: float=0.0,
        mode: str="min"
    ):
        if mode not in ("min", "max"):
            raise ValueError(f"Mode must be 'min' or 'max', not {mode}.")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.reset()

    def reset(self) -> None:
        """
        Forget everything seen so far, call this at the start of a fold.
        """
        self._best: float | None = None
        self._n_without_improvement = 0

    def should_stop(self, value: float) -> bool:
        """
        Register a new value and decide if training should stop.

        :param value: The monitored value of the latest epoch.
        :type value: float
        :return: True if there was no improvement for `patience` checks.
        :rtype: bool
        """
        if self._is_improvement(value):
            self._best = value
            self._n_without_improvement = 0
        else:
            self._n_without_improvement += 1
        return self._n_without_improvement >= self.patience

    def _is_improvement(self, value: float) -> bool:
        """
        :param value: Value to compare to the best value so far.
        :type value: float
        :return: True if `value` is better than the best by `min_delta`.
        :rtype: bool
        """
        if self._best is None:
            return True
        if self.mode == "min":
            return value < self._best - self.min_delta
        return value > self._best + self.min_delta
