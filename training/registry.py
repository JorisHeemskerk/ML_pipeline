"""
Generic name-to-class registry, used to select components (metrics,
visualisers, ...) with plain strings, so they can be set from a config
file.
"""

from typing import Any, Callable


class Registry:
    """
    Maps string names to classes.

    Classes are added with the `register` decorator and are later built
    from a spec: either just the registered name, or a tuple of the
    registered name and a dict of keyword arguments for the constructor.
    A `"name"` entry in that dict does not reach the constructor, it
    renames the resulting object (useful when the same class is used
    twice with different arguments).

    :param kind: Human readable description of what is stored, used in
        error messages.
    :type kind: str
    """

    def __init__(self, kind: str):
        self.kind = kind
        self._entries: dict[str, type] = {}

    def register(self, name: str) -> Callable[[type], type]:
        """
        Class decorator that registers a class under `name`.

        :param name: Name to register the class under.
        :type name: str
        :return: Decorator that registers and returns the class.
        :rtype: Callable[[type], type]
        """
        def decorator(cls: type) -> type:
            if name in self._entries:
                raise KeyError(f"{self.kind} '{name}' is already registered.")
            self._entries[name] = cls
            return cls
        return decorator

    def names(self) -> list[str]:
        """
        :return: All registered names, sorted alphabetically.
        :rtype: list[str]
        """
        return sorted(self._entries)

    def build(
        self, 
        spec: str | tuple[str, dict[str, Any]]
    ) -> tuple[str, Any]:
        """
        Build one registered object from a spec.

        :param spec: Registered name, or (registered name, kwargs).
        :type spec: str | tuple[str, dict[str, Any]]
        :return: The key to store the object under and the object.
        :rtype: tuple[str, Any]
        """
        if isinstance(spec, str):
            registered_name, kwargs = spec, {}
        else:
            registered_name, kwargs = spec
        kwargs = dict(kwargs)
        key = kwargs.pop("name", registered_name)

        if registered_name not in self._entries:
            raise KeyError(
                f"Unknown {self.kind} '{registered_name}'. "
                f"Available: {', '.join(self.names())}."
            )
        return key, self._entries[registered_name](**kwargs)

    def build_all(
        self, 
        specs: list[str | tuple[str, dict[str, Any]]]
    ) -> dict[str, Any]:
        """
        Build multiple registered objects from a list of specs.

        :param specs: List of specs, see `build`.
        :type specs: list[str | tuple[str, dict[str, Any]]]
        :return: Dict mapping keys to the built objects.
        :rtype: dict[str, Any]
        """
        built = {}
        for spec in specs:
            key, obj = self.build(spec)
            if key in built:
                raise ValueError(
                    f"Duplicate {self.kind} key '{key}'. Use the "
                    f"\"name\" kwarg to tell them apart."
                )
            built[key] = obj
        return built
