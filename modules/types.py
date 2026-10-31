from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class TensorSpecific:
    dtype: torch.dtype
    shape: tuple[int, ...]
    
    @classmethod
    def from_tensor(cls, x: torch.Tensor)-> "TensorType":
        return cls(
            dtype=x.dtype,
            shape=tuple(x.shape),
        )

@dataclass(frozen=True)
class TensorGeneric:
    dtype: torch.dtype

    @classmethod
    def from_tensor(cls, x: torch.Tensor)-> "TensorType":
        return cls(dtype=x.dtype)
