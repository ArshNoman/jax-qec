from abc import ABC, abstractmethod
from typing import Any
import jax.numpy as jnp


class QuantumCode(ABC):
    """Abstract base class for quantum error correcting codes."""

    def __init__(self, n_qubits: int, k_qubits: int):
        self.n = n_qubits  # total physical qubits
        self.k = k_qubits  # number of logical qubits

    @abstractmethod
    def encode(self, logical_state: jnp.ndarray) -> jnp.ndarray:
        pass

    @abstractmethod
    def decode(self, physical_state: jnp.ndarray) -> jnp.ndarray:
        pass

    @abstractmethod
    def measure_syndrome(self, physical_state: jnp.ndarray) -> Any:
        pass



