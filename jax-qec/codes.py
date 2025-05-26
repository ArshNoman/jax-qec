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


class GeneralRepetitionCode(QuantumCode):
    def __init__(self, n: int):
        assert n % 2 == 1, "number of physical qubits must be odd for majority voting to work"
        super().__init__(n_qubits=n, k_qubits=1)
        self.zero = jnp.array([1.0, 0.0])
        self.one = jnp.array([0.0, 1.0])

    def encode(self, logical_state: jnp.ndarray) -> jnp.ndarray:
        logical_zero = self.zero
        logical_one = self.one
        for _ in range(self.n - 1):
            logical_zero = jnp.kron(logical_zero, self.zero)
            logical_one = jnp.kron(logical_one, self.one)
        return logical_state[0] * logical_zero + logical_state[1] * logical_one

    def measure_syndrome(self, physical_state: jnp.ndarray) -> tuple:
        """
        Placeholder: just return dummy syndrome.
        Later you can compute parity of qubit pairs.
        """
        return (0, 0)  # placeholder

    def decode(self, physical_state: jnp.ndarray) -> jnp.ndarray:
        """
        Placeholder: naive majority vote decoder.
        """
        # Assume state is collapsed to basis state for simplicity.
        # You'd typically convert amplitude state to measurement outcome here.
        return jnp.array([1.0, 0.0])  # dummy decode to |0⟩

