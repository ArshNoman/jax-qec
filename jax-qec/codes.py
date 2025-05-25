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


class RepetitionCode(QuantumCode):
    """3-qubit bit-flip repetition code."""

    def __init__(self):
        super().__init__(n_qubits=3, k_qubits=1)

    def encode(self, logical_state: jnp.ndarray) -> jnp.ndarray:
        """
        logical_state: shape (2,) representing |ψ⟩ = a|0⟩ + b|1⟩
        returns: shape (8,) representing 3-qubit encoded state in 8D Hilbert space
        """
        zero = jnp.array([1.0, 0.0])
        one = jnp.array([0.0, 1.0])

        logical_zero = jnp.kron(jnp.kron(zero, zero), zero)
        logical_one = jnp.kron(jnp.kron(one, one), one)

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

