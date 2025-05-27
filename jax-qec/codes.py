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


class RepetitionEncode(QuantumCode):
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
        Compute parity of qubit pairs.
        """
        return (0, 0)  # placeholder

    def decode(self, physical_state: jnp.ndarray) -> jnp.ndarray:
        """
        Decode the corrupted physical state by majority vote.
        Assumes input is a basis state with 2^n amplitudes and only one non-zero entry.
        Returns: logical state |0⟩ or |1⟩ as a 2-element vector.
        """
        # Step 1: Find which basis state this is (i.e., where the non-zero amplitude is)
        index = jnp.argmax(jnp.abs(physical_state))  # get index with highest amplitude

        # Step 2: Convert index to binary string to extract physical qubit values
        bits = jnp.array(list(jnp.binary_repr(index, width=self.n)), dtype=int)

        # Step 3: Count 1s
        num_ones = jnp.sum(bits)

        # Step 4: Majority vote
        if num_ones > self.n // 2:
            return jnp.array([0.0, 1.0])  # logical |1⟩
        else:
            return jnp.array([1.0, 0.0])  # logical |0⟩


