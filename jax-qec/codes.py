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

    def measure_syndrome(self, physical_state: jnp.ndarray) -> jnp.ndarray:
        """
        Measures the syndrome by checking parity between adjacent qubits.
        Only works for collapsed basis states (i.e., single non-zero index).
        Returns: JAX array of length (n - 1) containing 0 (agree) or 1 (disagree)
        """
        # Step 1: Find which index is active (only works for basis states)
        index = int(jnp.argmax(jnp.abs(physical_state)))

        # Step 2: Convert to binary string of length n
        bits = jnp.array(list(bin(index)[2:].zfill(self.n)), dtype=int)

        # Step 3: Compute pairwise parity (XOR) between adjacent qubits
        syndrome = jnp.array([bits[i] ^ bits[i + 1] for i in range(self.n - 1)])

        return syndrome

    def decode(self, physical_state: jnp.ndarray) -> jnp.ndarray:
        """
        Decode a collapsed (basis) state by majority vote.
        Assumes input state has a single non-zero entry.
        """
        index = int(jnp.argmax(jnp.abs(physical_state)))
        bits = jnp.array(list(bin(index)[2:].zfill(self.n)), dtype=int)
        num_ones = int(jnp.sum(bits))
        if num_ones > self.n // 2:
            return jnp.array([0.0, 1.0])  # logical |1⟩
        else:
            return jnp.array([1.0, 0.0])  # logical |0⟩

    def simulate_measurement(self, state: jnp.ndarray) -> jnp.ndarray:
        """
        Simulates a quantum measurement of a state.
        Collapses the state probabilistically into a basis state.
        """
        probs = jnp.abs(state) ** 2
        probs = probs / jnp.sum(probs)
        index = int(np.random.choice(len(state), p=np.array(probs)))
        collapsed = jnp.zeros_like(state)
        return collapsed.at[index].set(1.0)


