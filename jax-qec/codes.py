from abc import ABC, abstractmethod
from typing import Any
import jax.numpy as jnp
import jax


class QuantumCode(ABC):
    """Abstract base class for quantum error correcting codes."""

    def __init__(self, n_qubits: int, k_qubits: int):
        self.n = n_qubits  # total physical qubits
        self.k = k_qubits  # number of logical qubits

    @abstractmethod
    def encode(self, logical_state: jnp.ndarray) -> jnp.ndarray:
        pass

    @abstractmethod
    def measure(self, logical_state: jnp.ndarray) -> jnp.ndarray:
        pass

    @abstractmethod
    def decode_collapsed(self, physical_state: jnp.ndarray) -> jnp.ndarray:
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

    def measure(self, state: jnp.ndarray, key: jax.random.PRNGKey) -> jnp.ndarray:
        """
        Simulates quantum measurement using JAX-compatible randomness.
        Returns a collapsed state vector.
        """
        probs = jnp.abs(state) ** 2
        probs = probs / jnp.sum(probs)

        index = jax.random.choice(key, a=len(probs), p=probs)
        collapsed = jnp.zeros_like(state)
        return collapsed.at[index].set(1.0)

    def decode_collapsed(self, physical_state: jnp.ndarray) -> jnp.ndarray:
        """
        Decode a collapsed (basis) state by majority vote.
        """

        if jnp.count_nonzero((physical_state != 0) & (physical_state != 1)) >= 1:
            raise ValueError("decode_collapsed() expects a collapsed basis state. Use decode_superposition() for superpositions.")

        index = int(jnp.argmax(jnp.abs(physical_state)))
        bits = jnp.array(list(bin(index)[2:].zfill(self.n)), dtype=int)
        num_ones = int(jnp.sum(bits))
        if num_ones > self.n // 2:
            return jnp.array([0.0, 1.0])  # logical |1⟩
        else:
            return jnp.array([1.0, 0.0])  # logical |0⟩

    def decode_superposition(self, state: jnp.ndarray, key: jax.random.PRNGKey) -> jnp.ndarray:
        """
        Decodes a superposition state by first simulating measurement, then applying majority vote.
        """
        collapsed = self.measure(state, key)
        return self.decode_collapsed(collapsed)


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


