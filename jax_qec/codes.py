from abc import ABC, abstractmethod
from typing import Any, List

from jax.experimental import sparse
import jax.numpy as jnp
from jax import random
import jax.lax as lax
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
    def measure(self, state: jnp.ndarray, key: jax.random.PRNGKey) -> jnp.ndarray:
        pass

    @abstractmethod
    def decode_collapsed(self, physical_state: jnp.ndarray) -> jnp.ndarray:
        pass

    @abstractmethod
    def decode_superposition(self, state: jnp.ndarray, key: jax.random.PRNGKey) -> jnp.ndarray:
        pass

    @abstractmethod
    def measure_syndrome_collapsed(self, physical_state: jnp.ndarray) -> jnp.ndarray:
        pass


class RepetitionEncode(QuantumCode):
    def __init__(self, n: int):
        assert n % 2 == 1, "number of physical qubits must be odd for majority voting to work"
        super().__init__(n_qubits=n, k_qubits=1)
        self.zero = jnp.array([1.0, 0.0])
        self.one = jnp.array([0.0, 1.0])

    def encode(self, logical_state: jnp.ndarray) -> jnp.ndarray:
        logical_zero = jnp.ones((2 ** self.n,))
        logical_zero = logical_zero.at[1:].set(0)

        logical_one = jnp.zeros((2 ** self.n,))
        logical_one = logical_one.at[-1].set(1)

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

        invalid = jnp.any((physical_state != 0) & (physical_state != 1) & (physical_state != -1))

        def proceed(_):
            index = jnp.argmax(jnp.abs(physical_state))
            bits = jnp.right_shift(index[None], jnp.arange(self.n)[::-1]) & 1
            num_ones = jnp.sum(bits)
            return jnp.where(num_ones > self.n // 2,
                             jnp.array([0.0, 1.0]),  # logical |1⟩
                             jnp.array([1.0, 0.0]))  # logical |0⟩

        def return_nan(_):
            return jnp.array([jnp.nan, jnp.nan])  # error signal

        return lax.cond(invalid, return_nan, proceed, operand=None)

    def decode_superposition(self, state: jnp.ndarray, key: jax.random.PRNGKey) -> jnp.ndarray:
        """
        Decodes a superposition state by first simulating measurement, then applying majority vote.
        """

        is_collapsed = jnp.count_nonzero(state == 1.0) == 1

        def return_nan(_):
            return jnp.array([jnp.nan, jnp.nan])  # signal: this is not a superposition

        def proceed(_):
            collapsed = self.measure(state, key)
            return self.decode_collapsed(collapsed)

        return jax.lax.cond(is_collapsed, return_nan, proceed, operand=None)

    def measure_syndrome_collapsed(self, state: jnp.ndarray) -> jnp.ndarray:
        """
        Measures syndrome bits for a collapsed (basis) state by comparing adjacent qubits.
        Assumes the input is a basis state (only one non-zero amplitude).
        Returns:
            JAX array of length n-1 where each bit indicates parity disagreement (1 = error detected)
        """

        index = jnp.argmax(jnp.abs(state))

        bits = jnp.right_shift(index, jnp.arange(self.n - 1, -1, -1)) & 1  # Convert index to bit representation (length n)

        syndrome = bits[:-1] ^ bits[1:]  # XOR adjacent bits to compute syndrome; shape (n-1,)

        return syndrome


class StabilizerCode(QuantumCode):
    def __init__(self, generators: List[jnp.ndarray]):
        self.generators = generators
        self.n = generators[0].shape[0] // 2
        super().__init__(n_qubits=self.n, k_qubits=self.n - len(generators))

    def encode(self, logical_state: jnp.ndarray) -> jnp.ndarray:
        """
        For now, encodes logical |0⟩ to the +1 eigenspace of the stabilizers.
        """
        dim = 2 ** self.n
        state = jnp.zeros((dim,))
        state = state.at[0].set(1.0)  # Start with |0...0⟩

        for g in self.generators:
            state = self.projector_apply(g, state)
        return state / jnp.linalg.norm(state)

    def projector_apply(self, generator, state):
        """
        Apply (I + S)/2 to the state, where S is the stabilizer operator from the symplectic form.
        For now, simulate as identity (since full Pauli matrix implementation is complex).
        """
        return state  # Placeholder — you may enhance this with full Pauli application later

    def measure(self, state: jnp.ndarray, key: jax.random.PRNGKey) -> jnp.ndarray:
        probs = jnp.abs(state) ** 2
        probs = probs / jnp.sum(probs)
        index = random.choice(key, a=len(probs), p=probs)
        collapsed = jnp.zeros_like(state)
        return collapsed.at[index].set(1.0)

    def measure_syndrome_collapsed(self, state: jnp.ndarray) -> jnp.ndarray:
        """
        Compute stabilizer syndrome by measuring each stabilizer generator.
        Each generator is represented as a 2n binary vector [X|Z].
        """
        index = jnp.argmax(jnp.abs(state))
        bits = jnp.right_shift(index, jnp.arange(self.n - 1, -1, -1)) & 1

        syndrome = []
        for g in self.generators:
            x = g[:self.n]
            z = g[self.n:]
            parity = jnp.sum((x + z) * bits) % 2
            syndrome.append(parity)
        return jnp.array(syndrome, dtype=jnp.uint8)

    def decode_collapsed(self, state: jnp.ndarray) -> jnp.ndarray:
        """
        Stub decoder: assumes correctable error and returns |0⟩
        """
        return jnp.array([1.0, 0.0])

    def decode_superposition(self, state: jnp.ndarray, key: jax.random.PRNGKey) -> jnp.ndarray:
        """
        Collapse then decode.
        """
        collapsed = self.measure(state, key)
        return self.decode_collapsed(collapsed)
