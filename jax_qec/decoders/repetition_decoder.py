from utils import braket_to_state, state_to_braket
from .base import Decoder

import jax.numpy as jnp
from jax import lax
import jax


class RepetitionXDecoder(Decoder):
    """
    Syndrome-based decoder for the repetition code.
    Assumes the code provides syndrome measurement rules.
    """

    def decode(self, state: jnp.ndarray, syndrome: jnp.ndarray) -> jnp.ndarray:
        """
        Applies correction based on the syndrome by flipping the minority bit.

        Parameters:
        - syndrome: jnp.ndarray of shape (n-1) for n physical qubits.
                    Each syndrome bit is the parity between two neighboring qubits.

        Returns: jnp.ndarray of shape (n) with the corrected state
        """

        num_amplitudes = state.shape[0]
        n = int(jnp.log2(num_amplitudes))

        # Get the index of the '1.0' in the state (collapsed)
        index = jnp.argmax(state)

        bits = jnp.array([(index >> i) & 1 for i in range(n)][::-1], dtype=jnp.int32)  # Convert index to binary array

        def body_fn(i, b):
            s_i = syndrome[i]
            s_ip1 = syndrome[i + 1]

            def case1(b):
                return b.at[i + 1].set(b[i])

            def case2(b):
                return b.at[i].set(b[i + 1])

            b = lax.cond(jnp.logical_and(s_i == 1, s_ip1 == 1), case1, lambda b: b, b)
            b = lax.cond(jnp.logical_and(s_i == 1, s_ip1 == 0), case2, lambda b: b, b)
            return b

        corrected_bits = lax.fori_loop(0, n - 1, body_fn, bits)

        # Convert corrected bits back to index
        powers = 2 ** jnp.arange(n - 1, -1, -1)
        corrected_index = jnp.sum(corrected_bits * powers)

        corrected_state = jnp.zeros_like(state).at[corrected_index].set(1.0)  # Create new one-hot vector

        return corrected_state


class RepetitionZDecoder(Decoder):
    """
    Decoder for repetition code that corrects phase (Z) errors.
    Assumes the code provides syndrome measurement rules using X-type stabilizers.
    """

    def decode(self, state: jnp.ndarray, syndrome: jnp.ndarray) -> jnp.ndarray:
        """
        Applies Z gate to qubit(s) based on syndrome. This corrects phase flips.

        Parameters:
        - state: jnp.ndarray, 2^n-dimensional collapsed state
        - syndrome: jnp.ndarray of shape (n-1), X-type parity checks

        Returns: corrected state (jnp.ndarray)
        """
        num_amplitudes = state.shape[0]
        n = int(jnp.log2(num_amplitudes))

        index = jnp.argmax(jnp.abs(state))
        phase = jnp.angle(state[index])  # Track if any existing phase

        # Convert index to bits
        bits = jnp.array([(index >> i) & 1 for i in range(n)][::-1], dtype=jnp.int32)

        # Determine which qubit likely has the Z error
        error_qubit = -1
        if jnp.all(syndrome == jnp.array([1, 1])):
            error_qubit = 1
        elif jnp.all(syndrome == jnp.array([1, 0])):
            error_qubit = 0
        elif jnp.all(syndrome == jnp.array([0, 1])):
            error_qubit = 2

        # Apply Z gate to that qubit (phase flip)
        def apply_z(index, qubit, n):
            mask = 1 << (n - qubit - 1)
            bit = (index & mask) >> (n - qubit - 1)
            return -1.0 if bit == 1 else 1.0

        phase_factor = lax.cond(
            error_qubit != -1,
            lambda _: apply_z(index, error_qubit, n),
            lambda _: 1.0,
            operand=None
        )

        corrected_state = jnp.zeros_like(state).at[index].set(phase_factor)

        return corrected_state

