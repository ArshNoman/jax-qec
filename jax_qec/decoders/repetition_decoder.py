import jax
from jax import lax
import jax.numpy as jnp
from .base import Decoder

from utils import braket_to_state, state_to_braket


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

        corrected_state = jnp.zeros_like(state).at[corrected_index].set(1.0) # Create new one-hot vector

        return corrected_state

