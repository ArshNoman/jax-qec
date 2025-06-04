import jax.numpy as jnp
from .base import Decoder


class RepetitionXDecoder(Decoder):
    """
    Syndrome-based decoder for the repetition code.
    Assumes the code provides syndrome measurement rules.
    """

    def decode(self, current: jnp.ndarray, syndrome: jnp.ndarray) -> jnp.ndarray:
        """
        Applies correction based on the syndrome by flipping the minority bit.

        Parameters:
        - syndrome: jnp.ndarray of shape (n-1) for n physical qubits.
                    Each syndrome bit is the parity between two neighboring qubits.

        Returns: jnp.ndarray of shape (n) with the corrected state
        """

        corrected_state = current.copy()

        for i in range(len(syndrome)):
            if syndrome[i] == 1:
                corrected_state = corrected_state.at[i + 1].set(corrected_state[i])

        return corrected_state

