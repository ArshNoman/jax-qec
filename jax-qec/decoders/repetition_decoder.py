import jax
import jax.numpy as jnp
from .base import Decoder

from utils import braket_to_state, state_to_braket


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

        corrected_state = state_to_braket(current.copy())
        corrected_state_list = list(corrected_state[1:-1])  # Strip the '|' and '⟩'

        for i in range(len(syndrome)):
            if syndrome[i] == 1 and syndrome[i+1] == 1:
                corrected_state_list[i+1] = corrected_state_list[i]
            elif syndrome[i] == 1 and syndrome[i+1] == 0:
                corrected_state_list[i] = corrected_state_list[i+1]

        corrected_state = ''
        for i in corrected_state_list:
            corrected_state += i
        return braket_to_state(corrected_state)

