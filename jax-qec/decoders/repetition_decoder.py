import jax.numpy as jnp
from .base import Decoder


class RepetitionXDecoder(Decoder):
    """
    Syndrome-based decoder for the repetition code.
    Assumes the code provides syndrome measurement rules.
    """

    def decode(self, syndrome: jnp.ndarray) -> jnp.ndarray:
        """
        Applies correction based on the syndrome by flipping the minority bit.

        Parameters:
        - syndrome: jnp.ndarray of shape (n-1,) for n physical qubits.
                    Each syndrome bit is the parity between two neighboring qubits.

        Returns:
        - correction: jnp.ndarray of shape (n,) with 0 (no flip) or 1 (flip)
        """
        n = self.code.num_qubits
        correction = jnp.zeros(n, dtype=jnp.int32)

        # Naive majority vote approach (bit-flip error model)
        # Flip any qubit whose neighbors disagree with it

        # Count the number of 1s in the syndrome
        total_flips = jnp.sum(syndrome)

        # If more than half of syndrome bits indicate disagreement,
        # assume the minority bit is wrong. Flip the first qubit.
        # This works for small repetition codes (3–7 qubits).
        if total_flips > (n - 1) // 2:
            correction = correction.at[0].set(1)  # flip first qubit

        return correction
