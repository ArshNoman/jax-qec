from jax import lax
import jax.numpy as jnp
import jax.random as random

from noise.base import NoiseModel


class BitFlipNoiseCollapsed(NoiseModel):
    def __init__(self, p: float):
        """
        Bit-flip noise model.

        Parameters:
        - p: Probability of flipping each qubit independently.
        """
        self.p = p

    def apply(self, state: jnp.ndarray, key) -> jnp.ndarray:
        """
        Apply bit-flip noise to each qubit independently.

        Parameters:
        - state: jnp.ndarray of shape (2**n,) or (batch_size, 2**n)
        - key: JAX PRNG key

        Returns:
        - Noisy state: same shape as input
        """
        # Detect if state is batched
        is_batched = state.ndim == 2
        n_qubits = int(jnp.log2(state.shape[-1]))

        def flip_each_qubit(state_i, key_i):
            for q in range(n_qubits):
                key_i, subkey = random.split(key_i)
                state_i = self._flip_qubit_with_prob(state_i, subkey, q)
            return state_i

        if is_batched:
            keys = random.split(key, state.shape[0])
            return jnp.stack([flip_each_qubit(s, k) for s, k in zip(state, keys)])
        else:
            return flip_each_qubit(state, key)

    def probability_flip(self, state, key, qubit_index):
        """
        Flips a qubit with probability p using a Pauli-X.

        Parameters:
        - state: jnp.ndarray of shape (2**n,)
        - key: PRNG key for randomness
        - qubit_index: index of the qubit to potentially flip

        Returns:
        - jnp.ndarray: modified state
        """
        should_flip = random.bernoulli(key, self.p)

        def apply_x(state_):
            bit = 1 << (state_.ndim * 0 + state_.shape[-1].bit_length() - qubit_index - 1)
            indices = jnp.arange(state_.shape[-1])
            flipped_indices = indices ^ bit
            return state_[flipped_indices]

        return lax.cond(should_flip, apply_x, lambda x: x, state)
