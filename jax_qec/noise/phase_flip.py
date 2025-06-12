from jax import lax
import jax.numpy as jnp
import jax.random as random

from noise.base import NoiseModel


class PhaseFlipNoise(NoiseModel):
    def __init__(self, p: float):
        """
        Phase-flip (Z gate) noise model for collapsed states.

        Parameters:
        - p: Probability of flipping the phase of a qubit
        """
        self.p = p

    def apply(self, state: jnp.ndarray, key) -> jnp.ndarray:
        """
        Apply phase-flip noise independently to each qubit.

        Parameters:
        - state: 1D or 2D jnp.ndarray (collapsed basis states only)
        - key: PRNG key

        Returns:
        - Modified state with phase flips applied
        """
        is_batched = state.ndim == 2
        n_qubits = int(jnp.log2(state.shape[-1]))

        def flip_each_qubit(state_i, key_i):
            for q in range(n_qubits):
                key_i, subkey = random.split(key_i)
                state_i = self.flip_qubit(state_i, subkey, q, n_qubits)
            return state_i

        if is_batched:
            keys = jax.random.split(key, state.shape[0])
            return jnp.stack([flip_each_qubit(s, k) for s, k in zip(state, keys)])
        else:
            return flip_each_qubit(state, key)

    def flip_qubit(self, state: jnp.ndarray, key, qubit_index: int, n_qubits: int) -> jnp.ndarray:
        """
        Apply a phase-flip (Z gate) to a specific qubit with probability p.

        Parameters:
        - state: jnp.ndarray of shape (2^n,)
        - key: JAX PRNG key
        - qubit_index: which qubit to flip

        Returns:
        - jnp.ndarray: modified state
        """
        should_flip = random.bernoulli(key, self.p)

        def apply_z(state_):
            indices = jnp.arange(state_.shape[0])
            mask = (indices >> (n_qubits - 1 - qubit_index)) & 1
            phase = 1 - 2 * mask  # 1 if 0, -1 if 1
            return state_ * phase

        return lax.cond(should_flip, apply_z, lambda x: x, state)

    def probability_flip(self, state: jnp.ndarray, key, qubit_index: int) -> jnp.ndarray:
        """
        Public method to apply a probabilistic bit-flip to a specific qubit.

        Parameters:
        - state: jnp.ndarray (collapsed or superposition)
        - key: PRNG key
        - qubit_index: index of the qubit to flip

        Returns:
        - jnp.ndarray: new state
        """
        n_qubits = int(jnp.log2(state.shape[-1]))
        return self.flip_qubit(state, key, qubit_index, n_qubits)
