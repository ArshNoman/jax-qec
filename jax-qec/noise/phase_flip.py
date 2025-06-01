from jax import lax
import jax.numpy as jnp
import jax.random as random

from noise.base import NoiseModel


class PhaseFlipNoiseCollapsed(NoiseModel):
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
                state_i = self._flip_phase_if_one(state_i, subkey, q)
            return state_i

        if is_batched:
            keys = random.split(key, state.shape[0])
            return jnp.stack([flip_each_qubit(s, k) for s, k in zip(state, keys)])
        else:
            return flip_each_qubit(state, key)

    def _flip_phase_if_one(self, state, key, qubit_index):
        """
        Flip the phase (multiply by -1) if the target qubit is 1, with probability p.

        Parameters:
        - state: jnp.ndarray of shape (2^n,)
        - key: PRNG key
        - qubit_index: which qubit to check

        Returns:
        - jnp.ndarray with phase flipped if condition met
        """
        should_flip = random.bernoulli(key, self.p)

        def apply_phase_flip(state_):
            index = jnp.argmax(state_)  # Only one non-zero amplitude
            bit_value = (index >> (state_.shape[-1].bit_length() - qubit_index - 1)) & 1
            return lax.cond(bit_value == 1,
                            lambda s: s.at[index].set(-s[index]),
                            lambda s: s,
                            state_)

        return lax.cond(should_flip, apply_phase_flip, lambda x: x, state)
