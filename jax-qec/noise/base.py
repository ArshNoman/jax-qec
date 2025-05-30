from abc import ABC, abstractmethod
import jax.numpy as jnp


class NoiseModel(ABC):
    @abstractmethod
    def apply(self, state: jnp.ndarray, key) -> jnp.ndarray:
        """
        Apply the noise model to the quantum state.

        Parameters:
        - state: jnp.ndarray of shape (2**n,) or (batch_size, 2**n)
        - key: JAX PRNG key

        Returns:
        - Noisy state: jnp.ndarray of the same shape
        """

        pass

    # def __call__(self, state: jnp.ndarray, key) -> jnp.ndarray:
    #     return self.apply(state, key)
