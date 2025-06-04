from abc import ABC, abstractmethod
import jax.numpy as jnp


class Decoder(ABC):
    """
    Abstract base class for syndrome-based decoders.
    All decoders should implement the `decode` method.
    """

    def __init__(self,  code: jnp.ndarray):
        """
        Initialize with a reference to the QEC code instance.
        This allows access to code-specific structures if needed.

        Parameters:
        - code: an instance of a QEC code class (e.g., RepetitionEncode)
        """
        self.code = code

    @abstractmethod
    def decode(self, current: jnp.ndarray, syndrome: jnp.ndarray) -> jnp.ndarray:
        """
        Given a measured syndrome, return a correction operator
        as a binary vector (same length as the number of physical qubits).

        Parameters:
        - syndrome: jnp.ndarray of shape (num_syndrome_bits,)

        Returns:
        - correction: jnp.ndarray of shape (num_physical_qubits,) where each entry is 0 (no flip) or 1 (apply bit-flip)
        """
        pass
