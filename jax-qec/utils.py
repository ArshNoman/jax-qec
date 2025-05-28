import jax.numpy as jnp
import jax

def state_to_braket(state: jnp.ndarray) -> str:
    """
    Converts a basis state vector into bra-ket notation.
    Assumes the input is a pure state (only one non-zero amplitude).
    """
    index = int(jnp.argmax(jnp.abs(state)))  # Find the index with amplitude 1
    n = int(jnp.log2(state.shape[0]))        # Number of qubits
    binary_str = bin(index)[2:].zfill(n)     # Convert to binary and pad
    return f"|{binary_str}⟩"