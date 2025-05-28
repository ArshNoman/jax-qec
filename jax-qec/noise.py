import jax.numpy as jnp


def bit_flip(state: jnp.ndarray, qubit_index: int) -> jnp.ndarray:
    """
    Applies a Pauli-X (bit-flip) error to a specific qubit in a multi-qubit basis state.
    Assumes the input state is a pure basis state with one non-zero amplitude.
    """
    n = int(jnp.log2(state.shape[0]))  # number of physical qubits
    index = int(jnp.argmax(jnp.abs(state)))  # index of the largest amplitude
    bits = list(bin(index)[2:].zfill(n))

    # flipping
    bits[qubit_index] = '1' if bits[qubit_index] == '0' else '0'
    flipped_index = int("".join(bits), 2)

    new_state = jnp.zeros_like(state)
    return new_state.at[flipped_index].set(1.0)
