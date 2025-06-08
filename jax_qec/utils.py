import jax.numpy as jnp
import jax


logical_zero = jnp.array([1.0, 0.0])  # Logical -> |0⟩     [1 0 0 0 0 0 0 0] (encoded)
logical_one = jnp.array([0.0, 1.0])  # Logical -> |1⟩     [0 0 0 0 0 0 0 1] (encoded)
plus_state = jnp.array([1.0 / jnp.sqrt(2), 1.0 / jnp.sqrt(2)])  # |+⟩ state -> [1/√2, 1/√2]     [0.70710677 0 0 0 0 0 0 0.70710677] (encoded)
minus_state = jnp.array([1.0 / jnp.sqrt(2), -1.0 / jnp.sqrt(2)])  # |-⟩ state -> [1/√2, -1/√2]     [0.70710677 0 0 0 0 0 0 -0.70710677] (encoded)


def state_to_braket(state: jnp.ndarray) -> str:
    """
    Converts a basis state vector into bra-ket notation.
    Assumes the input is a collapsed state.
    """
    if jnp.count_nonzero((state == 1)) == 0:
        raise ValueError('state_to_braket only accepts collapsed states as inputs')
    index = int(jnp.argmax(jnp.abs(state)))  # Find the index with amplitude 1
    n = int(jnp.log2(state.shape[0]))        # Number of qubits
    binary_str = bin(index)[2:].zfill(n)     # Convert to binary and pad
    return f"|{binary_str}⟩"


def braket_to_state(braket: str) -> jnp.ndarray:
    """
    Converts a bra-ket string like "|101⟩" into a basis state vector.

    Returns:
    - jnp.ndarray: A 1D array with a 1.0 at the index corresponding to the binary string.
    """

    if braket.startswith('|') and braket.endswith('⟩'):
        braket = braket[1:-1]  # Strip the '|' and '⟩'

    if not all(c in '01' for c in braket):
        raise ValueError("Bra-ket string must contain only binary digits between the symbols")

    index = int(braket, 2)
    size = 2 ** len(braket)

    state = jnp.zeros(size, dtype=jnp.float32)
    state = state.at[index].set(1.0)
    return state


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