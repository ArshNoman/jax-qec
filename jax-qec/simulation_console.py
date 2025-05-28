from codes import RepetitionEncode, QuantumCode
import jax.numpy as jnp
import random
import jax


# TEST COMMIT


def flip_qubit(state: jnp.ndarray, qubit_idx: int) -> jnp.ndarray:
    """
    Flips a single qubit (bit-flip error) by swapping amplitudes.
    Assumes input is a pure basis state (only one non-zero index).
    """
    n = int(jnp.log2(len(state)))
    index = int(jnp.argmax(jnp.abs(state)))
    bits = list(bin(index)[2:].zfill(n))

    # Flip the selected qubit
    bits[qubit_idx] = '1' if bits[qubit_idx] == '0' else '0'
    flipped_index = int("".join(bits), 2)

    # Create new state with flipped qubit
    new_state = jnp.zeros_like(state)
    new_state = new_state.at[flipped_index].set(1.0)
    return new_state


def main():
    code = RepetitionEncode(3)

    logical_zero = jnp.array([1.0, 0.0])  # Logical -> |0⟩     [1 0 0 0 0 0 0 0]
    logical_one = jnp.array([0.0, 1.0])  # Logical -> |1⟩     [0 0 0 0 0 0 0 1]
    plus_state = jnp.array([1.0 / jnp.sqrt(2), 1.0 / jnp.sqrt(2)])  # |+⟩ state -> [1/√2, 1/√2]     [0.70710677 0 0 0 0 0 0 0.70710677]
    minus_state = jnp.array([1.0 / jnp.sqrt(2), -1.0 / jnp.sqrt(2)])  # |-⟩ state -> [1/√2, -1/√2]     [0.70710677 0 0 0 0 0 0 -0.70710677]

    current = logical_zero
    print("Current state:\n", current)

    # encoded = code.encode(current)
    # print("Encoded state:\n", encoded)

    key = jax.random.PRNGKey(random.randint(1, 100))
    key, subkey = jax.random.split(key)
    measured = code.measure(current, subkey)
    print("Measured state:\n", measured)

    # decoded = code.decode(encoded)
    # print("Decoded state:\n", decoded)


if __name__ == "__main__":
    main()
