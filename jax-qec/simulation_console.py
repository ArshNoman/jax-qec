from codes import RepetitionEncode, QuantumCode
from utils import state_to_braket
from noise import bit_flip
import jax.numpy as jnp
import random
import jax


def main():
    code = RepetitionEncode(3)

    logical_zero = jnp.array([1.0, 0.0])  # Logical -> |0⟩     [1 0 0 0 0 0 0 0] (encoded)
    logical_one = jnp.array([0.0, 1.0])  # Logical -> |1⟩     [0 0 0 0 0 0 0 1] (encoded)
    plus_state = jnp.array([1.0 / jnp.sqrt(2), 1.0 / jnp.sqrt(2)])  # |+⟩ state -> [1/√2, 1/√2]     [0.70710677 0 0 0 0 0 0 0.70710677]
    minus_state = jnp.array([1.0 / jnp.sqrt(2), -1.0 / jnp.sqrt(2)])  # |-⟩ state -> [1/√2, -1/√2]     [0.70710677 0 0 0 0 0 0 -0.70710677]

    current = logical_zero
    print("Current state:\n", current)

    encoded = code.encode(current)
    print("Encoded state:\n", encoded)

    encoded_flipped = bit_flip(encoded, 1)
    print("Flipped encoded state:\n", state_to_braket(encoded_flipped))

    # key = jax.random.PRNGKey(random.randint(1, 100))
    # key, subkey = jax.random.split(key)
    # measured = code.measure(current, subkey)
    # print("Measured state:\n", measured)

    # decoded = code.decode(encoded)
    # print("Decoded state:\n", decoded)


if __name__ == "__main__":
    main()
