from ./noise/bi
from codes import RepetitionEncode, QuantumCode
from utils import state_to_braket
from noise import bit_flip
import jax.numpy as jnp
import random
import jax

logical_zero = jnp.array([1.0, 0.0])  # Logical -> |0⟩     [1 0 0 0 0 0 0 0] (encoded)
logical_one = jnp.array([0.0, 1.0])  # Logical -> |1⟩     [0 0 0 0 0 0 0 1] (encoded)
plus_state = jnp.array([1.0 / jnp.sqrt(2), 1.0 / jnp.sqrt(2)])  # |+⟩ state -> [1/√2, 1/√2]     [0.70710677 0 0 0 0 0 0 0.70710677]
minus_state = jnp.array([1.0 / jnp.sqrt(2), -1.0 / jnp.sqrt(2)])  # |-⟩ state -> [1/√2, -1/√2]     [0.70710677 0 0 0 0 0 0 -0.70710677]


def codes_test():
    code = RepetitionEncode(5)

    current = plus_state
    print("Encoded state:", current)
    # print("Current state:", current, "->", state_to_braket(current))

    current = code.encode(current)
    print("Encoded state:\n", current)
    # print("Encoded state:\n", current, "->", state_to_braket(current))

    # current = bit_flip(current, 0)
    # print("Flipped encoded state:", state_to_braket(current))
    # current = bit_flip(current, 1)
    # print("Flipped encoded state:", state_to_braket(current))
    # current = bit_flip(current, 2)
    # print("Flipped encoded state:", state_to_braket(current))

    # syndrome = code.measure_syndrome_collapsed(current)
    # print("Syndrome:", syndrome)

    # key = jax.random.PRNGKey(random.randint(1, 100))
    # key, subkey = jax.random.split(key)
    # current = code.measure(current, subkey)
    # print("Measured state:\n", current, "->", state_to_braket(current))

    # current = code.decode_collapsed(current)
    # print("Decoded state:", current, "->", state_to_braket(current))

    key = jax.random.PRNGKey(random.randint(1, 100))
    key, subkey = jax.random.split(key)
    current = code.decode_superposition(current, subkey)
    print("Decoded state:\n", current, "->", state_to_braket(current))


def noise_test():

    # JAX PRNG setup
    key = jax.random.PRNGKey(42)

    # 1. Define logical state (|1⟩ in this case)
    current = logical_one  # |1⟩
    print("Logical State:", state_to_braket(current))

    # 2. Encode using 3-qubit repetition code
    code = RepetitionEncode(3)
    current = code.encode(current)
    print("Encoded State:", state_to_braket(current))

    # 3. Apply bit-flip noise (collapsed version, simulate error on one qubit)
    # Force collapse of the encoded state to simulate post-measurement scenario
    # For testing purposes, let's pretend a specific collapsed state occurred
    # collapsed_state = jnp.array([0., 0., 0., 0., 0., 0., 0., 1.])  # |111⟩
    # print("Collapsed Encoded State:", state_to_braket(collapsed_state))

    # Instantiate noise model with p = 1.0 for deterministic flip
    noise_model = BitFlipNoiseCollapsed(p=1.0)

    # Flip qubit 1 (middle qubit of |111⟩ => becomes |101⟩)
    current = noise_model._flip_qubit_with_prob(current, key, qubit_index=1)
    print("Noisy Encoded State (after bit-flip on qubit 1):", state_to_braket(current))

    # 4. Decode
    current = code.decode_collapsed(current)
    print("Decoded Logical State:", state_to_braket(current))


if __name__ == "__main__":
    noise_test()
