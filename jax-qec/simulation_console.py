from noise.phase_flip import PhaseFlipNoiseCollapsed
from noise.bit_flip import BitFlipNoiseCollapsed
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

    current = jnp.array([0.0, -1.0])
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

    current = code.decode_collapsed(current)
    print("Decoded state:", current, "->", state_to_braket(current))

    # key = jax.random.PRNGKey(random.randint(1, 100))
    # key, subkey = jax.random.split(key)
    # current = code.decode_superposition(current, subkey)
    # print("Decoded state:\n", current, "->", state_to_braket(current))


def bit_flip_test():

    key = jax.random.PRNGKey(42)

    # 1. Define logical state (|1⟩ in this case)
    current = logical_one  # |1⟩
    print("Logical State:", state_to_braket(current))

    # 2. Encode using 3-qubit repetition code
    code = RepetitionEncode(3)
    current = code.encode(current)
    print("Encoded State:", state_to_braket(current))

    # Instantiate noise model with p = 1.0 for deterministic flip
    noise_model = BitFlipNoiseCollapsed(p=1.0)

    # Flip qubit 1 (middle qubit of |111⟩ => becomes |101⟩)
    current = noise_model.probability_flip(current, key, qubit_index=1)
    current = noise_model.probability_flip(current, key, qubit_index=2)
    print("Noisy Encoded State (after bit-flip on qubit 1):", state_to_braket(current))

    # 3. Decode
    current = code.decode_collapsed(current)
    print("Decoded Logical State:", state_to_braket(current))


def phase_flip_test():

    key = jax.random.PRNGKey(123)

    current = logical_one
    print("Logical State:", state_to_braket(current))

    # 1. Encode using 3-qubit repetition code
    code = RepetitionEncode(3)
    current = code.encode(current)
    print("Encoded State:", state_to_braket(current))

    noise_model = PhaseFlipNoiseCollapsed(p=1.0)

    # 2. Test probability_flip() on qubit 1
    key, subkey = jax.random.split(key)
    current = noise_model.probability_flip(current, subkey, qubit_index=1) ≈A
    print("After probability_flip on qubit 1:", state_to_braket(current))

    # 3. Test _flip_phase_if_one() directly on qubit 2
    key, subkey = jax.random.split(key)
    current = noise_model.flip_phase_if_one(current, subkey, qubit_index=2)
    print("After flip_phase_if_one on qubit 2:", state_to_braket(current))

    # 4. Test apply() across all qubits (this will phase flip again on qubit 1 and 2)
    key, subkey = jax.random.split(key)
    current = noise_model.apply(current, subkey)
    print("After apply() to all qubits:", state_to_braket(current))

    # 5. Decode and print result
    current = code.decode_collapsed(current)
    print("Decoded Logical State:", state_to_braket(current))


if __name__ == "__main__":
    phase_flip_test()
