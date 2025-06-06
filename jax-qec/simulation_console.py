from utils import state_to_braket, logical_zero, logical_one, plus_state, minus_state, bit_flip

from decoders.repetition_decoder import RepetitionXDecoder
from noise.phase_flip import PhaseFlipNoiseCollapsed
from simulator.simulate import simulate_qec_cycle
from noise.bit_flip import BitFlipNoise
from codes import RepetitionEncode

import jax.random as random
import jax.numpy as jnp
from jax import lax
import jax

import random as r

# logical_zero = jnp.array([1.0, 0.0])  # Logical -> |0⟩     [1 0 0 0 0 0 0 0] (encoded)
# logical_one = jnp.array([0.0, 1.0])  # Logical -> |1⟩     [0 0 0 0 0 0 0 1] (encoded)
# plus_state = jnp.array([1.0 / jnp.sqrt(2), 1.0 / jnp.sqrt(2)])  # |+⟩ state -> [1/√2, 1/√2]     [0.70710677 0 0 0 0 0 0 0.70710677]
# minus_state = jnp.array([1.0 / jnp.sqrt(2), -1.0 / jnp.sqrt(2)])  # |-⟩ state -> [1/√2, -1/√2]     [0.70710677 0 0 0 0 0 0 -0.70710677]


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
    #
    # current = logical_zero
    # print("Logical State:", state_to_braket(current))
    #
    # # Encoding
    # code = RepetitionEncode(3)
    # current = code.encode(current)
    # print("Encoded State:", state_to_braket(current))
    #
    noise_model = BitFlipNoise(p=1.0)
    #
    # # Testing probability flip on a collapsed state
    # # key, subkey = jax.random.split(key)
    # # current = noise_model.probability_flip(current, subkey, qubit_index=2)
    # # print("Collapsed State after flip on qubit 1:", state_to_braket(current))
    #
    # # Test apply() on collapsed state: flip all qubits
    # key, subkey = jax.random.split(key)
    # current = noise_model.apply(current, subkey)
    # print("Collapsed State after apply() to all qubits:", state_to_braket(current))
    #
    # # Decode collapsed state to logical state
    # decoded = code.decode_collapsed(current)
    # print("Decoded Logical State:", state_to_braket(decoded))

    current = plus_state
    print("Superposition State:", current)

    # Encoding
    code = RepetitionEncode(3)
    current = code.encode(current)
    print("Encoded State:", state_to_braket(current))

    key, subkey = jax.random.split(key)
    current = noise_model.probability_flip(current, subkey, qubit_index=0)
    print("Superposition after flip on qubit 0:", current)

    # Apply full bit-flip noise to superposition state
    # key, subkey = jax.random.split(key)
    # current = noise_model.apply(current, subkey)
    # print("Superposition after apply() to all qubits:", state_to_braket(current))

    # 11. Confirm normalization
    norm = jnp.linalg.norm(current)
    print("Superposition Norm (should be 1.0):", norm)


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
    current = noise_model.probability_flip(current, subkey, qubit_index=1)
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


def main():
    current = logical_zero
    code = RepetitionEncode(3)
    noise = BitFlipNoise(0.3)
    decoder = RepetitionXDecoder(code)
    key = jax.random.PRNGKey(0)

    err_rate = estimate_logical_error_rate(
        current, code=code, noise_model=noise, decoder=repetition_decoder, key=key, trials=1000
    )

    print("Logical Error Rate:", err_rate)


def batched_bit_flip_example():
    print("=== Batched Bit Flip Noise Test ===")

    key = random.PRNGKey(0)
    noise_model = BitFlipNoise(p=1.0)  # Always flip for clarity

    # Create a batch of two superposition states: (|000⟩ + |111⟩) / √2
    plus_state = jnp.zeros(8)
    plus_state = plus_state.at[0].set(1 / jnp.sqrt(2))
    plus_state = plus_state.at[7].set(1 / jnp.sqrt(2))

    batch = jnp.stack([plus_state, plus_state])  # shape = (2, 8)

    print("Original Batch:")
    for i, state in enumerate(batch):
        print(f"  State {i}:", state_to_braket(state))

    # Apply bit-flip noise
    key, subkey = random.split(key)
    noisy_batch = noise_model.apply(batch, subkey)

    print("\nNoisy Batch (after bit flips on all qubits):")
    for i, state in enumerate(noisy_batch):
        print(f"  State {i}:", state_to_braket(state))

    # Optional: Confirm norms
    norms = jnp.linalg.norm(noisy_batch, axis=1)
    print("\nNorms of noisy states (should be 1.0):", norms)


def decoder_test():

    current = logical_zero  # |0⟩
    code = RepetitionEncode(3)
    key = jax.random.PRNGKey(r.randint(0,1000))

    print('Initial logical state:', current, '->', state_to_braket(current))

    current = code.encode(current)
    print('Encoded logical state:', current, '->', state_to_braket(current))

    # Bit-flip noise model
    noise = BitFlipNoise(p=1.0)
    key, subkey = jax.random.split(key)

    # current = noise.apply(current, key)
    current = noise.probability_flip(current, key, 0)
    current = noise.probability_flip(current, key, 1)
    print("Noisy State:", current, '->', state_to_braket(current))

    # Syndrome measurement
    syndrome = code.measure_syndrome_collapsed(current)
    print('Syndrome:', syndrome)

    # Decode
    decoder = RepetitionXDecoder(code)
    current = decoder.decode(current, syndrome)
    print('Decoded state:', current, '->', state_to_braket(current))


if __name__ == "__main__":
    main()

