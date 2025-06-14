from utils import state_to_braket, logical_zero, logical_one, plus_state, minus_state, bit_flip

from decoders.repetition_decoder import RepetitionXDecoder, RepetitionZDecoder
from discovery.rl_discoverer import QECEnv, RLCodeDiscoverer
from qecsimulator.simulate import simulate_superposition
from qecsimulator.benchmark import estimate_error_rate
from noise.phase_flip import PhaseFlipNoise
from noise.bit_flip import BitFlipNoise
from codes import RepetitionEncode

import jax.random as random
import jax.numpy as jnp
from jax import lax
import numpy as np
import jax

import random as r


def codes_test():
    code = RepetitionEncode(3)

    current = logical_zero
    # print("Encoded state:", current)
    print("Current state:", current, "->", state_to_braket(current))

    current = code.encode(current)
    # print("Encoded state:\n", current)
    print("Encoded state:\n", current, "->", state_to_braket(current))

    current = bit_flip(current, 0)
    print("Flipped encoded state:", state_to_braket(current))
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
    key = jax.random.PRNGKey(42)
    noise = PhaseFlipNoise(p=1.0)

    # Encode states using repetition code
    encoder = RepetitionEncode(3)
    encoded_zero = encoder.encode(logical_zero)
    encoded_plus = encoder.encode(plus_state)

    # Apply noise
    noisy_zero = noise.apply(encoded_zero, key)
    key, subkey = jax.random.split(key)
    noisy_plus = noise.apply(encoded_plus, subkey)

    print("Original Encoded Logical Zero:", encoded_zero)
    print("Noisy Logical Zero (should remain unchanged):", noisy_zero)

    print("\nOriginal Encoded Logical Plus:", encoded_plus)
    print("Noisy Logical Plus (phases should flip):", noisy_plus)


def main():
    current = plus_state
    code = RepetitionEncode(3)
    noise = BitFlipNoise(0.8)
    decoder = RepetitionXDecoder(code)
    key = jax.random.PRNGKey(432)

    err_rate = estimate_error_rate(current, code=code, noise_model=noise, decoder=decoder, key=key, trials=1000)

    print("Logical Error Rate:", err_rate)


def batched_bit_flip_example():
    key = random.PRNGKey(0)
    noise_model = BitFlipNoise(p=1.0)
    code = RepetitionEncode(3)

    batch = jnp.stack([code.encode(plus_state), code.encode(plus_state)])  # shape = (2, 8)

    print(batch)

    print("Original Batch:")
    for i, state in enumerate(batch):
        print(f"  State {i}:", state)

    # Apply bit-flip noise
    key, subkey = random.split(key)
    noisy_batch = noise_model.apply(batch, subkey)

    print("\nNoisy Batch (after bit flips on all qubits):")
    for i, state in enumerate(noisy_batch):
        print(f"  State {i}:", state)

    # Optional: Confirm norms
    norms = jnp.linalg.norm(noisy_batch, axis=1)
    print("\nNorms of noisy states (should be 1.0):", norms)


def decoder_test():
    current = logical_one  # |0⟩
    code = RepetitionEncode(3)
    key = jax.random.PRNGKey(r.randint(0, 1000))

    print('Initial logical state:', current, '->', state_to_braket(current))

    current = code.encode(current)
    print('Encoded logical state:', current, '->', state_to_braket(current))

    # Bit-flip noise model
    noise = PhaseFlipNoise(p=1.0)
    key, subkey = jax.random.split(key)

    # current = noise.apply(current, key)
    current = noise.probability_flip(current, key, 0)
    # current = noise.probability_flip(current, key, 1)
    print("Noisy State:", current, '->', state_to_braket(current))

    # Syndrome measurement
    syndrome = code.measure_syndrome_collapsed(current)
    print('Syndrome:', syndrome)

    # Decode
    decoder = RepetitionZDecoder(code)
    current = decoder.decode(current, syndrome)
    print('Decoded state:', current, '->', state_to_braket(current))


def stabilizer_validation():
    """
    Two operators commute if the order in which you apply them doesn't matter: AB = BA
    In stabilizer codes, this is critical: all generators must commute so they can be
    simultaneously diagonalizable i.e. share a common +1 eigenspace (the code space).

    Symplectic form: We represent each n-qubit Pauli operator as a binary vector
    of length 2n: [x|z] where xi = 1 if the operator X is on qubit i and similarly,
    zi = 1 if the operator Z is on qubit i.

    Commuting in symplectic form: Let p1 = [x1|z1] and p2 = [x2|z2]
    They commute iff x1*z2 + x2*z1 = 0 mod 2 (symplectic inner product)
    If it's 0 mod 2, they commute. If it's 1 then they anti-commute.
    """

    def to_symplectic(x_str, z_str):
        x = jnp.array([int(b) for b in x_str], dtype=jnp.uint8)
        z = jnp.array([int(b) for b in z_str], dtype=jnp.uint8)
        return jnp.concatenate([x, z])

    n = 3  # number of qubits
    k = 1  # number of logical qubits

    # Example 1: Correct [[3,1]] repetition code
    s1 = to_symplectic("000", "110")  # ZZI
    s2 = to_symplectic("000", "011")  # IZZ
    print("Example 1 - [ZZI, IZZ]:", valid_nk_code([s1, s2], n, k))

    # Example 2: Non-commuting pair
    s3 = to_symplectic("010", "101")  # ZIZ
    s4 = to_symplectic("101", "010")  # XZX
    print("Example 2 - [ZIZ, XZX]:", valid_nk_code([s3, s4], n, k))

    # Example 3: Duplicate stabilizer (not independent)
    s5 = to_symplectic("000", "110")  # ZZI
    print("Example 3 - [ZZI, ZZI]:", valid_nk_code([s5, s5], n, k))

    # Example 4: 3 generators → too many for [[3,1]]
    s6 = to_symplectic("000", "011")  # IZZ
    s7 = to_symplectic("000", "110")  # ZZI
    s8 = to_symplectic("000", "101")  # ZIZ
    print("Example 4 - [IZZ, ZZI, ZIZ]:", valid_nk_code([s6, s7, s8], n, k))


def symplectic_to_pauli(symp: jnp.ndarray) -> str:
    """
    Convert a binary symplectic vector to a Pauli string.
    """
    n = symp.shape[0] // 2
    x = symp[:n]
    z = symp[n:]
    pauli = []
    for xi, zi in zip(x, z):
        if xi == 0 and zi == 0:
            pauli.append('I')
        elif xi == 1 and zi == 0:
            pauli.append('X')
        elif xi == 0 and zi == 1:
            pauli.append('Z')
        elif xi == 1 and zi == 1:
            pauli.append('Y')
    return ''.join(pauli)


def rlagent():
    print("Running RL QEC Environment Test")

    num_episodes = 100

    env = QECEnv(n=3, k=1)
    agent = RLCodeDiscoverer(env)
    key = jax.random.PRNGKey(0)

    best_reward = -1
    best_generators = []

    for ep in range(num_episodes):
        total_reward = 0
        state = env.reset()
        done = False
        key, subkey = jax.random.split(key)

        while not done:
            subkey, action_key = jax.random.split(subkey)
            action, probs = agent.sample_action(state, action_key)
            state, reward, done, info = env.step(action)
            total_reward += reward

        print(f"Episode {ep + 1}: Reward = {total_reward:.4f}")
        if total_reward > best_reward:
            best_reward = total_reward
            best_generators = list(env.generators)

    print("\nBest Episode Reward:", best_reward)
    print("Best Stabilizer Generators:")

    for i, g in enumerate(best_generators):
        print(f"  Generator {i + 1}:")
        print(f"    Symplectic: {g.tolist()}")
        print(f"    Pauli:      {symplectic_to_pauli(g)}")


if __name__ == "__main__":
    decoder_test()
