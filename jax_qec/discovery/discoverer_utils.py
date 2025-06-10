import itertools
import jax.numpy as jnp

def generate_all_symplectic_generators(n):
    """
    Generate all possible 2n-bit binary vectors that represent n-qubit Pauli operators
    in symplectic form: [X-part | Z-part].
    - Each generator is a length-2n binary vector: [x1, ..., xn, z1, ..., zn]
    - Excludes the all-zero vector (identity).
    """
    total_bits = 2 * n
    all_binary_vectors = itertools.product([0, 1], repeat=total_bits)
    non_identity = [jnp.array(v, dtype=jnp.uint8) for v in all_binary_vectors if any(v)]
    return non_identity