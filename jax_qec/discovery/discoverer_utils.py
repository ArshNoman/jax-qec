import itertools
from typing import List
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


def symplectic_commute(v1: jnp.ndarray, v2: jnp.ndarray) -> bool:
    """
    Returns True if two symplectic vectors commute.
    Uses the symplectic inner product:
        v1 ⋅ v2 = x1⋅z2 + x2⋅z1 mod 2
    where each v is [x|z] (length 2n)
    """
    n = v1.shape[0] // 2
    x1, z1 = v1[:n], v1[n:]
    x2, z2 = v2[:n], v2[n:]
    inner = jnp.dot(x1, z2) + jnp.dot(x2, z1)
    return inner % 2 == 0


def all_commute(generators: list) -> bool:
    """
    Returns True if all symplectic generators commute pairwise.
    """
    for i in range(len(generators)):
        for j in range(i + 1, len(generators)):
            if not symplectic_commute(generators[i], generators[j]):
                return False
    return True


def are_independent(generators: list) -> bool:
    """
    Returns True if the generators are linearly independent over GF(2).
    """
    if not generators:
        return False
    mat = jnp.stack(generators).T % 2
    rank = jnp.linalg.matrix_rank(mat)
    return rank == len(generators)


def is_valid_stabilizer_set(generators: list) -> bool:
    """
    Returns True if generators define a valid stabilizer code:
    - All generators commute
    - They are linearly independent
    """
    return all_commute(generators) and are_independent(generators)


def valid_nk_code(generators: List[jnp.ndarray], n: int, k: int) -> bool:
    """
    Returns True if:
    - Number of generators == n - k
    - Generators commute pairwise
    - Generators are independent (full rank over GF(2))
    """
    if len(generators) != n - k:
        return False
    return is_valid_stabilizer_set(generators)


def diagnostics(generators, n, k):
    """
    Returns a string explaining why the stabilizer set is invalid.
    """
    reasons = []

    if len(generators) != n - k:
        reasons.append(f"Expected {n - k} generators, got {len(generators)}.")

    if not all_commute(generators):
        reasons.append("Generators do not commute pairwise.")

    if not are_independent(generators):
        reasons.append("Generators are linearly dependent.")

    return " | ".join(reasons) if reasons else "Valid stabilizer set."
