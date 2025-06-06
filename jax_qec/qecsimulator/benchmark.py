from qecsimulator.simulate import simulate_superposition

import jax.numpy as jnp
from jax import Array
import jax


def is_logical_error(corrected_state: jnp.ndarray, logical_state: jnp.ndarray) -> jnp.bool:
    """
    Check if corrected state corresponds to the input logical state.

    Parameters:
    - corrected_state: jnp.ndarray (2^n)
    - logical_state: jnp.ndarray (2)

    Returns:
    - True if logical error occurred
    """
    corrected_index = jnp.argmax(jnp.abs(corrected_state))
    logical_index = jnp.argmax(jnp.abs(logical_state))
    return corrected_index != logical_index


def estimate_error_rate(logical_state: jnp.ndarray, code, noise_model, decoder, key, trials: int = 1000) -> Array:
    """
    Run many QEC cycles and compute logical error rate.

    Returns:
    - Logical error rate (fraction of failed corrections)
    """
    keys = jax.random.split(key, trials)

    def run_trial(k):
        corrected = simulate_superposition(logical_state, code, noise_model, decoder, k)
        return is_logical_error(corrected, logical_state)

    results = jax.vmap(run_trial)(keys)

    return jnp.mean(results.astype(jnp.float32))
