import jax
import jax.numpy as jnp
from jax import Array


def is_logical_error(corrected_state: jnp.ndarray, logical_state: jnp.ndarray) -> bool:
    """
    Check if corrected state corresponds to the input logical state.

    Parameters:
    - corrected_state: jnp.ndarray (2^n)
    - logical_state: jnp.ndarray (2)

    Returns:
    - True if logical error occurred
    """
    return not jnp.allclose(corrected_state, logical_state, atol=1e-4)


def estimate_error_rate(logical_state: jnp.ndarray, code, noise_model, decoder, key, trials: int = 1000) -> Array:
    """
    Run many QEC cycles and compute logical error rate.

    Returns:
    - Logical error rate (fraction of failed corrections)
    """
    keys = jax.random.split(key, trials)

    def run_trial(k):
        corrected = simulate_qec_cycle(logical_state, code, noise_model, decoder, k)
        return is_logical_error(corrected, logical_state)

    results = jax.vmap(run_trial)(keys)

    return jnp.mean(results.astype(jnp.float32))
