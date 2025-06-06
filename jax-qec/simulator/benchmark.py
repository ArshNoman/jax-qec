

def is_logical_error(corrected_state: jnp.ndarray, logical_state: jnp.ndarray) -> bool:
    """
    Check if corrected state corresponds to the input logical state.

    Parameters:
    - corrected_state: jnp.ndarray (2^n,)
    - logical_state: jnp.ndarray (2,)

    Returns:
    - True if logical error occurred
    """
    return not jnp.allclose(corrected_state, logical_state, atol=1e-4)