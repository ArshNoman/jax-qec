import jax.numpy as jnp
import jax.lax as lax
import jax

def step(state, gate):
    return gate @ state, None

@jax.jit
def apply(state: jnp.ndarray, gates: jnp.ndarray) -> jnp.ndarray:
    """
    Apply a gate or a sequence of gates to a qubit

    Parameters:
    - state: 1D or 2D jnp.ndarray
    - gates: JAX array of gates to sequentially apply

    Returns:
    - Modified state with flips applied
    """

    state_, _ = lax.scan(step, state, gates)
    return state_
