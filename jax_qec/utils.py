import jax.numpy as jnp

logical_zero = jnp.array([1.0, 0.0])  # |0⟩ -> [1, 0]
logical_one = jnp.array([0.0, 1.0])  # |1⟩ -> [0, 1]
plus_state = jnp.array([1.0 / jnp.sqrt(2), 1.0 / jnp.sqrt(2)])  # |+⟩ state -> [1/√2, 1/√2]
minus_state = jnp.array([1.0 / jnp.sqrt(2), -1.0 / jnp.sqrt(2)])  # |-⟩ state -> [1/√2, -1/√2]

pauli_x = jnp.array([[0,1],[1,0]])
pauli_z = jnp.array([[1,0],[0,-1]])
h = jnp.array([[1.0 / jnp.sqrt(2),1.0 / jnp.sqrt(2)],[1.0 / jnp.sqrt(2),-1.0 / jnp.sqrt(2)]])

GATE_MAP = {"X": pauli_x, "Z": pauli_z}

def gateStack(gates: str):
    """
    Convert a string of gates to a stack of JAX arrays and thus prepared
    for use in applying gates to qubits

    Parameters:
    - gates: str sequence of a gate or a sequence of gates

    Returns:
    - JAX array of a stack of gates
    """
    return jnp.stack([GATE_MAP[i] for i in gates])
