import jax.numpy as jnp

logical_zero = jnp.array([1.0, 0.0])  # |0⟩ -> [1, 0]
logical_one = jnp.array([0.0, 1.0])  # |1⟩ -> [0, 1]
plus_state = jnp.array([1.0 / jnp.sqrt(2), 1.0 / jnp.sqrt(2)])  # |+⟩ state -> [1/√2, 1/√2]
minus_state = jnp.array([1.0 / jnp.sqrt(2), -1.0 / jnp.sqrt(2)])  # |-⟩ state -> [1/√2, -1/√2]

pauli_x = jnp.array([[0,1],[1,0]])
pauli_z = jnp.array([[1,0],[0,-1]])
h = jnp.array([[1.0 / jnp.sqrt(2),1.0 / jnp.sqrt(2)],[1.0 / jnp.sqrt(2),-1.0 / jnp.sqrt(2)]])
s = jnp.array([[1.0, 0.0], [0.0, 1.0j]])
c = jnp.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,0,1],
    [0,0,1,0]
])

GATE_MAP = {"X": pauli_x, "Z": pauli_z, "H": h, "S": s,}

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

def kron(a: jnp.ndarray, b: jnp.ndarray):
    """
    Return a block matrix of a and b, known as the Kronecker product

    Parameters:
    - a: 1D jnp.ndarray
    - b: another 1D jnp.ndarray

    Returns:
    - Kronecker product of a and b
    """
    return jnp.kron(a, b)
