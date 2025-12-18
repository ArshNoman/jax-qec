import jax.numpy as jnp

logical_zero = jnp.array([1.0, 0.0])  # |0⟩ -> [1, 0]
logical_one = jnp.array([0.0, 1.0])  # |1⟩ -> [0, 1]
plus_state = jnp.array([1.0 / jnp.sqrt(2), 1.0 / jnp.sqrt(2)])  # |+⟩ state -> [1/√2, 1/√2]
minus_state = jnp.array([1.0 / jnp.sqrt(2), -1.0 / jnp.sqrt(2)])  # |-⟩ state -> [1/√2, -1/√2]

pauli_x = jnp.array([[0,1],[1,0]])
pauli_z = jnp.array([[1,0],[0,-1]])

PAULI_MAP = {"X": pauli_x, "Z": pauli_z}

def create_gate_stack(gates: str):
    return jnp.stack([PAULI_MAP[i] for i in gates])
