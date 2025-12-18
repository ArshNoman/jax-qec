from jax_qec.clifford import gates
from jax_qec.clifford.gates import apply
from jax_qec.utils import *

# Create a qubit that is a logical zero
qubit = logical_zero

# Create a sequence of Pauli Gates
PauliGates = gateStack("XZ")

# Apply them to the initial qubit and print the result
qubit = apply(qubit, PauliGates)
print("Gate XZ applied to a logical zero: ", qubit, '\n')

# Create two standard qubits
a = logical_one
b = logical_zero

# Apply the Kronecker product between them to create a block matrix
two_qubit_state = kron(a, b)

# Standard matrix multiplication to apply the CNOT
out = c @ two_qubit_state
print("CNOT applied to a logical one and logical zero:", out, '\n')
# the output of the above is [0. 0. 0. 1.], this is equal to the computational basis state ∣11⟩