from jax_qec.clifford import gates
from jax_qec.clifford.gates import apply
from jax_qec.utils import *

# Create a qubit that is a logical zero
qubit = logical_zero

# Create a sequence of Pauli Gates
PauliGates = gateStack("XZ")

print(PauliGates)

# Apply them to the inital qubit and print the result
qubit = apply(qubit, PauliGates)
print(qubit)

print(s)