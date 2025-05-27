from codes import QuantumCode, RepititionEncode
import jax.numpy as jnp


def main():
    code = GeneralRepetitionCode(3)
    psi = jnp.array([1.0 / jnp.sqrt(2), 1.0 / jnp.sqrt(2)])  # |+⟩ state -> [1/√2, 1/√2]
    encoded = code.encode(psi)
    print("Encoded state:", encoded)


if __name__ == "__main__":
    main()
