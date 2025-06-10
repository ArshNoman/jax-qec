from discovery.discoverer_utils import generate_all_symplectic_generators, valid_nk_code

import jax.numpy as jnp
import numpy as np
from typing import List, Tuple


class QECEnv:
    def __init__(self):
        self.n = 3  # Number of physical qubits
        self.k = 1  # Number of logical qubits
        self.max_steps = self.n - self.k  # = 2 stabilizer generators
        self.possible_generators = [
            jnp.array([1, 1, 0]),  # ZZI
            jnp.array([0, 1, 1]),  # IZZ
            jnp.array([1, 0, 1]),  # ZIZ (invalid for repetition code)
        ]

        self.possible_generators = generate_all_symplectic_generators(self.n)
        self.reset()

    def reset(self) -> jnp.ndarray:
        """Resets the environment to an empty code."""
        self.generators: List[jnp.ndarray] = []
        self.step_count = 0
        return self._get_state()

    def _get_state(self) -> jnp.ndarray:
        """Returns the padded stabilizer matrix as current state."""
        padded = jnp.zeros((self.max_steps, self.n))
        for i, g in enumerate(self.generators):
            padded = padded.at[i].set(g)
        return padded

    def step(self, action: int) -> Tuple[jnp.ndarray, float, bool, dict]:
        """
        Executes an action by adding a stabilizer generator from the predefined list.
        Returns: new_state, reward, done, info
        """
        gen = self.possible_generators[action]
        self.generators.append(gen)
        self.step_count += 1
        done = self.step_count >= self.max_steps

        reward = float(
            len(self.generators) == self.max_steps and
            is_valid_stabilizer_set(self.generators)
        )
        return self._get_state(), reward, done, {}

    def is_valid_code(self) -> bool:
        """
        Returns True if the current set of generators defines a valid [[n, k]] code.
        """
        return valid_nk_code(self.generators, self.n, self.k)
