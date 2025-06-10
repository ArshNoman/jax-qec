from utils.symplectic_utils import generate_all_symplectic_generators

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

        reward = float(self.is_valid_code())
        return self._get_state(), reward, done, {}

    def is_valid_code(self) -> bool:
        """
        A valid [[3,1]] repetition code requires 2 linearly independent Z-type
        generators. We check if the current stabilizers meet that condition.
        """
        if len(self.generators) != self.max_steps:
            return False
        mat = jnp.stack(self.generators)
        rank = jnp.linalg.matrix_rank(mat)
        return rank == self.max_steps