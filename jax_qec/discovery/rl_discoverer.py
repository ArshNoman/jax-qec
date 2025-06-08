import jax.numpy as jnp
import numpy as np
from typing import List, Tuple


class QECEnv:
    def __init__(self):
        self.n = 3  # 3 physical qubits
        self.k = 1  # 1 logical qubit
        self.max_steps = self.n - self.k
        self.possible_generators = [
            jnp.array([1, 1, 0]),  # XX_
            jnp.array([0, 1, 1]),  # _XX
            jnp.array([1, 0, 1]),  # X_X
        ]
        self.reset()

    def reset(self) -> jnp.ndarray:
        self.generators: List[jnp.ndarray] = []
        self.step_count = 0
        return self._get_state()

    def _get_state(self) -> jnp.ndarray:
        # Pad with zeros to get shape (max_steps, n)
        if len(self.generators) == 0:
            return jnp.zeros((self.max_steps, self.n))
        else:
            padded = jnp.zeros((self.max_steps, self.n))
            for i, g in enumerate(self.generators):
                padded = padded.at[i].set(g)
            return padded

    def step(self, action: int) -> Tuple[jnp.ndarray, float, bool, dict]:
        gen = self.possible_generators[action]
        self.generators.append(gen)
        self.step_count += 1
        done = self.step_count >= self.max_steps

        reward = float(self.is_valid_code())
        return self._get_state(), reward, done, {}

    def is_valid_code(self) -> bool:
        # Check if we have exactly 2 independent generators
        if len(self.generators) != self.max_steps:
            return False
        mat = jnp.stack(self.generators)
        rank = jnp.linalg.matrix_rank(mat)
        return rank == self.max_steps