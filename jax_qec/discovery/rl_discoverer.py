from discovery.discoverer_utils import generate_all_symplectic_generators, valid_nk_code, diagnostics

import jax.numpy as jnp
import numpy as np
from typing import List, Tuple


class QECEnv:
    def __init__(self):
        self.step_count = None
        self.generators = None
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
        return self.get_state()

    def get_state(self) -> jnp.ndarray:
        """Returns the padded matrix of symplectic generators added so far."""
        padded = jnp.zeros((self.max_steps, 2 * self.n), dtype=jnp.uint8)
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

        valid = valid_nk_code(self.generators, self.n, self.k)
        reward = float(valid)

        info = {
            "valid": valid,
            "reason": "Valid stabilizer set." if valid else diagnostics(self.generators, self.n, self.k)
        }

        return self.get_state(), reward, done, info

    def is_valid_code(self) -> bool:
        """
        Returns True if the current set of generators defines a valid [[n, k]] code.
        """
        return valid_nk_code(self.generators, self.n, self.k)


class RLCodeDiscoverer:
    def __init__(self, env: QECEnv):
        self.env = env
        self.best_code = None
        self.best_reward = -1

    def search(self, num_episodes=1000):
        for ep in range(num_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0
            actions = []

            while not done:
                action = np.random.choice(len(self.env.possible_generators))
                actions.append(action)
                state, reward, done, info = self.env.step(action)
                total_reward += reward

            if total_reward > self.best_reward:
                self.best_reward = total_reward
                self.best_code = list(self.env.generators)
                print(f"\nNew Best Code (Ep {ep}):")
                for g in self.best_code:
                    print(g)
                print("Reward:", self.best_reward)
