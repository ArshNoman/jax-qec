from discovery.discoverer_utils import generate_all_symplectic_generators, generate_z_only_generators, valid_nk_code, diagnostics
from decoders.repetition_decoder import RepetitionXDecoder
from qecsimulator.benchmark import estimate_error_rate
from noise.bit_flip import BitFlipNoise
from codes import StabilizerCode
from utils import logical_zero


from flax.training.train_state import TrainState
from typing import List, Tuple
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import jax


class QECEnv:
    def __init__(self, n, k=1):
        self.step_count = None
        self.generators = None
        self.n = n  # Number of physical qubits
        self.k = k  # Number of logical qubits
        self.max_steps = self.n - self.k  # = 2 stabilizer generators
        self.possible_generators = generate_z_only_generators(self.n)
        # self.possible_generators = generate_all_symplectic_generators(self.n)

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
        reward = 0.0
        if len(self.generators) == self.max_steps and self.is_valid_code():
            # Build a code object from the symplectic generators
            code = StabilizerCode(self.generators)
            noise = BitFlipNoise(p=0.05)
            decoder = RepetitionXDecoder(code)
            key = jax.random.PRNGKey(np.random.randint(0, 1e6))

            error_rate = estimate_error_rate(logical_zero, code, noise, decoder, key, trials=1000)
            reward = float(jnp.clip(1.0 - error_rate, 0.0, 1.0))

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
    def __init__(self, env: QECEnv, seed: int = 0, learning_rate: float = 1e-2):
        self.env = env
        self.key = jax.random.PRNGKey(seed)
        self.policy = PolicyNet(action_dim=len(env.possible_generators))

        # Initialize parameters
        dummy_state = jnp.zeros((env.max_steps, 2 * env.n))
        self.params = self.policy.init(self.key, dummy_state)

        # Optimizer
        self.optimizer = optax.adam(learning_rate)
        self.state = TrainState.create(apply_fn=self.policy.apply, params=self.params, tx=self.optimizer)

    def sample_action(self, state, key):
        probs = self.policy.apply(self.state.params, state)
        action = jax.random.choice(key, a=probs.shape[0], p=probs)
        return int(action), probs

    def compute_loss(self, log_probs, rewards):
        """
        REINFORCE loss: L = -E[log(pi(a|s)) * reward]
        """
        returns = jnp.array(rewards)
        log_probs = jnp.stack(log_probs)
        return -jnp.mean(log_probs * returns)

    def train_step(self, log_probs, rewards):
        def loss_fn(param):
            return self.compute_loss(log_probs, rewards)
        grads = jax.grad(loss_fn)(self.state.params)
        self.state = self.state.apply_gradients(grads=grads)

    def search(self, num_episodes=500):
        for ep in range(num_episodes):
            state = self.env.reset()
            done = False
            log_probs = []
            rewards = []
            self.key, subkey = jax.random.split(self.key)

            while not done:
                subkey, sk = jax.random.split(subkey)
                action, probs = self.sample_action(state, sk)
                log_prob = jnp.log(probs[action] + 1e-8)  # avoid log(0)
                state, reward, done, info = self.env.step(action)

                log_probs.append(log_prob)
                rewards.append(reward)

            total_reward = sum(rewards)
            self.train_step(log_probs, rewards)

            if total_reward > 0:
                print(f"\nEpisode {ep}: Reward = {total_reward}")
                for g in self.env.generators:
                    print(g)

            probs = self.policy.apply(self.state.params, state)
            print("Action probabilities:", probs)


class PolicyNet(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass:
        - state shape: (max_steps, 2n)
        - returns: action probabilities (shape: [action_dim])
        """
        x = state.flatten()
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return nn.softmax(x)
