import numpy as np


class QLearningAgent:
    def __init__(
        self,
        state_size,
        action_size,
        alpha,
        gamma,
        epsilon,
        seed,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.q_table = np.zeros((state_size, action_size), dtype=np.float32)

    def select_action(self, state):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.action_size))
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state):
        best_next = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error
