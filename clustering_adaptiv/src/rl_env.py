actions = [
    "lr_up",
    "lr_down",
    "dropout_up",
    "dropout_down",
    "k_up",
    "k_down",
]


class HyperparamEnv:
    def __init__(
        self,
        lr_values,
        dropout_values,
        k_values,
    ):
        self.lr_values = lr_values
        self.dropout_values = dropout_values
        self.k_values = k_values
        self.lr_idx = min(1, len(lr_values) - 1)
        self.dropout_idx = min(1, len(dropout_values) - 1)
        self.k_idx = min(1, len(k_values) - 1)

    def reset(self):
        self.lr_idx = min(1, len(self.lr_values) - 1)
        self.dropout_idx = min(1, len(self.dropout_values) - 1)
        self.k_idx = min(1, len(self.k_values) - 1)
        return self.state_index()

    def step(self, action):
        if action == 0:
            self.lr_idx = min(self.lr_idx + 1, len(self.lr_values) - 1)
        elif action == 1:
            self.lr_idx = max(self.lr_idx - 1, 0)
        elif action == 2:
            self.dropout_idx = min(self.dropout_idx + 1, len(self.dropout_values) - 1)
        elif action == 3:
            self.dropout_idx = max(self.dropout_idx - 1, 0)
        elif action == 4:
            self.k_idx = min(self.k_idx + 1, len(self.k_values) - 1)
        elif action == 5:
            self.k_idx = max(self.k_idx - 1, 0)
        return self.state_index()

    def current_values(self):
        return (
            self.lr_values[self.lr_idx],
            self.dropout_values[self.dropout_idx],
            self.k_values[self.k_idx],
        )

    def state_index(self):
        return (
            (self.lr_idx * len(self.dropout_values) + self.dropout_idx)
            * len(self.k_values)
            + self.k_idx
        )

    def state_size(self):
        return len(self.lr_values) * len(self.dropout_values) * len(self.k_values)

    def action_size(self):
        return len(actions)
