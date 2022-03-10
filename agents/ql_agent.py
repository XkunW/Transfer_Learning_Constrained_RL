from collections import defaultdict
import numpy as np

from agents.agent import Agent


class QL(Agent):
    def __init__(
        self, 
        learning_rate, 
        epsilon=0.1,
        epsilon_decay=1.0,
        epsilon_min=0.0,
        *args, 
        **kwargs
    ):
        """
        Creates a new tabular Q-learning agent.

        Parameters
        ----------
        learning_rate : float
            the learning rate to use in order to update Q-values
        epsilon : float
            the initial exploration parameter for epsilon greedy (defaults to 0.1)
        epsilon_decay : float
            the amount to anneal epsilon in each time step (defaults to 1, no annealing)
        epsilon_min : float
            the minimum allowed value of epsilon (defaults to 0)
        """
        super(QL, self).__init__(*args, **kwargs)
        self.alpha = learning_rate
        self.epsilon_init = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def get_Q_values(self, s, s_enc):
        return self.Q[s]

    def train_agent(self, s, s_enc, a, r, c, s1, s1_enc, gamma):
        target = r + gamma * np.max(self.Q[s1])
        error = target - self.Q[s][a]
        self.Q[s][a] += self.alpha * error

    def set_active_training_task(self):
        super(QL, self).set_active_training_task()
        self.Q = defaultdict(
            lambda: np.random.uniform(low=-0.01, high=0.01, size=(self.n_actions,))
        )
