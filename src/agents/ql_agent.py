"""
Q Learning agent class
"""
from collections import defaultdict

import numpy as np

from src.agents.agent import Agent


class QL(Agent):
    """
    Q Learning agent class
    """
    def __init__(
        self, 
        learning_rate: float, 
        *args, 
        **kwargs
    ) -> None:
        """
        Initializes a new tabular Q-learning agent.

        Args:
            learning_rate (float): Learning rate for Q-value update
        """
        super(QL, self).__init__(*args, **kwargs)
        self.lr = learning_rate


    def get_Q_values(self, s: tuple) -> np.ndarray:
        """
        Returns the action value function evaluated in the specified state.
        
        Args:
            s (tuple): Current state

        Returns:
            np.ndarray : The estimated Q-values of the current state
        """
        return self.Q[s]

    def train_agent(self, s: tuple, a: int, r: float, c: float, s1: tuple, gamma: float) -> None:
        """
        Trains the current agent on the provided transition.

        Args:
            s (tuple): Current state
            a (int): Current action
            r (float): Immediate reward 
            c (float): Immediate cost
            s1 (tuple): Next state
            gamma (float): Discount factor
        """
        target = r + gamma * np.max(self.Q[s1])
        error = target - self.Q[s][a]
        self.Q[s][a] += self.lr * error

    def set_active_training_task(self) -> None:
        """
        Sets the new task as active task
        """
        super(QL, self).set_active_training_task()
        self.Q = defaultdict(
            lambda: np.random.uniform(low=-0.01, high=0.01, size=(self.n_actions,))
        )
