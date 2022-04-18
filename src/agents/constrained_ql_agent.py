"""
Constrained Q Learning agent class
"""
import sys
from collections import defaultdict
import random

import numpy as np

from src.agents.agent import Agent


class ConstrainedQL(Agent):
    """
    Constrained Q Learning agent class
    """
    def __init__(
        self, 
        learning_rate: float, 
        threshold: float,
        *args, 
        **kwargs
    ) -> None:
        """
        Initializes a new Constrained Q-learning agent.

        Args:
            learning_rate (float): Learning rate for Q-value update
            threshold (float): One step cost threshold
        """
        super().__init__(*args, **kwargs)
        self.lr = learning_rate
        self.threshold = threshold

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
        # Prune unsafe actions
        q1 = self.Q[s1]

        for action in range(self.n_actions):
            if not self.safe_actions[s1[0]][action]:
                # Temporarily set unsafe action q value to negative value
                q1[action] = -10

        target = r + gamma * np.max(q1)
        self.Q[s][a] = (1 - self.lr) * self.Q[s][a] + self.lr * target

    def initialize(self) -> None:
        """
        Initializes or resets task specific agent params
        """
        super().initialize()
        self.interval_violate_hist = []
        self.violation_hist = []

    def set_active_training_task(self) -> None:
        """
        Sets the new task as active task
        """
        super().set_active_training_task()
        self.interval_violate_count = 0
        self.violation_count = 0
        self.Q = defaultdict(
            lambda: np.random.uniform(low=-0.01, high=0.01, size=(self.n_actions,))
        )
        costs = list(self.active_task.obstacles.values())
        self.task_threshold = (max(costs) - min(costs)) * self.threshold + min(costs)
        self.safe_actions = defaultdict(lambda: np.ones(self.n_actions))

    def update_hist(self) -> None:
        """
        Updates the agent performance stats
        """
        super().update_hist()
        if self.steps % self.logging_freq == 0:
            self.interval_violate_hist.append(self.interval_violate_count)
            self.interval_violate_count = 0
        self.violation_hist.append(self.violation_count)

    def _epsilon_greedy(self, q: np.ndarray) -> int:
        """
        Epsilon greedy policy

        Args:
            q (np.ndarray): Current state Q values

        Returns:
            int: Action picked by epsilon greedy
        """
        # Prune unsafe actions
        safe_actions = list(range(self.n_actions))
        for action in range(self.n_actions):
            if not self.safe_actions[self.s[0]][action]:
                safe_actions.remove(action)
                # Temporarily set unsafe action q value to negative value
                q[action] = -10

        # Sample from a Bernoulli distribution with epsilon
        if random.random() <= self.epsilon:
            if not safe_actions:
                error_message = "A state should not have all actions labelled as unsafe, check config or action prune logic!"
                self.logger.error(error_message)
                sys.exit(1)
            a = random.choice(safe_actions)
        else:
            a = np.argmax(q)

        # Decrease epsilon gradually
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        return a

    def state_transition(self, a: int) -> tuple:
        """
        Returns the transitioned state

        Args:
            a (int): Current action

        Returns:
            tuple: The new state, immediate reward and cost, and if terminal state is reached
        """
        s1, r, c, terminal = self.active_task.transition(a)
        if c > self.task_threshold:
            self.safe_actions[self.s[0]][a] = 0
            self.violation_count += 1
            self.interval_violate_count += 1
        return s1, r, c, terminal
