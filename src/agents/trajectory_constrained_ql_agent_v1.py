"""
Trajectory Constrained Reinforcement Learning agent class 1st version
The safety of an action is determined by whether cumulative_cost + c < task threshold
"""
import sys
import random
from collections import defaultdict

import numpy as np

from src.agents.agent import Agent


class TrajectoryConstrainedQLV1(Agent):
    """
    Trajectory Constrained Reinforcement Learning agent class 1st version
    """
    def __init__(
        self, 
        learning_rate: float, 
        threshold: float,
        *args, 
        **kwargs
    ) -> None:
        """
        Initializes a new Trajectory Constrained Q Learning agent (V1)

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
        safe_actions = list(range(self.n_actions))
        q1 = self.get_Q_values(s1)
        j1 = self.J[s1[0]]

        for action in range(self.n_actions):
            if j1[action] + self.cost_since_last_episode + c > self.task_threshold:
                # Temporarily set unsafe action q value to negative inf
                safe_actions.remove(action)
                q1[action] = -10

        # If no safe actions for next state under current estimation, ignore safety
        if not safe_actions:
            a1 = np.argmax(self.get_Q_values(s1))
        else:
            a1 = np.argmax(q1)
        
        target_q = r + gamma * q1[a1]
        self.Q[s][a] += self.lr * (target_q - self.Q[s][a])

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
            if self.J[self.s[0]][action] + self.cost_since_last_episode > self.task_threshold:
                safe_actions.remove(action)
                # Temporarily set unsafe action q value to negative value
                q[action] = -10

        # If no safe actions from current estimation, ignore safety
        if not safe_actions:
            a = random.choice(list(range(self.n_actions)))
        else:
            # Sample from a Bernoulli distribution with epsilon
            if random.random() <= self.epsilon:
                a = random.choice(safe_actions)
            else:
                a = np.argmax(q)

        # Decrease epsilon gradually
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        return a

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
        self.J = defaultdict(
            lambda: np.zeros(self.n_actions)
        )

        costs = list(self.active_task.obstacles.values())
        avg_cost = 0.5 * (max(costs) + min(costs))
        self.task_threshold = avg_cost * self.threshold * len(self.active_task.obstacle_ids)

    def update_hist(self) -> None:
        """
        Updates the agent performance stats
        """
        super().update_hist()
        if self.steps % self.logging_freq == 0:
            self.interval_violate_hist.append(self.interval_violate_count)
            self.interval_violate_count = 0
        self.violation_hist.append(self.violation_count)

    def compute_episode_stat(self) -> None:
        """
        Compute the performance stats for the previous episode
        """
        super().compute_episode_stat()
        if self.cost_since_last_episode > self.task_threshold:
            self.interval_violate_count += 1

    def check_episode_termination(self) -> bool:
        """
        Check if the episode termination condition is met

        Returns:
            bool: Whether episode should be terminated
        """
        step_violation = self.steps_since_last_episode >= self.T
        cost_violation = self.cost_since_last_episode > self.task_threshold
        if cost_violation:
            self.violation_count += 1
        return step_violation or cost_violation
        
    def state_transition(self, a: int) -> tuple:
        """
        Returns the transitioned state

        Args:
            a (int): Current action

        Returns:
            tuple: The new state, immediate reward and cost, and if terminal state is reached
        """
        s1, r, c, terminal = self.active_task.transition(a)
        self.J[self.s[0]][a] = c

        return s1, r, c, terminal