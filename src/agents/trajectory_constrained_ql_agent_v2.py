"""
Trajectory Constrained Reinforcement Learning agent class 2nd version
The safety of an action is determined by whether cumulative_cost + J(s,a) < task threshold
"""
import sys
import random
from collections import defaultdict

import numpy as np

from src.agents.trajectory_constrained_ql_agent_v1 import TrajectoryConstrainedQLV1


class TrajectoryConstrainedQLV2(TrajectoryConstrainedQLV1):
    """
    Trajectory Constrained Q Learning agent class 2nd version
    """
    def __init__(
        self, 
        learning_rate, 
        threshold,
        *args, 
        **kwargs
    ) -> None:
        """
        Initializes a new Trajectory Constrained Q Learning agent (V2)

        Args:
            learning_rate (float): Learning rate for Q-value update
            threshold (float): One step cost threshold
        """
        super().__init__(learning_rate, threshold, *args, **kwargs)

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
                # Temporarily set unsafe action q value to negative value
                safe_actions.remove(action)
                q1[action] = -10

        # If no safe actions for next state under current estimation, ignore safety
        if not safe_actions:
            a1 = np.argmax(self.get_Q_values(s1))
        else:
            a1 = np.argmax(q1)
        
        target_q = r + gamma * q1[a1]
        self.Q[s][a] += self.lr * (target_q - self.Q[s][a])

        target_j = c + gamma * j1[a1]
        self.J[s[0]][a] += self.lr * (target_j - self.J[s[0]][a])

    def state_transition(self, a: int) -> tuple:
        """
        Returns the transitioned state

        Args:
            a (int): Current action

        Returns:
            tuple: The new state, immediate reward and cost, and if terminal state is reached
        """
        return self.active_task.transition(a)
        