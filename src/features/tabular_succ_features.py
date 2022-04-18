"""
Tabular Successor Features class
"""
from collections import defaultdict
from copy import deepcopy

import numpy as np


class TabularSuccessorFeatures:
    """
    Tabular Successor Features class
    """
    def __init__(
        self,
        learning_rate_w: float,
        learning_rate: float,
        noise_init=lambda size: np.random.uniform(-0.01, 0.01, size=size),
    ) -> None:
        """
        Initializes a new tabular representation of successor features

        Args:
            learning_rate_w (float): Learning rate for reward weights using gradient descent
            learning_rate (float): Learning rate for successor features
            noise_init (Callable, optional): Noise to initialize successor features, defaults to Uniform[-0.01, 0.01]
        """
        self.lr_w = learning_rate_w
        self.lr = learning_rate
        self.noise_init = noise_init

    def initialize(self) -> None:
        """
        Initializes or resets all trained successor features, learned rewards reward weights,
        and task information.
        """
        self.n_tasks = 0
        # shape [n_tasks (policy_index), n_states, n_actions, n_features]
        self.psi = []
        self.true_w = []
        self.fit_w = []
        self.gpi_counters = []

    def add_training_task(self, task, source: int = None) -> None:
        """
        Adds successor features for the specified task.

        Args:
            task (Task object): New MDP environment for which to learn successor features
            source (int, optional): Source task index
        """
        self.psi.append(self.build_successor(task, source))
        self.n_tasks = len(self.psi)

        # Build new reward function
        true_w = task.get_w()
        self.true_w.append(true_w)
        n_features = task.feature_dim()
        fit_w = np.random.uniform(low=-0.01, high=0.01, size=(n_features, 1))
        self.fit_w.append(fit_w)

        for i in range(len(self.gpi_counters)):
            self.gpi_counters[i] = np.append(self.gpi_counters[i], 0)
        self.gpi_counters.append(np.zeros((self.n_tasks,), dtype=int))

    def build_successor(self, task, source: int = None) -> dict:
        """
        Initializes successor features for the specified task. 

        Args:
            task (Task object): New MDP environment for which to learn successor features
            source (int, optional): Source task index
        
        Returns:
            dict : The successor features for the new task 
        """
        if not source or len(self.psi) == 0:
            n_actions = task.action_count()
            n_features = task.feature_dim()
            return defaultdict(lambda: self.noise_init((n_actions, n_features)))
        else:
            return deepcopy(self.psi[source])

    def update_successor(
        self, 
        state: tuple, 
        action: int, 
        phi: np.ndarray, 
        next_state: tuple, 
        next_action: int, 
        gamma: float, 
        policy_index: int
    ) -> None:
        """
        Updates the successor feature on the given transition by linear regression.

        Args:
            state (tuple): Current state
            action (int): Current action
            phi (np.ndarray): Reward features
            next_state (tuple): Next state
            next_action (int): Next action
            gamma (float): Discount factor
            policy_index (int): The policy of which the successor feature is from
        """
        psi = self.psi[policy_index]
        targets = phi.flatten() + gamma * psi[next_state][next_action, :]
        errors = targets - psi[state][action, :]
        self.psi[policy_index][state][action, :] += self.lr * errors

    def update_reward(self, phi: np.ndarray, r: float, task_index: int) -> None:
        """
        Updates the reward weights for the given task based on the observed reward sample
        from the environment using linear regression. 
        
        Args:
            phi (np.ndarray): Reward features
            r (float): Current reward
            task_index (int): The index of the task from which this reward was sampled
        """
        w = self.fit_w[task_index]
        phi = phi.reshape(w.shape)
        r_fit = np.sum(phi * w)
        self.fit_w[task_index] = w + self.lr_w * (r - r_fit) * phi
    
    def GPE(self, state: tuple, policy_index: int, task_index: int) -> np.ndarray:
        """
        Implements generalized policy evaluation, which uses the learned reward weights of 
        one task and successor features of a policy to estimate the Q-values of the policy 
        if it were executed on that task.

        Args:
            state (tuple): Current state
            policy_index (int): The index of the task whose policy to evaluate
            task_index (int): The index of the task to use to evaluate the policy

        Returns:
            np.ndarray : the estimated Q-values of shape [number of actions]
        """
        # shape (n_actions, n_features)
        psi = self.psi[policy_index][state]
        # shape (n_actions)
        q = psi @ self.fit_w[task_index]  
        return q

    def GPI(self, state: tuple, task_index: int, update_counters: bool = False) -> tuple:
        """
        Implements generalized policy improvement 

        Args:
            state (tuple): Current state
            task_index (int): The index of the task in which the GPI action will be used
            update_counters (bool): Whether to keep track of which policies are active in GPI

        Returns:
            tuple[np.ndarray, np.ndarray]:         
                q: The maximum Q-values computed by GPI for selecting actions
                of shape [number of tasks, number ofactions]
                task: the tasks that are active in GPI
        """
        # shape (n_tasks, n_actions, n_features)
        psi = np.array([psi[state] for psi in self.psi])
        # shape (n_tasks, n_actions)
        q = (psi @ self.fit_w[task_index])[:, :, 0]  
        task = np.squeeze(np.argmax(np.max(q, axis=1), axis=0))  

        if update_counters:
            self.gpi_counters[task_index][task] += 1
        return q, task

    def gpi_usage_percent(self, task_index: int) -> float:
        """
        Counts the number of times that actions were transferred from other tasks.

        Args:
            task_index (int): The index of the task

        Returns:
            float: the (normalized) number of actions that were transferred from other
                tasks in GPI.
        """
        counts = self.gpi_counters[task_index]
        return 1 - (counts[task_index] / np.sum(counts))
