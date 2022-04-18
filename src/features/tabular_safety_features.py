"""
Tabular Safety Features class
"""
from collections import defaultdict
from copy import deepcopy

import numpy as np

from src.features.tabular_succ_features import TabularSuccessorFeatures

class TabularSafetyFeatures(TabularSuccessorFeatures):
    """
    Tabular Safety Features class
    """
    def __init__(
        self, 
        learning_rate_w: float,
        learning_rate_w_c: float, 
        learning_rate: float,
        learning_rate_c: float, 
        noise_init=lambda size: np.random.uniform(-0.01, 0.01, size=size),
    ) -> None:
        """
        Initializes a new tabular representation of safety features

        Args:
            learning_rate_w (float): Learning rate for reward weights using gradient descent
            learning_rate_w_c (float): Learning rate for cost weights using gradient descent
            learning_rate (float): Learning rate for successor features
            learning_rate_c (float): Learning rate for safety features
            noise_init (Callable, optional): Noise to initialize successor features, defaults to Uniform[-0.01, 0.01].
        """
        super().__init__(learning_rate_w, learning_rate, noise_init)
        self.lr_w_c = learning_rate_w_c
        self.lr_c = learning_rate_c

    def initialize(self) -> None:
        """
        Initializes or resets all trained successor features, learned rewards reward weights,
        and task information.
        """
        super().initialize()
        # shape [n_tasks (policy_index), n_states, n_actions, n_obstacles]
        self.psi_c = []
        self.true_w_c = []
        self.fit_w_c = []

    def add_training_task(self, task, source: int = None) -> None:
        """
        Adds successor features for the specified task.

        Args:
            task (Task object): New MDP environment for which to learn successor features
            source (int, optional): Source task index
        """
        super().add_training_task(task, source)

        self.psi_c.append(self.build_safety(task, source))

        true_w_c = task.get_w_c()
        self.true_w_c.append(true_w_c)
        n_obstacles = task.cost_feature_dim()
        fit_w_c = np.random.uniform(low=-0.01, high=0.01, size=(n_obstacles, 1))
        self.fit_w_c.append(fit_w_c)

    def build_safety(self, task, source: int = None) -> dict:
        """
        Initializes safety features for the specified task. 

        Args:
            task (Task object): New MDP environment for which to learn safety features
            source (int, optional): Source task index
        
        Returns:
            dict : The safety features for the new task 
        """
        if not source or len(self.psi_c) == 0:
            n_actions = task.action_count()
            n_cost_features = task.cost_feature_dim()
            return defaultdict(lambda: self.noise_init((n_actions, n_cost_features)))
        else:
            return deepcopy(self.psi_c[source])

    def update_safety(
        self, 
        state: tuple, 
        action: int, 
        phi_c: float, 
        next_state: tuple, 
        next_action: int, 
        gamma: float, 
        policy_index: int
    )-> None:
        """
        Updates the safety feature on the given transition by linear regression.

        Args:
            state (tuple): Current state
            action (int): Current action
            phi_c (np.ndarray): Cost features
            next_state (tuple): Next state
            next_action (int): Next action
            gamma (float): Discount factor
            policy_index (int): The policy of which the safety feature is from
        """
        psi_c = self.psi_c[policy_index]
        targets = phi_c.flatten() + gamma * psi_c[next_state][next_action, :]
        errors = targets - psi_c[state][action, :]
        self.psi_c[policy_index][state][action, :] += self.lr_c * errors

    def update_cost(self, phi_c: np.ndarray, c: float, task_index: int):
        """
        Updates the cost weights for the given task based on the observed cost sample
        from the environment using linear regression. 
        
        Args:
            phi_c (np.ndarray): Cost features
            c (float): Current cost
            task_index (int): The index of the task from which this cost was sampled
        """
         # update cost using linear regression
        w_c = self.fit_w_c[task_index]
        phi_c = phi_c.reshape(w_c.shape)
        c_fit = np.sum(phi_c * w_c)
        self.fit_w_c[task_index] = w_c + self.lr_w_c * (c - c_fit) * phi_c

    def GPE(self, state: tuple, policy_index: int, task_index: int) -> tuple:
        """
        Implements generalized policy evaluation

        Args:
            state (tuple): Current state
            policy_index (int): The index of the task whose policy to evaluate
            task_index (int): The index of the task to use to evaluate the policy

        Returns:
            tuple[np.ndarray, np.ndarray]: 
                q: the estimated Q-values of shape [number of actions]
                j: the estimated J-values of shape [number of actions]
        """
        # shape (n_actions, n_features)
        psi = self.psi[policy_index][state]
        # shape (n_actions)
        q = psi @ self.fit_w[task_index] 

        # shape (n_actions, n_obstacles)
        psi_c = self.psi_c[policy_index][state]
        # shape (n_actions)
        j = psi_c @ self.fit_w_c[task_index]  
        return q, j

    def GPI(self, state: tuple, task_index: int) -> tuple:
        """
        Implements generalized policy improvement 

        Args:
            state (tuple): Current state
            task_index (int): The index of the task in which the GPI action will be used

        Returns:
            tuple[np.ndarray, np.ndarray]:         
                q: The maximum Q-values computed by GPI for selecting actions
                of shape [number of tasks, number ofactions]
                j: The maximum J-values computed by GPI for selecting actions
                of shape [number of tasks, number ofactions]
        """
        # shape (n_tasks, n_actions, n_features)
        psi = np.array([psi[state] for psi in self.psi])
        # shape (n_tasks, n_actions)
        q = (psi @ self.fit_w[task_index])[:, :, 0]

        # shape (n_tasks, n_actions, n_obstacles)
        psi_c = np.array([psi_c[state] for psi_c in self.psi_c])
        # shape (n_tasks, n_actions)
        j = (psi_c @ self.fit_w_c[task_index])[:, :, 0]  

        return q, j