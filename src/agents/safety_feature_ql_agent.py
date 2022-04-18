"""
Safety Feature Q Learning agent class
"""
import numpy as np

from src.agents.agent import Agent
from src.features.tabular_safety_features import TabularSafetyFeatures


class SafetyFeatureQL(Agent):
    """
    Safety Feature Q Learning agent class
    """
    def __init__(
        self, 
        lookup_table: TabularSafetyFeatures, 
        threshold: float,
        *args, 
        use_gpi=True, 
        **kwargs
    ) -> None:
        """
        Initializes a new tabular safety feature agent.
        
        Args:
            lookup_table (TabularSafetyFeatures): Tabular safety features 
            threshold (float): Trajectory cost threshold
            use_gpi (bool): Whether or not to use transfer learning, defaults to True
        """
        super().__init__(*args, **kwargs)
        self.sf = lookup_table
        self.threshold = threshold
        self.use_gpi = use_gpi
        
    def get_Q_values(self, s: tuple) -> np.ndarray:
        """
        Returns the action value function evaluated in the specified state.
        
        Args:
            s (tuple): Current state

        Returns:
            np.ndarray : The estimated Q-values of the current state
        """
        q, j = self.sf.GPI(s, self.task_index)
        safe_q = self.get_safe_Q_values(q, j)
        self.source_t = np.argmax(np.max(safe_q, axis=1))
        if not self.use_gpi:
            self.source_t = self.task_index
        return q[self.source_t,:]

    def get_safe_Q_values(self, q: np.ndarray, j: np.ndarray, c: float = 0.0) -> np.ndarray:
        """
        Returns Q values where unsafe actions are set to negative Q values

        Args:
            q (np.ndarray): Q value
            j (np.ndarray): J value
            c (float, optional): current cost. Defaults to 0.0.

        Returns:
            np.ndarray: Q values where unsafe actions are set to negative Q values
        """
        j = j + self.cost_since_last_episode + c
        safe_transitions = np.where(j <= self.task_threshold, 1, -10)
        safe_q = safe_transitions * q
        return safe_q
    
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
        # Update w
        phi = self.phi(s, a, s1)
        phi_c = self.phi_c(s, a, s1)
        self.sf.update_reward(phi, r, self.task_index)
        self.sf.update_cost(phi_c, c, self.task_index)
        
        # Update SF for the current task t
        if self.use_gpi:
            q1, j1 = self.sf.GPI(s1, self.task_index)
            safe_q1 = self.get_safe_Q_values(q1, j1, c)
            source_task_1 = np.argmax(np.max(safe_q1, axis=1))
            j1 = j1[source_task_1]
            next_action = np.argmax(np.max(safe_q1, axis=0))
        else:
            q1, j1 = self.sf.GPE(s1, self.task_index, self.task_index)
            safe_q1 = self.get_safe_Q_values(q1, j1, c)
            next_action = np.argmax(safe_q1)
        self.sf.update_successor(s, a, phi, s1, next_action, gamma, self.task_index)
        self.sf.update_safety(s, a, phi_c, s1, next_action, gamma, self.task_index)
        
        # Update SF for source task c
        if self.source_t != self.task_index:
            q1, j1 = self.sf.GPE(s1, self.source_t, self.source_t)
            safe_q1 = self.get_safe_Q_values(q1, j1, c)
            next_action = np.argmax(safe_q1)
            self.sf.update_successor(s, a, phi, s1, next_action, gamma, self.source_t)
            self.sf.update_safety(s, a, phi_c, s1, next_action, gamma, self.source_t)
    
    def initialize(self) -> None:
        """
        Initializes or resets task specific agent params
        """
        super(SafetyFeatureQL, self).initialize()
        self.sf.initialize()
        self.interval_violate_hist = []
        self.violation_hist = []
        
    def add_training_task(self, task) -> None:
        """
        Adds a training task 
        """
        super(SafetyFeatureQL, self).add_training_task(task)
        self.sf.add_training_task(task, -1)

    def set_active_training_task(self) -> None:
        """
        Sets the new task as active task
        """
        self.task_index = self.n_tasks - 1
        super().set_active_training_task()
        self.interval_violate_count = 0
        self.violation_count = 0

        costs = list(self.active_task.obstacles.values())
        avg_cost = 0.5 * (max(costs) + min(costs))
        self.task_threshold = avg_cost * self.threshold * len(self.active_task.obstacle_ids)

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

    def update_hist(self) -> None:
        """
        Updates the agent performance stats
        """
        super().update_hist()
        if self.steps % self.logging_freq == 0:
            self.interval_violate_hist.append(self.interval_violate_count)
            self.interval_violate_count = 0
        self.violation_hist.append(self.violation_count)
    
    def get_progress_strings(self) -> tuple:
        """
        Returns the agent's learning progress 

        Returns:
            tuple[str, str]: agent's learaning progress
        """
        sample_str, reward_str = super(SafetyFeatureQL, self).get_progress_strings()
        w_error = np.linalg.norm(self.sf.fit_w[self.task_index] - self.sf.true_w[self.task_index])
        w_c_error = np.linalg.norm(self.sf.fit_w_c[self.task_index] - self.sf.true_w_c[self.task_index])
        gpi_str = 'w_err \t {:.4f} \t w_c_err \t {:.4f}'.format(w_error, w_c_error)
        return sample_str, reward_str, gpi_str