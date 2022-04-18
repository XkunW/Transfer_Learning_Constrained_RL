"""
Successor Feature Q Learning agent class
"""
import numpy as np

from src.agents.agent import Agent
from src.features.tabular_succ_features import TabularSuccessorFeatures


class SuccessorFeatureQL(Agent):
    """
    Successor Feature Q Learning agent class
    """
    def __init__(
        self, 
        lookup_table: TabularSuccessorFeatures, 
        *args, 
        use_gpi: bool = True, 
        **kwargs
    ) -> None:
        """
        Initializes a new tabular successor feature agent.
        
        Args:
            lookup_table (TabularSafetyFeatures): Tabular safety features 
            use_gpi (bool): Whether or not to use transfer learning, defaults to True
        """
        super().__init__(*args, **kwargs)
        self.sf = lookup_table
        self.use_gpi = use_gpi
        
    def get_Q_values(self, s: tuple) -> np.ndarray:
        """
        Returns the action value function evaluated in the specified state.
        
        Args:
            s (tuple): Current state

        Returns:
            np.ndarray : The estimated Q-values of the current state
        """
        q, self.source_t = self.sf.GPI(s, self.task_index, update_counters=self.use_gpi)
        if not self.use_gpi:
            self.source_t = self.task_index
        return q[self.source_t,:]
    
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
        t = self.task_index
        phi = self.phi(s, a, s1)
        self.sf.update_reward(phi, r, t)
        
        # Update SF for the current task t
        if self.use_gpi:
            q1, _ = self.sf.GPI(s1, t)
            q1 = np.max(q1, axis=0)
        else:
            q1 = self.sf.GPE(s1, t, t)
        next_action = np.argmax(q1)
        self.sf.update_successor(s, a, phi, s1, next_action, gamma, t)
        
        # Update SF for source task c
        if self.source_t != t:
            q1 = self.sf.GPE(s1, self.source_t, self.source_t)
            next_action = np.argmax(q1)
            self.sf.update_successor(s, a, phi, s1, next_action, gamma, self.source_t)
    
    def initialize(self) -> None:
        """
        Initializes or resets task specific agent params
        """
        super().initialize()
        self.sf.initialize()
        
    def add_training_task(self, task) -> None:
        """
        Adds a training task 
        """
        super().add_training_task(task)
        self.sf.add_training_task(task, -1)

    def set_active_training_task(self) -> None:
        """
        Sets the new task as active task
        """
        self.task_index = self.n_tasks - 1
        super().set_active_training_task()
    
    def get_progress_strings(self) -> tuple:
        """
        Returns the agent's learning progress 

        Returns:
            tuple[str, str]: agent's learaning progress
        """
        sample_str, reward_str = super().get_progress_strings()
        gpi_percent = self.sf.gpi_usage_percent(self.task_index)
        w_error = np.linalg.norm(self.sf.fit_w[self.task_index] - self.sf.true_w[self.task_index])
        gpi_str = 'GPI% \t {:.4f} \t w_err \t {:.4f}'.format(gpi_percent, w_error)
        return sample_str, reward_str, gpi_str