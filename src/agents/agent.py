"""
Parent class for all RL agents
"""
from typing import Callable
import random
from abc import ABC, abstractmethod
from logging import getLogger

import numpy as np


class Agent(ABC):
    """
    Parent class for all RL agents
    """
    def __init__(
        self,
        gamma: float,
        T: int,
        logging_freq: int,
        update_freq: int,
        epsilon: float = 0.1,
        epsilon_decay: float = 1.0,
        epsilon_min: float = 0.0, 
    ) -> None:
        """
        Initializes an abstract reinforcement learning agent.

        Args:
            gamma (float): Discount factor
            T (int): Max steps per episode
            logging_freq (int): Logging frequency
            update_freq (int): Statistics update frequency
            epsilon (float, optional): Initial exploration parameter for epsilon greedy, defaults to 0.1
            epsilon_decay (float, optional): Annealing amount per step, defaults to 1
            epsilon_min (float, optional): The minimum allowed value of epsilon, defaults to 0
        """
        self.gamma = gamma
        self.T = T
        self.logging_freq = logging_freq
        self.update_freq = update_freq
        self.epsilon_init = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.logger = getLogger("root")

    @abstractmethod
    def get_Q_values(self, s: tuple) -> np.ndarray:
        """
        Returns the action value function evaluated in the specified state.
        
        Args:
            s (tuple): Current state

        Returns:
            np.ndarray : The estimated Q-values of the current state
        """
        pass

    @abstractmethod
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
        pass

    def initialize(self) -> None:
        """
        Initializes or resets task specific agent params
        """
        self.tasks = []
        self.phis = []
        self.phi_cs = []

        self.cumulative_reward = 0
        self.cumulative_cost = 0
        self.cumulative_episode = 0

        self.reward_hist = []
        self.cost_hist = []
        self.reward_collection_hist = []
        
    def add_training_task(self, task) -> None:
        """
        Adds a training task 
        """
        self.tasks.append(task)
        self.n_tasks = len(self.tasks)
        self.phis.append(task.features)
        self.phi_cs.append(task.cost_features)
        if self.n_tasks == 1:
            self.n_actions = task.action_count()
            self.n_features = task.feature_dim()

    def set_active_training_task(self) -> None:
        """
        Sets the new task as active task
        """
        self.active_task = self.tasks[-1] 
        self.phi = self.phis[-1] 
        self.phi_c = self.phi_cs[-1]

        # Initialize task-dependent counters
        self.s = None
        self.new_episode = True
        self.episode = 0
        self.prev_episode_reward = 0
        self.prev_episode_cost = 0
        self.steps_since_last_episode = 0
        self.reward_since_last_episode = 0
        self.cost_since_last_episode = 0
        self.steps = 0
        self.reward = 0
        self.cost = 0
        self.rewards_collected_since_last_episode = 0
        self.reward_collection_percentage = 0
        self.epsilon = self.epsilon_init

    def _epsilon_greedy(self, q: np.ndarray) -> int:
        """
        Epsilon greedy policy

        Args:
            q (np.ndarray): Current state Q values

        Returns:
            int: Action picked by epsilon greedy
        """
        # Sample from a Bernoulli distribution with epsilon
        if random.random() <= self.epsilon:
            a = random.randrange(self.n_actions)
        else:
            a = np.argmax(q)

        # Decrease epsilon gradually
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        return a

    def get_progress_strings(self) -> tuple:
        """
        Returns the agent's learning progress 

        Returns:
            tuple[str, str]: agent's learaning progress
        """
        sample_str = (
            "task \t {} \t steps \t {} \t episodes \t {} \t eps \t {:.4f}".format(
                self.tasks.index(self.active_task), self.steps, self.episode, self.epsilon
            )
        )
        reward_cost_str = "ep_reward \t {:.4f} \t reward \t {:.4f} ep_cost \t {:.4f} \t cost \t {:.4f}".format(
            self.prev_episode_reward, self.reward, self.prev_episode_cost, self.cost
        )
        return sample_str, reward_cost_str

    def pick_action(self) -> int:
        """
        Returns action to take based on exploration policy

        Returns:
            int: Picked action
        """
        q = self.get_Q_values(self.s)
        a = self._epsilon_greedy(q)
        return a

    def state_transition(self, a: int) -> tuple:
        """
        Returns the transitioned state

        Args:
            a (int): Current action

        Returns:
            tuple: The new state, immediate reward and cost, and if terminal state is reached
        """
        return self.active_task.transition(a)

    def compute_episode_stat(self) -> None:
        """
        Compute the performance stats for the previous episode
        """
        self.reward_collection_percentage = self.rewards_collected_since_last_episode / len(self.active_task.reward_ids)

    def check_episode_termination(self) -> bool:
        """
        Check if the episode termination condition is met

        Returns:
            bool: Whether episode should be terminated
        """
        return self.steps_since_last_episode >= self.T

    def update_hist(self) -> None:
        """
        Updates the agent performance stats
        """
        self.reward_hist.append(self.reward)
        self.cost_hist.append(self.cost)
        self.reward_collection_hist.append(self.reward_collection_percentage)

    def next_sample(self) -> None:
        """
        Updates the agent by performing one interaction with the current training environment.
        This function performs all interactions with the environment, data and storage manipulations,
        training the agent, and updating all history.
        """
        if self.new_episode:
            if self.episode > 0:
                self.compute_episode_stat()

            self.s = self.active_task.initialize()
            self.new_episode = False
            self.episode += 1
            self.steps_since_last_episode = 0
            self.prev_episode_reward = self.reward_since_last_episode
            self.prev_episode_cost = self.cost_since_last_episode
            self.reward_since_last_episode = 0
            self.cost_since_last_episode = 0
            self.rewards_collected_since_last_episode = 0

        a = self.pick_action()

        s1, r, c, terminal = self.state_transition(a)

        if r and not terminal:
            self.rewards_collected_since_last_episode +=1

        if terminal:
            gamma = 0
            self.new_episode = True
        else:
            gamma = self.gamma

        self.train_agent(self.s, a, r, c, s1, gamma)

        self.s = s1
        self.steps += 1
        self.reward += r
        self.cost += c
        self.steps_since_last_episode += 1
        self.reward_since_last_episode += r
        self.cost_since_last_episode += c
        self.cumulative_reward += r
        self.cumulative_cost += c

        if not self.new_episode:
            self.new_episode = self.check_episode_termination()

        if self.steps % self.update_freq == 0:
            self.update_hist()

        if self.steps % self.logging_freq == 0:
            self.logger.info("\t".join(self.get_progress_strings()))

    def train_on_task(self, train_task, n_samples: int) -> None:
        """
        Trains the agent on the current task.

        Args
        ----------
        train_task (Task object): The training task instance
        n_samples (int): The number of samples used to train the agent
        """
        self.add_training_task(train_task)
        self.set_active_training_task()
        for _ in range(n_samples):
            self.next_sample()
        
