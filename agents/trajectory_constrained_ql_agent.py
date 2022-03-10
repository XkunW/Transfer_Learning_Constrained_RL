import sys
import random
from collections import defaultdict

import numpy as np

from agents.agent import Agent
from utils import LOGGER


class TrajectoryConstrainedQL(Agent):
    def __init__(
        self, 
        learning_rate, 
        threshold,
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
        threshold: float
            the one step cost threshold
        epsilon : float
            the initial exploration parameter for epsilon greedy (defaults to 0.1)
        epsilon_decay : float
            the amount to anneal epsilon in each time step (defaults to 1, no annealing)
        epsilon_min : float
            the minimum allowed value of epsilon (defaults to 0)
        """
        super().__init__(*args, **kwargs)
        self.alpha = learning_rate
        self.threshold = threshold
        self.epsilon_init = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def get_Q_values(self, s, s_enc=None):
        return self.Q[s]

    def train_agent(self, s, s_enc, a, r, c, s1, s1_enc, gamma):
        # prune unsafe actions
        safe_actions = list(range(self.n_actions))
        q1 = self.get_Q_values(s1)

        for action in range(self.n_actions):
            if self.J[s1[0]][action] + self.cost_since_last_episode > self.task_threshold:
                # temporarily set unsafe action q value to negative inf
                safe_actions.remove(action)
                q1[action] = -float('inf')
        
        target_q = r + gamma * np.max(q1)
        self.Q[s][a] += self.alpha * (target_q - self.Q[s][a])
        

    def set_active_training_task(self):
        super().set_active_training_task()
        self.Q = defaultdict(
            lambda: np.random.uniform(low=-0.01, high=0.01, size=(self.n_actions,))
        )
        self.J = defaultdict(
            lambda: np.zeros(self.n_actions)
        )

        costs = list(self.active_task.obstacles.values())
        avg_cost = 0.5 * (max(costs) + min(costs))
        self.task_threshold = avg_cost * self.threshold * len(self.active_task.obstacle_ids)

    def _epsilon_greedy(self, q):
        assert q.size == self.n_actions
            
        # prune unsafe actions
        safe_actions = list(range(self.n_actions))
        s = self.s[0]
        
        for action in range(self.n_actions):
            if self.J[s][action] + self.cost_since_last_episode > self.task_threshold:
                safe_actions.remove(action)
                # temporarily set unsafe action q value to negative inf
                q[action] = -float('inf')

        # sample from a Bernoulli distribution with parameter epsilon
        if random.random() <= self.epsilon:
            if not safe_actions:
                error_message = "A state should not have all actions labelled as unsafe, check config or action prune logic!"
                LOGGER.error(error_message)
                sys.exit(1)
            a = random.choice(safe_actions)
        else:
            a = np.argmax(q)

        # decrease the exploration gradually
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        return a
    
    def state_transition(self, a):
        s1, r, c, terminal = self.active_task.transition(a)
        self.J[self.s[0]][a] = c

        return s1, r, c, terminal
        
    def next_sample(self):
        """
        Updates the agent by performing one interaction with the current training environment.
        This function performs all interactions with the environment, data and storage manipulations,
        training the agent, and updating all history.
        """

        # start a new episode
        if self.new_episode:
            self.s = self.active_task.initialize()
            self.s_enc = self.encoding(self.s)
            self.new_episode = False
            self.episode += 1
            self.steps_since_last_episode = 0
            self.prev_episode_reward = self.reward_since_last_episode
            self.prev_episode_cost = self.cost_since_last_episode
            self.reward_since_last_episode = 0
            self.cost_since_last_episode = 0
            if self.episode > 1:
                self.episode_reward_hist.append(self.prev_episode_reward)
                self.episode_cost_hist.append(self.prev_episode_cost)

        # pick an action to take
        a = self.pick_action()

        # take action a and observe reward r and next state s'
        s1, r, c, terminal = self.state_transition(a)
        s1_enc = self.encoding(s1)
        if terminal:
            gamma = 0
            self.new_episode = True
        else:
            gamma = self.gamma

        # update episode cost before training 
        self.cost_since_last_episode += c

        # train the agent
        self.train_agent(self.s, self.s_enc, a, r, c, s1, s1_enc, gamma)

        # update counters
        self.s = s1
        self.s_enc = s1_enc
        self.steps += 1
        self.reward += r
        self.cost += c
        self.steps_since_last_episode += 1
        self.reward_since_last_episode += r
        self.cumulative_reward += r
        self.cumulative_cost += c

        # if current cost exceeds threshold, start a new episode
        if self.steps_since_last_episode >= self.T or self.cost_since_last_episode > self.task_threshold:
            self.new_episode = True

        if self.steps % self.save_ev == 0:
            self.reward_hist.append(self.reward)
            self.cumulative_reward_hist.append(self.cumulative_reward)
            self.cost_hist.append(self.cost)
            self.cumulative_cost_hist.append(self.cumulative_cost)

        # printing
        if self.steps % self.print_ev == 0:
            print("\t".join(self.get_progress_strings()))
