from collections import defaultdict
import numpy as np
import random
from copy import deepcopy

from agents.agent import Agent
from util import LOGGER


class ConstrainedQL(Agent):
    def __init__(self, learning_rate, threshold, *args, **kwargs):
        """
        Creates a new tabular Q-learning agent.

        Parameters
        ----------
        learning_rate : float
            the learning rate to use in order to update Q-values
        """
        super(ConstrainedQL, self).__init__(*args, **kwargs)
        self.alpha = learning_rate
        self.threshold = threshold

    def get_Q_values(self, s, s_enc):
        return self.Q[s]

    def train_agent(self, s, s_enc, a, r, c, s1, s1_enc, gamma):
        # prune unsafe actions
        q1 = {action: value for action, value in enumerate(self.Q[s1])}

        for action in range(self.n_actions):
            if not self.safe_actions[s1[0]][action]:
                # temporarily set unsafe action q value to negative inf
                q1[action] = -float('inf')

        target = r + gamma * np.max(tuple(q1.values()))
        self.Q[s][a] = (1 - self.alpha) * self.Q[s][a] + self.alpha * target

    def set_active_training_task(self):
        super(ConstrainedQL, self).set_active_training_task()
        self.Q = defaultdict(
            lambda: np.random.uniform(low=-0.01, high=0.01, size=(self.n_actions,))
        )
        costs = list(self.active_task.obstacles.values())
        self.task_threshold = (max(costs) - min(costs)) * self.threshold + min(costs)
        self.safe_actions = self.intialize_safe_action_dict()

    def _epsilon_greedy(self, q):
        assert q.size == self.n_actions

        q_dict = {action: value for action, value in enumerate(q)}
            
        # prune unsafe actions
        safe_actions = list(range(self.n_actions))
        s = self.s[0]
        for action in range(self.n_actions):
            if not self.safe_actions[s][action]:
                safe_actions.remove(action)
                # temporarily set unsafe action q value to negative inf
                q_dict[action] = -float('inf')

        # sample from a Bernoulli distribution with parameter epsilon
        if random.random() <= self.epsilon:
            if safe_actions:
                a = random.choice(safe_actions)
            else:
                LOGGER.warning("No state should have 4 unsafe actions, check your maze config or action prune logic!")
                a = random.choice(range(self.n_actions))
        else:
            a = np.argmax(tuple(q_dict.values()))

        # decrease the exploration gradually
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        return a

    def state_transition(self, a):
        s1, r, c, terminal = self.active_task.transition(a)
        if c > self.task_threshold:
            self.safe_actions[self.s[0]][a] = False

        return s1, r, c, terminal

    def intialize_safe_action_dict(self):
        """
        Returns initial safe action matrix, where every state action transition is assumed
        to be safe
        """
        single_state_action_dict = {}
        for a in range(self.n_actions):
            single_state_action_dict[a] = True

        safe_action_dict = {}
        for row in range(self.active_task.height):
            for col in range(self.active_task.width):
                safe_action_dict[(row, col)] = deepcopy(single_state_action_dict)

        return safe_action_dict
        