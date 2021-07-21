import random
import numpy as np
from abc import ABC, abstractmethod


class Agent(ABC):
    def __init__(
        self,
        gamma,
        T,
        encoding=lambda s: s,
        epsilon=0.1,
        epsilon_decay=1.0,
        epsilon_min=0.0,
        print_ev=1000,
        save_ev=100,
    ):
        """
        Creates a new abstract reinforcement learning agent.

        Parameters
        ----------
        gamma : float
            the discount factor in [0, 1]
        T : integer
            the maximum length of an episode
        encoding : function
            encodes the state of the task instance into a numpy array
        epsilon : float
            the initial exploration parameter for epsilon greedy (defaults to 0.1)
        epsilon_decay : float
            the amount to anneal epsilon in each time step (defaults to 1, no annealing)
        epsilon_min : float
            the minimum allowed value of epsilon (defaults to 0)
        print_ev : integer
            how often to print learning progress
        save_ev :
            how often to save learning progress to internal memory
        """
        self.gamma = gamma
        self.T = T
        self.encoding = encoding 
        self.epsilon_init = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.print_ev = print_ev
        self.save_ev = save_ev

    @abstractmethod
    def get_Q_values(self, s, s_enc):
        """
        Returns the value function evaluated in the specified state.
        An array of size [n_batch, n_actions], where:
            n_batch is the number of states provided
            n_actions is the number of possible actions in the current task

        Parameters
        ----------
        s : iterable of object
            raw states of the task
        s_enc : np.ndarray
            collection of encoded states of the shape [n_batch, -1]

        Returns
        -------
        np.ndarray : array of the shape [n_batch, n_actions] returning the estimated
        Q-values of the current instance
        """
        pass

    @abstractmethod
    def train_agent(self, s, s_enc, a, r, c, s1, s1_enc, gamma):
        """
        Trains the current agent on the provided transition.

        Parameters
        ----------
        s : object
            the raw state of the task
        s_enc : np.ndarray
            the encoded state of the task
        a : integer
            the action taken in state s
        r : float
            the reward obtained in the current transition
        c: float
            the cost occurred in the current transition
        s1 : object
            the raw successor state of the task
        s1_enc : np.ndarray
            the encoded next state s1 of the task
        gamma : float
            discount factor to apply to the transition - should be zero if a terminal transition
        """
        pass

    # ===========================================================================
    # TASK MANAGEMENT
    # ===========================================================================
    def reset(self):
        """
        Resets the agent, including all value functions and internal memory/history.
        """
        self.tasks = []
        self.phis = []

        # reset counter history
        self.cumulative_reward = 0
        self.cumulative_cost = 0
        # maybe set a cost history as well?
        self.reward_hist = []
        self.cumulative_reward_hist = []
        self.cost_hist = []
        self.cumulative_cost_hist = []

    def add_training_task(self, task):
        """
        Adds a training task to be trained by the agent.
        """
        self.tasks.append(task)
        self.n_tasks = len(self.tasks)
        self.phis.append(task.features)
        if self.n_tasks == 1:
            self.n_actions = task.action_count()
            self.n_features = task.feature_dim()
            if self.encoding == "task":
                self.encoding = task.encode

    def set_active_training_task(self):
        """
        Sets the task at the requested index as the current task the agent will train on.
        The index is based on the order in which the training task was added to the agent.
        """

        # set the new task as active task
        self.active_task = self.tasks[-1] 
        self.phi = self.phis[-1] 

        # reset task-dependent counters
        self.s = None
        self.s_enc = None
        self.new_episode = True
        self.episode = 0
        self.episode_reward = 0
        self.episode_cost = 0
        self.steps_since_last_episode = 0
        self.reward_since_last_episode = 0
        self.cost_since_last_episode = 0
        self.steps = 0
        self.reward = 0
        self.cost = 0
        self.epsilon = self.epsilon_init
        self.episode_reward_hist = []
        self.episode_cost_hist = []

    # ===========================================================================
    # TRAINING
    # ===========================================================================
    def _epsilon_greedy(self, q):
        """
        Epsilon greedy policy
        """
        assert q.size == self.n_actions

        # sample from a Bernoulli distribution with parameter epsilon
        if random.random() <= self.epsilon:
            a = random.randrange(self.n_actions)
        else:
            a = np.argmax(q)

        # decrease the exploration gradually
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        return a

    def get_progress_strings(self):
        """
        Returns a string that displays the agent's learning progress. This includes things like
        the current training task index, steps and episodes of training, exploration parameter,
        the previous episode reward obtained and cumulative reward, and other information
        depending on the current implementation.
        """
        sample_str = (
            "task \t {} \t steps \t {} \t episodes \t {} \t eps \t {:.4f}".format(
                self.tasks.index(self.active_task), self.steps, self.episode, self.epsilon
            )
        )
        reward_cost_str = "ep_reward \t {:.4f} \t reward \t {:.4f} ep_cost \t {:.4f} \t cost \t {:.4f}".format(
            self.episode_reward, self.reward, self.episode_cost, self.cost
        )
        return sample_str, reward_cost_str

    def pick_action(self):
        """
        Returns action to take based on exploration policy
        """
        # compute the Q-values in the current state
        q = self.get_Q_values(self.s, self.s_enc)

        # choose an action using the epsilon-greedy policy
        a = self._epsilon_greedy(q)
        return a

    def state_transition(self, a):
        """
        Perform state transition taking action from current state 
        """
        return self.active_task.transition(a)

    def next_sample(self, viewer=None, n_view_ev=None):
        """
        Updates the agent by performing one interaction with the current training environment.
        This function performs all interactions with the environment, data and storage manipulations,
        training the agent, and updating all history.

        Parameters
        ----------
        viewer : object
            a viewer that displays the agent's exploration behavior on the task based on its update() method
            (defaults to None)
        n_view_ev : integer
            how often (in training episodes) to invoke the viewer to display agent's learned behavior
            (defaults to None)
        """

        # start a new episode
        if self.new_episode:
            self.s = self.active_task.initialize()
            self.s_enc = self.encoding(self.s)
            self.new_episode = False
            self.episode += 1
            self.steps_since_last_episode = 0
            self.episode_reward = self.reward_since_last_episode
            self.episode_cost = self.cost_since_last_episode
            self.reward_since_last_episode = 0
            self.cost_since_last_episode = 0
            if self.episode > 1:
                self.episode_reward_hist.append(self.episode_reward)
                self.episode_cost_hist.append(self.episode_cost)

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
        self.cost_since_last_episode += c
        self.cumulative_reward += r
        self.cumulative_cost += c

        if self.steps_since_last_episode >= self.T:
            self.new_episode = True

        if self.steps % self.save_ev == 0:
            self.reward_hist.append(self.reward)
            self.cumulative_reward_hist.append(self.cumulative_reward)
            self.cost_hist.append(self.cost)
            self.cumulative_cost_hist.append(self.cumulative_cost)

        # viewing
        if viewer and self.episode % n_view_ev == 0:
            viewer.update()

        # printing
        if self.steps % self.print_ev == 0:
            print("\t".join(self.get_progress_strings()))

    def train_on_task(self, train_task, n_samples, viewer=None, n_view_ev=None):
        """
        Trains the agent on the current task.

        Parameters
        ----------
        train_task : Task
            the training task instance
        n_samples : integer
            how many samples should be generated and used to train the agent
        viewer : object
            a viewer that displays the agent's exploration behavior on the task based on its update() method
            (defaults to None)
        n_view_ev : integer
            how often (in training episodes) to invoke the viewer to display agent's learned behavior
            (defaults to None)
        """
        self.add_training_task(train_task)
        self.set_active_training_task()
        for _ in range(n_samples):
            self.next_sample(viewer, n_view_ev)
