from collections import defaultdict
from copy import deepcopy
import numpy as np


class TabularSF:
    """
    A successor feature representation implemented using lookup tables. Storage is lazy and implemented efficiently
    using defaultdict.
    """

    def __init__(
        self,
        learning_rate_w,
        use_true_reward,
        learning_rate,
        noise_init=lambda size: np.random.uniform(-0.01, 0.01, size=size),
    ):
        """
        Creates a new tabular representation of successor features.

        Parameters
        ----------
        learning_rate_w : float
            the learning rate to use for learning the reward weights using gradient descent
        use_true_reward : boolean
            whether or not to use the true reward weights from the environment, or learn them
            using gradient descent
        learning_rate : float
            the learning rate
        noise_init : function
            instruction to initialize action-values, defaults to Uniform[-0.01, 0.01]
        """
        self.alpha_w = learning_rate_w
        self.use_true_reward = use_true_reward
        self.alpha = learning_rate
        self.noise_init = noise_init

    def build_successor(self, task, source=None):
        """
        Builds a new successor feature map for the specified task. This method should not be called directly.
        Instead, add_task should be called instead.

        Parameters
        ----------
        task : Task
            a new MDP environment for which to learn successor features
        source : integer
            if specified and not None, the parameters of the successor features for the task at the source
            index should be copied to the new successor features, as suggested in [1]

        Returns
        -------
        object : the successor feature representation for the new task, which can be a Keras model,
        a lookup table (dictionary) or another learning representation
        """
        if not source or len(self.psi) == 0:
            n_actions = task.action_count()
            n_features = task.feature_dim()
            return defaultdict(lambda: self.noise_init((n_actions, n_features)))
        else:
            return deepcopy(self.psi[source])

    def get_successor(self, state, policy_index):
        """
        Evaluates the successor features in given states for the specified task.

        Parameters
        ----------
        state : object
            a state or collection of states of the MDP
        policy_index : integer
            the index of the task whose successor features to evaluate

        Returns
        -------
        np.ndarray : the evaluation of the successor features, which is of shape
        [n_batch, n_actions, n_features], where
            n_batch is the number of states in the state argument
            n_actions is the number of actions of the MDP
            n_features is the number of features in the SF representation
        """
        return np.expand_dims(self.psi[policy_index][state], axis=0)

    def get_successors(self, state):
        """
        Evaluates the successor features in given states for all tasks.

        Parameters
        ----------
        state : object
            a state or collection of states of the MDP

        Returns
        -------
        np.ndarray : the evaluation of the successor features, which is of shape
        [n_batch, n_tasks, n_actions, n_features], where
            n_batch is the number of states in the state argument
            n_tasks is the number of tasks
            n_actions is the number of actions of the MDP
            n_features is the number of features in the SF representation
        """
        return np.expand_dims(np.array([psi[state] for psi in self.psi]), axis=0)

    def update_successor(self, transitions, policy_index):
        """
        Updates the successor representation by training it on the given transition.

        Parameters
        ----------
        transitions : object
            collection of transitions
        policy_index : integer
            the index of the task whose successor features to update
        """
        for state, action, phi, next_state, next_action, gamma in transitions:
            psi = self.psi[policy_index]
            targets = phi.flatten() + gamma * psi[next_state][next_action, :]
            errors = targets - psi[state][action, :]
            psi[state][action, :] = psi[state][action, :] + self.alpha * errors

    def reset(self):
        """
        Removes all trained successor feature representations from the current object, all learned rewards,
        and all task information.
        """
        self.n_tasks = 0
        self.psi = []
        self.true_w = []
        self.fit_w = []
        self.gpi_counters = []

    def add_training_task(self, task, source=None):
        """
        Adds a successor feature representation for the specified task.

        Parameters
        ----------
        task : Task
            a new MDP environment for which to learn successor features
        source : integer
            if specified and not None, the parameters of the successor features for the task at the source
            index should be copied to the new successor features, as suggested in [1]
        """

        # add successor features to the library
        self.psi.append(self.build_successor(task, source))
        self.n_tasks = len(self.psi)

        # build new reward function
        true_w = task.get_w()
        self.true_w.append(true_w)
        if self.use_true_reward:
            fit_w = true_w
        else:
            n_features = task.feature_dim()
            fit_w = np.random.uniform(low=-0.01, high=0.01, size=(n_features, 1))
        self.fit_w.append(fit_w)

        # add statistics
        for i in range(len(self.gpi_counters)):
            self.gpi_counters[i] = np.append(self.gpi_counters[i], 0)
        self.gpi_counters.append(np.zeros((self.n_tasks,), dtype=int))

    def update_reward(self, phi, r, task_index, exact=False):
        """
        Updates the reward parameters for the given task based on the observed reward sample
        from the environment. 
        
        Parameters
        ----------
        phi : np.ndarray
            the state features
        r : float
            the observed reward from the MDP
        task_index : integer
            the index of the task from which this reward was sampled
        exact : boolean
            if True, validates the true reward from the environment and the linear representation
        """
        
        # update reward using linear regression
        w = self.fit_w[task_index]
        phi = phi.reshape(w.shape)
        r_fit = np.sum(phi * w)
        self.fit_w[task_index] = w + self.alpha_w * (r - r_fit) * phi
    
        # validate reward
        r_true = np.sum(phi * self.true_w[task_index])
        if exact and not np.allclose(r, r_true):
            raise Exception('sampled reward {} != linear reward {} - please check task {}!'.format(
                r, r_true, task_index))

    def GPE(self, state, policy_index, task_index):
        """
        Implements generalized policy evaluation according to [1]. In summary, this uses the
        learned reward parameters of one task and successor features of a policy to estimate the Q-values of
        the policy if it were executed in that task.

        Parameters
        ----------
        state : object
            a state or collection of states of the MDP
        policy_index : integer
            the index of the task whose policy to evaluate
        task_index : integer
            the index of the task (e.g. reward) to use to evaluate the policy

        Returns
        -------
        np.ndarray : the estimated Q-values of shpae [n_batch, n_actions], where
            n_batch is the number of states in the state argument
            n_actions is the number of actions in the MDP
        """
        psi = self.get_successor(state, policy_index)
        q = psi @ self.fit_w[task_index]  # shape (n_batch, n_actions)
        return q

    def GPI(self, state, task_index, update_counters=False):
        """
        Implements generalized policy improvement according to [1].

        Parameters
        ----------
        state : object
            a state or collection of states of the MDP
        task_index : integer
            the index of the task in which the GPI action will be used
        update_counters : boolean
            whether or not to keep track of which policies are active in GPI

        Returns
        -------
        np.ndarray : the maximum Q-values computed by GPI for selecting actions
        of shape [n_batch, n_tasks, n_actions], where:
            n_batch is the number of states in the state argument
            n_tasks is the number of tasks
            n_actions is the number of actions in the MDP
        np.ndarray : the tasks that are active in each state of state_batch in GPI
        """
        psi = self.get_successors(state)
        q = (psi @ self.fit_w[task_index])[:, :, :, 0]  # shape (n_batch, n_tasks, n_actions)
        task = np.squeeze(np.argmax(np.max(q, axis=2), axis=1))  # shape (n_batch,)

        if update_counters:
            self.gpi_counters[task_index][task] += 1
        return q, task

    def GPI_usage_percent(self, task_index):
        """
        Counts the number of times that actions were transferred from other tasks.

        Parameters
        ----------
        task_index : integer
            the index of the task

        Returns
        -------
        float : the (normalized) number of actions that were transferred from other
            tasks in GPI.
        """
        counts = self.gpi_counters[task_index]
        return 1 - (counts[task_index] / np.sum(counts))
