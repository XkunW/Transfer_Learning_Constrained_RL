import numpy as np
from collections import defaultdict
from agents.agent import Agent


class ConstrainedSFQL(Agent):
    def __init__(self, lookup_table, *args, use_gpi=True, **kwargs):
        """
        Creates a new tabular successor feature agent.

        Parameters
        ----------
        lookup_table : TabularSF
            a tabular successor feature representation
        use_gpi : boolean
            whether or not to use transfer learning (defaults to True)
        """
        super(ConstrainedSFQL, self).__init__(*args, **kwargs)
        self.sf = lookup_table
        self.use_gpi = use_gpi

    def get_Q_values(self, s, s_enc):
        q, self.c = self.sf.GPI(s_enc, self.task_index, update_counters=self.use_gpi)
        if not self.use_gpi:
            self.c = self.task_index
        return q[:, self.c, :]

    def train_agent(self, s, s_enc, a, r, c, s1, s1_enc, gamma):

        # update w
        t = self.task_index
        phi = self.phi(s, a, s1)
        self.sf.update_cost(phi, c, t)

        # update SF for the current task t
        if self.use_gpi:
            q1, _ = self.sf.GPI(s1_enc, t)
            q1 = np.max(q1[0, :, :], axis=0)
        else:
            q1 = self.sf.GPE(s1_enc, t, t)[0, :]
        next_action = np.argmax(q1)
        transitions = [(s_enc, a, phi, s1_enc, next_action, gamma)]
        self.sf.update_successor(transitions, t)

        # update SF for source task c
        if self.c != t:
            q1 = self.sf.GPE(s1_enc, self.c, self.c)
            next_action = np.argmax(q1)
            transitions = [(s_enc, a, phi, s1_enc, next_action, gamma)]
            self.sf.update_successor(transitions, self.c)

    def reset(self):
        super(ConstrainedSFQL, self).reset()
        self.sf.reset()

    def add_training_task(self, task):
        super(ConstrainedSFQL, self).add_training_task(task)
        self.sf.add_training_task(task, -1)

    def set_active_training_task(self, index):
        super(ConstrainedSFQL, self).set_active_training_task(index)
        self.Q = defaultdict(
            lambda: np.random.uniform(low=-0.01, high=0.01, size=(self.n_actions,))
        )

    def get_progress_strings(self):
        sample_str, reward_str = super(SFQL, self).get_progress_strings()
        gpi_percent = self.sf.GPI_usage_percent(self.task_index)
        w_error = np.linalg.norm(
            self.sf.fit_w[self.task_index] - self.sf.true_w[self.task_index]
        )
        gpi_str = "GPI% \t {:.4f} \t w_err \t {:.4f}".format(gpi_percent, w_error)
        return sample_str, reward_str, gpi_str
