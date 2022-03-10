import numpy as np

from agents.agent import Agent


class SFQL(Agent):
    
    def __init__(
        self, 
        lookup_table, 
        epsilon=0.1,
        epsilon_decay=1.0,
        epsilon_min=0.0,
        *args, 
        use_gpi=True, 
        **kwargs
    ):
        """
        Creates a new tabular successor feature agent.
        
        Parameters
        ----------
        epsilon : float
            the initial exploration parameter for epsilon greedy (defaults to 0.1)
        epsilon_decay : float
            the amount to anneal epsilon in each time step (defaults to 1, no annealing)
        epsilon_min : float
            the minimum allowed value of epsilon (defaults to 0)
        lookup_table : TabularSF
            a tabular successor feature representation
        use_gpi : boolean
            whether or not to use transfer learning (defaults to True)
        """
        super().__init__(*args, **kwargs)
        self.sf = lookup_table
        self.epsilon_init = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.use_gpi = use_gpi
        
    def get_Q_values(self, s, s_enc):
        q, self.source_t = self.sf.GPI(s_enc, self.task_index, update_counters=self.use_gpi)
        if not self.use_gpi:
            self.source_t = self.task_index
        return q[:, self.source_t,:]
    
    def train_agent(self, s, s_enc, a, r, c, s1, s1_enc, gamma):
        
        # update w
        t = self.task_index
        phi = self.phi(s, a, s1)
        self.sf.update_reward(phi, r, t)
        
        # update SF for the current task t
        if self.use_gpi:
            q1 = self.sf.GPI(s1_enc, t)[0]
            q1 = np.max(q1[0,:,:], axis=0)
        else:
            q1 = self.sf.GPE(s1_enc, t, t)[0,:]
        next_action = np.argmax(q1)
        transitions = [(s_enc, a, phi, s1_enc, next_action, gamma)]
        self.sf.update_successor(transitions, t)
        
        # update SF for source task c
        if self.source_t != t:
            q1 = self.sf.GPE(s1_enc, self.source_t, self.source_t)
            next_action = np.argmax(q1)
            transitions = [(s_enc, a, phi, s1_enc, next_action, gamma)]
            self.sf.update_successor(transitions, self.source_t)
    
    def reset(self):
        super(SFQL, self).reset()
        self.sf.reset()
        
    def add_training_task(self, task):
        super(SFQL, self).add_training_task(task)
        self.sf.add_training_task(task, -1)

    def set_active_training_task(self):
        self.task_index = self.n_tasks - 1
        super().set_active_training_task()
    
    def get_progress_strings(self):
        sample_str, reward_str = super(SFQL, self).get_progress_strings()
        gpi_percent = self.sf.GPI_usage_percent(self.task_index)
        w_error = np.linalg.norm(self.sf.fit_w[self.task_index] - self.sf.true_w[self.task_index])
        gpi_str = 'GPI% \t {:.4f} \t w_err \t {:.4f}'.format(gpi_percent, w_error)
        return sample_str, reward_str, gpi_str