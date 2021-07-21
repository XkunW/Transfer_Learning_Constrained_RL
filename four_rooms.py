import numpy as np
import random


class Gridworld:
    """
    A discretized version of the gridworld environment introduced in [1]. Here, an agent learns to
    collect shapes with positive reward, while avoid those with negative reward, and then travel to a fixed goal.
    The gridworld is split into four rooms separated by walls with passage-ways.

    References
    ----------
    [1] Barreto, Andre, et al. "Successor Features for Transfer in Reinforcement Learning." NIPS. 2017.
    """

    actions = {
        0: (0, -1), # LEFT
        1: (-1, 0), # UP
        2: (0, 1), # RIGHT
        3: (1, 0) # DOWN
    } 

    def __init__(self, maze, rewards, obstacles):
        """
        Creates a new instance of the shapes environment.

        Parameters
        ----------
        maze : np.ndarray
            an array of string values representing the type of each cell in the environment:
                G indicates a goal state (terminal state)
                _ indicates an initial state (there can be multiple, and one is selected at random
                    at the start of each episode)
                X indicates a barrier
                0, 1, .... 9 indicates the type of shape to be placed in the corresponding cell
                entries containing other characters are treated as regular empty cells
        rewards : dict
            a dictionary mapping the type of shape (1, 2, ... ) to a corresponding reward to provide
            to the agent for collecting an object of that type
        obstacles: dict
            a dictionary mapping the type of shape (A, B, ... ) to a corresponding obstacle to provide
            to the agent for avoiding an object of that type
        """
        self.height, self.width = maze.shape
        self.maze = maze
        self.rewards = rewards
        self.obstacles = obstacles

        reward_types = sorted(list(rewards.keys()))
        obstacle_types = sorted(list(obstacles.keys()))
        self.rewards_index = dict(zip(reward_types, range(len(reward_types))))
        self.obstacles_index = dict(zip(obstacle_types, range(len(obstacle_types))))

        self.goal = None
        self.initial = []
        self.occupied = set()
        self.reward_ids = dict()
        self.obstacle_ids = dict()
        for c in range(self.width):
            for r in range(self.height):
                if maze[r, c] == "G":
                    self.goal = (r, c)
                elif maze[r, c] == "_":
                    self.initial.append((r, c))
                elif maze[r, c] == "X":
                    self.occupied.add((r, c))
                elif maze[r, c].isnumeric():
                    self.reward_ids[(r, c)] = len(self.reward_ids)
                elif maze[r, c].isalpha():
                    # Since "G" and "X" are already checked, no need to exclude them
                    self.obstacle_ids[(r, c)] = len(self.obstacle_ids)

    def clone(self):
        """
        Creates an identical copy of the current environment, for use in testing.

        Returns
        -------
        Task : the copy of the current task
        """
        return Gridworld(self.maze, self.rewards, self.obstacles)

    def initialize(self):
        """
        Resets the state of the environment.

        Returns
        -------
        object : the initial state of the MDP
        """
        self.state = (
            random.choice(self.initial),
            tuple(False for _ in range(len(self.reward_ids)))
        )
        return self.state

    def action_count(self):
        """
        Returns the number of possible actions in the MDP.

        Returns
        -------
        integer : number of possible actions
        """
        return 4

    def transition(self, action):
        """
        Applies the specified action in the environment, updating the state of the MDP.

        Parameters
        ----------
        action : integer
            the action to apply to the environment

        Returns
        -------
        object : the next state of the MDP
        float : the immediate reward observed in the transition
        float : the immediate cost observed in the transition
        boolean : whether or not a terminal state has been reached
        """
        (row, col), rewards_collected = self.state

        # perform the movement
        if action > 3 or action < 0:
            raise Exception("bad action {}".format(action))

        row += Gridworld.actions[action][0]
        col += Gridworld.actions[action][1]        

        # out of bounds, cannot move
        if col < 0 or col >= self.width or row < 0 or row >= self.height:
            return self.state, 0, 0, False

        # next state
        s_next = (row, col)

        # into a blocked cell, cannot move
        if s_next in self.occupied:
            return self.state, 0, 0, False

        # valid action, move to next cell
        self.state = (s_next, rewards_collected)

        # into a goal cell
        if s_next == self.goal:
            return self.state, 1, 0, True

        # into a reward cell
        if s_next in self.reward_ids:
            reward_id = self.reward_ids[s_next]
            if rewards_collected[reward_id]:
                # already collected this flag
                return self.state, 0, 0, False
            else:
                # collect the new flag
                rewards_collected = list(rewards_collected)
                rewards_collected[reward_id] = True
                self.state = (s_next, tuple(rewards_collected))
                reward = self.rewards[self.maze[row, col]]
                return self.state, reward, 0, False

        # into a obstacle cell
        if s_next in self.obstacle_ids:
            # collect the new flag
            self.state = (s_next, rewards_collected)
            cost = self.obstacles[self.maze[row, col]]
            return self.state, 0, cost, False

        # into an empty cell
        return self.state, 0, 0, False

    # ===========================================================================
    # STATE ENCODING FOR DEEP LEARNING
    # ===========================================================================
    def encode(self, state):
        """
        Encodes the state of the MDP according to its canonical encoding.

        Parameters
        ----------
        state : object
            the state of the MDP to encode

        Returns
        -------
        np.ndarray : the encoding of the state
        """
        (row, col), rewards_collected, _ = state
        n_state = self.width + self.height
        result = np.zeros((n_state + len(self.reward_ids) + len(self.obstacle_ids),))
        result[row] = 1
        result[self.height + col] = 1
        # Convert boolean to int
        result[n_state : -len(self.obstacle_ids)] = np.multiply(
            np.array(rewards_collected), 1
        )
        if (row, col) in self.obstacle_ids:
            obstacle_id = self.obstacle_ids[(row, col)]
            result[n_state + len(self.reward_ids) + obstacle_id] = 1
        result = result.reshape((1, -1))
        return result

    def encode_dim(self):
        """
        Returns the dimension of the canonical state encoding.

        Returns
        -------
        integer : the dimension of the canonical state encoding
        """
        return self.width + self.height + len(self.reward_ids) + len(self.obstacle_ids)

    # ===========================================================================
    # SUCCESSOR FEATURES
    # ===========================================================================
    def cost_features(self, next_state):
        """
        Computes the state features for the current environment, used for learning successor
        feature representations. First introduced in [1].

        Parameters
        ----------
        next_state : object
            the next state (successor state) of the MDP

        Returns
        -------
        np.ndarray : the state features of the transition

        References
        ----------
        [1] Dayan, Peter. "Improving generalization for temporal difference learning:
        The successor representation." Neural Computation 5.4 (1993): 613-624.
        """
        s_next = next_state[0]
        nc = len(self.obstacles_index)
        phi = np.zeros((nc,))
        if s_next in self.obstacle_ids:
            row, col = s_next
            obstacle_index = self.obstacles_index[self.maze[row, col]]
            phi[obstacle_index] = 1
        return phi

    def cost_feature_dim(self):
        """
        Returns the dimension of the state feature representation.

        Returns
        -------
        integer : the dimension of the state feature representation
        """
        return len(self.obstacles_index)

    def get_w_cost(self):
        """
        Returns a vector of parameters that represents the cost function for the current task.
        Mathematically, given the state features phi(s,a,s') and cost parameters w, the cost
        function is represented as c(s,a,s') = < phi(s,a,s'), w >.

        Returns
        -------
        np.ndarray : a linear parameterization of the cost function of the current MDP
        """
        ns = len(self.obstacles_index)
        w = np.zeros((ns, 1))
        for obstacle_type, obstacle_index in self.obstacles_index.items():
            w[obstacle_index, 0] = self.obstacles[obstacle_type]
        return w

    def features(self, state, next_state):
        """
        Computes the state features for the current environment, used for learning successor
        feature representations. First introduced in [1].
        
        Parameters
        ----------
        state : object
            the state of the MDP
        action : integer
            the action selected in the state
        next_state : object
            the next state (successor state) of the MDP
        
        Returns
        -------
        np.ndarray : the state features of the transition
        
        References
        ----------
        [1] Dayan, Peter. "Improving generalization for temporal difference learning: 
        The successor representation." Neural Computation 5.4 (1993): 613-624.
        """
        s1, _ = next_state
        _, collected = state
        nc = len(self.rewards_index)
        phi = np.zeros((nc + 1,))
        if s1 in self.reward_ids:
            if collected[self.reward_ids[s1]] != 1:
                y, x = s1
                shape_index = self.rewards_index[self.maze[y, x]]
                phi[shape_index] = 1
        elif s1 == self.goal:
            phi[nc] = 1
        return phi
    
    def feature_dim(self):
        """
        Returns the dimension of the state feature representation.
        
        Returns
        -------
        integer : the dimension of the state feature representation
        """
        return len(self.rewards_index) + 1
    
    def get_w(self):
        """
        Returns a vector of parameters that represents the reward function for the current task.
        Mathematically, given the state features phi(s,a,s') and reward parameters w, the reward function
        is represented as r(s,a,s') = < phi(s,a,s'), w >. 
        
        Returns
        -------
        np.ndarray : a linear parameterization of the reward function of the current MDP
        """
        ns = len(self.rewards_index)
        w = np.zeros((ns + 1, 1))
        for reward_type, reward_index in self.rewards_index.items():
            w[reward_index, 0] = self.rewards[reward_type]
        w[ns, 0] = 1.
        return w
