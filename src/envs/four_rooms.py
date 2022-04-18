"""
Four rooms environment
"""
import numpy as np


class FourRooms:
    """
    A four rooms grid world environment
    """

    actions = {
        0: (0, -1), # LEFT
        1: (-1, 0), # UP
        2: (0, 1), # RIGHT
        3: (1, 0) # DOWN
    } 

    def __init__(
        self, 
        maze: np.ndarray, 
        rewards: dict, 
        obstacles: dict
    ) -> None:
        """
        Initializes a new instance of environment.

        Args
            maze (np.ndarray): An array of string values representing the type of each cell in the environment:
                G: Goal state (terminal state)
                _: Initial state 
                X: Wall
                0, 1, ..., 9: Reward state
                A, B, ..., Z: Obstacle state
            rewards (dict): A dictionary mapping the rewards
            obstacles (dict): A dictionary mapping the obstacles
        """
        self.height, self.width = maze.shape
        self.maze = maze
        self.rewards = rewards
        self.obstacles = obstacles

        reward_types = sorted(list(rewards.keys()))
        obstacle_types = sorted(list(obstacles.keys()))
        # Type index map, len equals to number of types
        self.rewards_index = dict(zip(reward_types, range(len(reward_types))))
        self.obstacles_index = dict(zip(obstacle_types, range(len(obstacle_types))))

        self.goal = None
        # TO DO: List might not be neccessary, start and end locations are in pairs
        self.initial = None
        self.occupied = set()
        # Index for each reward/obstacle, len equals to total number of rewards/obstacles
        self.reward_ids = dict()
        self.obstacle_ids = dict()
        for col in range(self.width):
            for row in range(self.height):
                if maze[row, col] == "G":
                    self.goal = (row, col)
                elif maze[row, col] == "_":
                    self.initial = (row, col)
                elif maze[row, col] == "X":
                    self.occupied.add((row, col))
                elif maze[row, col].isnumeric():
                    self.reward_ids[(row, col)] = len(self.reward_ids)
                elif maze[row, col].isalpha():
                    # Since "G" and "X" are already checked, no need to exclude them
                    self.obstacle_ids[(row, col)] = len(self.obstacle_ids)        

    def initialize(self) -> tuple:
        """
        Initializes the state of the environment.

        Returns:
            tuple: The initial state
        """
        self.state = (
            self.initial,
            tuple(False for _ in range(len(self.reward_ids)))
        )
        return self.state

    def action_count(self) -> None:
        """
        Returns the number of possible actions in the MDP.

        Returns:
            int: Number of possible actions
        """
        return 4

    def transition(self, action: int) -> tuple:
        """
        Agent takes the specified action, updating the state of the MDP.

        Args:
            action (int): Current action

        Returns:
            tuple:
                tuple : the next state 
                float : the immediate reward observed in the transition
                float : the immediate cost observed in the transition
                boolean : whether or not a terminal state has been reached
        """
        (row, col), rewards_collected = self.state

        # Take action
        if action > 3 or action < 0:
            raise Exception("bad action {}".format(action))

        row += FourRooms.actions[action][0]
        col += FourRooms.actions[action][1]        

        # Out of bounds, cannot move
        if col < 0 or col >= self.width or row < 0 or row >= self.height:
            return self.state, 0, 0, False

        s_next = (row, col)

        # In a blocked cell, cannot move
        if s_next in self.occupied:
            return self.state, 0, 0, False

        # Valid action, move to next cell
        self.state = (s_next, rewards_collected)

        # In a goal cell
        if s_next == self.goal:
            return self.state, 1, 0, True

        # In a reward cell
        if s_next in self.reward_ids:
            reward_id = self.reward_ids[s_next]
            if rewards_collected[reward_id]:
                # Already collected this flag
                return self.state, 0, 0, False
            else:
                # Collect the new flag
                rewards_collected = list(rewards_collected)
                rewards_collected[reward_id] = True
                self.state = (s_next, tuple(rewards_collected))
                reward = self.rewards[self.maze[row, col]]
                return self.state, reward, 0, False

        # In a obstacle cell
        if s_next in self.obstacle_ids:
            # Collect the new flag
            self.state = (s_next, rewards_collected)
            cost = self.obstacles[self.maze[row, col]]
            return self.state, 0, cost, False

        # In an empty cell
        return self.state, 0, 0, False

    # ===========================================================================
    # SUCCESSOR FEATURES
    # ===========================================================================
    def cost_features(self, state: tuple, action: int, next_state: tuple) -> np.ndarray:
        """
        Computes the cost features for the current environment, used for learning safety
        feature representations. 

        Args:
            state (tuple): Current state
            action (int): Current action
            next_state (tuple): Next state

        Returns:
            np.ndarray: The cost features of the transition
        """
        s_next = next_state[0]
        num_cost = len(self.obstacles_index)
        phi_c = np.zeros((num_cost,))
        if s_next in self.obstacle_ids:
            row, col = s_next
            obstacle_index = self.obstacles_index[self.maze[row, col]]
            phi_c[obstacle_index] = 1
        return phi_c

    def cost_feature_dim(self) -> int:
        """
        Returns the dimension of the cost features.

        Returns:
            int: The dimension of the cost feature 
        """
        return len(self.obstacles_index)

    def get_w_c(self) -> np.ndarray:
        """
        Returns a vector of parameters that represents the cost function for the current task.

        Returns:
            np.ndarray: A linear parameterization of the cost function of the current MDP
        """
        num_cost = len(self.obstacles_index)
        w_c = np.zeros((num_cost, 1))
        for obstacle_type, obstacle_index in self.obstacles_index.items():
            w_c[obstacle_index, 0] = self.obstacles[obstacle_type]
        return w_c

    def features(self, state, action, next_state) -> np.ndarray:
        """
        Computes the reward features for the current environment, used for learning successor
        feature representations. 

        Args:
            state (tuple): Current state
            action (int): Current action
            next_state (tuple): Next state

        Returns:
            np.ndarray: The reward features of the transition
        """
        s1 = next_state[0]
        collected = state[1]
        num_reward = len(self.rewards_index)
        phi = np.zeros((num_reward + 1,))
        if s1 in self.reward_ids and collected[self.reward_ids[s1]] != 1:
            y, x = s1
            shape_index = self.rewards_index[self.maze[y, x]]
            phi[shape_index] = 1
        elif s1 == self.goal:
            phi[num_reward] = 1
        return phi
    
    def feature_dim(self) -> int:
        """
        Returns the dimension of the reward features.

        Returns:
            int: The dimension of the reward feature 
        """
        return len(self.rewards_index) + 1
    
    def get_w(self) -> np.ndarray:
        """
        Returns a vector of parameters that represents the reward function for the current task.

        Returns:
            np.ndarray: A linear parameterization of the reward function of the current MDP
        """
        num_reward = len(self.rewards_index)
        w = np.zeros((num_reward + 1, 1))
        for reward_type, reward_index in self.rewards_index.items():
            w[reward_index, 0] = self.rewards[reward_type]
        w[num_reward, 0] = 1
        return w
