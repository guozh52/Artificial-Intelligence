import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


class Gridworld_HMM:
    def __init__(self, size, epsilon: float = 0, walls: bool = False):
        if walls:
            self.grid = np.zeros(size)
            for cell in walls:
                self.grid[cell] = 1
        else:
            self.grid = np.random.randint(2, size=size)

        self.epsilon = epsilon
        self.trans = self.initT()
        self.obs = self.initO()

    def neighbors(self, cell):
        i, j = cell
        M, N = self.grid.shape
        adjacent = [
            (i - 1, j - 1), (i - 1, j), (i - 1, j + 1), (i, j - 1), 
            (i, j),(i, j + 1), (i + 1, j - 1), (i + 1, j), (i + 1, j + 1),
        ]
        neighbors = []
        for a in adjacent:
            if a[0] >= 0 and a[0] < M and a[1] >= 0 and a[1] < N and self.grid[a] == 0:
                neighbors.append(a)
        return neighbors

    """
    4.1 Transition and observation probabilities
    """
    
    
    def initT(self):
        
        """
        Create and return NxN transition matrix, where N = size of grid.
        """
        M, N = self.grid.shape
        T = np.zeros((M * N, M * N)) # Transition matrix

        for i in range(M):
            for j in range(N):
                cell = [i, j]
                neighbors = self.neighbors(cell)
                pr = 1 / len(neighbors)
                for neighbor_ in neighbors:
                    T[neighbor_[0] * N + neighbor_[1], cell[0] * N + cell[1]] = pr
        
        for i in range(M * N):
            T[:, i] = T[:, i] / sum(T[:, i])

        
        return T

    def initO(self):
        """
        Create and return 16xN matrix of observation probabilities, where N = size of grid.
        """
        M, N = self.grid.shape
        O = np.zeros((16, M * N))
        # TODO:
        for i in range(M):
            for j in range(N):
                cell = [i, j]
                neighbors = self.neighbors(cell)
                obs_binary = []

                #Observing the environment, following the sequence of NESW
                #If it is a wall, return 1
                if (i-1, j) in neighbors:
                    obs_binary.append(0)
                else:
                    obs_binary.append(1)
                if (i, j+1) in neighbors:
                    obs_binary.append(0)
                else:
                    obs_binary.append(1)
                if (i+1, j) in neighbors:
                    obs_binary.append(0)
                else:
                    obs_binary.append(1)
                if (i, j-1) in neighbors:
                    obs_binary.append(0)
                else:
                    obs_binary.append(1) 

                #Convert binary string into integer
                obs_binary_string = ''.join(str(bit) for bit in obs_binary)
                obs_decimal = int(obs_binary_string, 2)
                #Compare obs_integer with all 16 possible obs
                for index_ in range(16):
                    num_of_one = bin(obs_decimal ^ index_).count('1')
                    O[index_, i * N + j] = pow((1 - self.epsilon), 4 - num_of_one) * pow(self.epsilon, num_of_one)
        
        return O

    """
    4.2 Inference: Forward, backward, filtering, smoothing
    """

    def forward(self, alpha: npt.ArrayLike, observation: int):
        """Perform one iteration of the forward algorithm.
        Args:
          alpha (np.ndarray): Current belief state.
          observation (int): Integer representation of bitstring observation.
        Returns:
          np.ndarray: Updated belief state.
        """
        # TODO:
        alpha = self.trans @ alpha
        alpha = self.obs[observation] * alpha    

        return alpha

    def backward(self, beta: npt.ArrayLike, observation: int):
        """Perform one iteration of the backward algorithm.
        Args:
          beta (np.ndarray): Current "message" of probabilities.
          observation (int): Integer representation of bitstring observation.
        Returns:
          np.ndarray: Updated message.
        """

        # TODO: 
        beta = self.obs[observation] * beta
        beta = self.trans.T @ beta

        return beta

    def filtering(self, init: npt.ArrayLike, observations: list[int]):
        """Perform filtering over all observations.
        Args:
          init (np.ndarray): Initial belief state.
          observations (list[int]): List of integer observations.
        Returns:
          np.ndarray: Estimated belief state at each timestep.
        """
        # TODO:
        N = self.grid.size
        T = len(observations)
        est_belief_state = np.zeros((N, T))
          
        est_belief_state[:, 0] = self.forward(init, observations[0])
        for i in range(1, T):
            est_belief_state[:, i] = self.forward(est_belief_state[:, i-1], observations[i])
            est_belief_state[:, i] = est_belief_state[:, i] / sum(est_belief_state[:, i])
        
        return est_belief_state
        

    def smoothing(self, init: npt.ArrayLike, observations: list[int]):
        """Perform smoothing over all observations.
        Args:
          init (np.ndarray): Initial belief state.
          observations (list[int]): List of integer observations.
        Returns:
          np.ndarray: Smoothed belief state at each timestep.
        """
        # TODO:
        est_belief_state = self.filtering(init, observations)

        beta = np.zeros(est_belief_state.shape)
        beta[:, -1] = 1

        for column_ in range(est_belief_state.shape[1] - 2, -1, -1):
            beta[:, column_] = self.backward(beta[:, column_ + 1], observations[column_ + 1])
        
        smooth_belief_state = est_belief_state * beta
        for column_ in range(smooth_belief_state.shape[1]):
            smooth_belief_state[:, column_] = smooth_belief_state[:, column_] / sum(smooth_belief_state[:, column_])

        return smooth_belief_state


    """
    4.3 Localization error
    """

    def loc_error(self, beliefs: npt.ArrayLike, trajectory: list[int]):
        """Compute localization error at each timestep.
        Args:
          beliefs (np.ndarray): Belief state at each timestep.
          trajectory (list[int]): List of states visited.
        Returns:
          list[int]: Localization error at each timestep.
        """
        # TODO:
        trajectory_pred = []
        local_err = []
        for i in range(beliefs.shape[1]):
            trajectory_pred.append(beliefs[:, i].argmax())

        M, N = self.grid.shape

        for i in range(len(trajectory)):
            row_, column_ = divmod(trajectory[i], N)
            #print([row_, column_])
            row_p, column_p = divmod(trajectory_pred[i], N)
            #print([row_p, column_p])
            manhattan_dis = abs(row_ - row_p) + abs(column_ - column_p)
            local_err.append(manhattan_dis)

        return local_err
