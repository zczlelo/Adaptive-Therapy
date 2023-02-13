from matplotlib import gridspec
import numpy as np
import random
import matplotlib.pyplot as plt

class AMB_model:
    def __init__(self, N, T, S0, R0, grS, grR, drS, drR, divrS):
        self.N = N
        self.T = T
        self.S0 = S0
        self.R0 = R0
        self.grS = grS
        self.grR = grR
        self.drS = drS
        self.drR = drR
        self.divrS = divrS
        self.grid = np.zeros((self.N, self.N))
        self.data = np.zeros((T, 3))
        self.current_therapy = 1
        self.data[0, 0] = np.sum(self.grid == 1)
        self.data[0, 1] = np.sum(self.grid == 2)
        self.data[0, 2] = self.current_therapy
        

    def set_initial_condition(self, initial_condition_type):
        if initial_condition_type == 'random':
            # select random 2D coordinates for S0 cells
            # randmoly set S0 grid points to 1[
            S0_idx = [divmod(i, self.N) for i in random.sample(range(self.N ** 2), self.S0)]
            for idx in S0_idx:
                self.grid[idx] = 1
            # randmoly set R0 grid points to 2
            R0_idx = [divmod(i, self.N) for i in random.sample(range(self.N ** 2), self.R0)]
            for idx in R0_idx:
                self.grid[idx] = 2
        elif initial_condition_type == 'cluster':
            # make a ball of S0 cells
            for i in range(self.N):
                for j in range(self.N):
                    if (i - self.N / 2) ** 2 + (j - self.N / 2) ** 2 < self.S0:
                        self.grid[i, j] = 1
            # make a ball of R0 cells
            for i in range(self.N):
                for j in range(self.N):
                    if (i - self.N / 2) ** 2 + (j - self.N / 2) ** 2 < self.R0:
                        self.grid[i, j] = 2
        elif initial_condition_type == 'two_clusters':
            # make a ball of S0 cells
            for i in range(self.N):
                for j in range(self.N):
                    if (i - self.N / 4) ** 2 + (j - self.N / 4) ** 2 < self.S0:
                        self.grid[i, j] = 1
            # make a ball of R0 cells
            for i in range(self.N):
                for j in range(self.N):
                    if (i - 3 * self.N / 4) ** 2 + (j - 3 * self.N / 4) ** 2 < self.R0:
                        self.grid[i, j] = 2
        else:
            print('initial condition type not recognized')

        # # set up grid
        # grid = np.zeros((self.N, self.N))
        # # make a ball of S0 cells
        # for i in range(self.N):
        #     for j in range(self.N):
        #         if (i - self.N / 2) ** 2 + (j - self.N / 2) ** 2 < self.S0:
        #             grid[i, j] = 1
        # # make a ball of R0 cells
        # for i in range(self.N):
        #     for j in range(self.N):
        #         if (i - self.N / 2) ** 2 + (j - self.N / 2) ** 2 < self.R0:
        #             grid[i, j] = 2

        # # select random 2D coordinates for S0 cells
        # # randmoly set S0 grid points to 1[
        # S0_idx = [divmod(i, self.N) for i in random.sample(range(self.N ** 2), self.S0)]
        # for idx in S0_idx:
        #     grid[idx] = 1
        # # randmoly set R0 grid points to 2
        # R0_idx = [divmod(i, self.N) for i in random.sample(range(self.N ** 2), self.R0)]
        # for idx in R0_idx:
        #     grid[idx] = 2
        # # resample points for which grid[i] == 1
        # # while np.any(grid[R0_idx] == 1):
        #     # R0_idx = [divmod(i, self.N) for i in random.sample(range(self.N ** 2), self.R0)]
        # return gridspec

    def run(self, therapy_type):
        # run model for T iterations
        for t in range(1, self.T):
            if t % 50 == 0:
                self.print_grid()
            self.set_therapy(therapy_type, t)
            # compute death of cells
            self.compute_death()
            # compute growth of cells
            self.compute_growth_S()
            # apply therapy
            self.compute_growth_R()

            # compute number of cells with 1 or 2
            self.data[t, 0] = np.sum(self.grid == 1)
            self.data[t, 1] = np.sum(self.grid == 2)
            self.data[t, 2] = self.current_therapy
    
    def print_grid(self):
        # plot the grid in different colors for S and R
        # make scatter plot
        plt.scatter(np.argwhere(self.grid == 1)[:, 0],np.argwhere(self.grid == 1)[:, 1], color='blue')
        plt.scatter(np.argwhere(self.grid == 2)[:, 0],np.argwhere(self.grid == 2)[:, 1], color='red')
        plt.xlim(0, self.N)
        plt.ylim(0, self.N)
        plt.show()

    def compute_death(self):
        # compute death of cells
        # get all cells with S
        cells = np.argwhere(self.grid == 1)
        for cell in cells:
            if np.random.random() < self.drS:
                # cell dies
                self.grid[cell[0]][cell[1]] = 0
        cells = np.argwhere(self.grid == 2)
        for cell in cells:
            if np.random.random() < self.drR:
                # cell dies
                self.grid[cell[0]][cell[1]] = 0

    def compute_growth_S(self):
        # get all cells with S
        cells = np.argwhere(self.grid == 1)
        for cell in cells:
            # it grows with probability grS
            if np.random.random() < self.grS:
                # get all neigbours
                neigbours = self.get_neigbours(cell)
                count = 0
                for neigbour in neigbours:
                    if self.grid[neigbour[0]][neigbour[1]] == 0:
                        count += 1
                # check if a neigbour is empty
                if count > 0:
                    # check if therapy is succeful
                    if self.current_therapy and np.random.random() < self.divrS:
                            # cell dies during division
                            self.grid[cell[0]][cell[1]] = 0
                    else:
                        # shuffle neigbours
                        np.random.shuffle(neigbours)
                        # check one by one if they are empy
                        for neigbour in neigbours:
                            if self.grid[neigbour[0]][neigbour[1]] == 0:
                                # if empty, cell divides
                                self.grid[neigbour[0]][neigbour[1]] = 1
                                break
                

    def compute_growth_R(self):
        # get all cells with R
        cells = np.argwhere(self.grid == 2)
        for cell in cells:
            # it grows with probability grR
            if np.random.random() < self.grR:
                # get all neigbours
                neigbours = self.get_neigbours(cell)
                count = 0
                for neigbour in neigbours:
                    if self.grid[neigbour[0]][neigbour[1]] == 0:
                        count += 1
                # check if a neigbour is empty
                if count > 0:
                    # shuffle neigbours
                    np.random.shuffle(neigbours)
                    # check one by one if they are empy
                    for neigbour in neigbours:
                        if self.grid[neigbour[0]][neigbour[1]] == 0:
                            # if empty, cell divides
                            self.grid[neigbour[0]][neigbour[1]] = 2
                            break

    def get_neigbours(self, cell):
        # get neigbours of cell
        neigbours = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i != 0 or j != 0:
                    neigbours.append([cell[0] + i, cell[1] + j])
        # check if neigbours are in the grid
        neigbours = [neigbour for neigbour in neigbours if neigbour[0] >= 0 and neigbour[0] < self.N and neigbour[1] >= 0 and neigbour[1] < self.N]
        return neigbours

    def set_therapy(self, therapy_type, t):
        # set current therapy
        if therapy_type == 'notherapy':
            self.current_therapy = 0
        elif therapy_type == 'continuous':
            self.current_therapy = 0
        elif therapy_type == 'adaptive':
            
            N = np.sum(self.grid != 0)
            N0 = self.S0 + self.R0
            if self.current_therapy and N < 0.5 * N0:
                self.current_therapy = 0
            elif not self.current_therapy and N > N0:
                self.current_therapy = 1
        else:
            raise ValueError('Therapy type not recognized')

    def get_data(self):
        # get data
        return self.data

    def plot_celltypes_density(self, ax):
        # plot cell types density
        ax.plot(np.arange(1,self.T), self.data[1:, 0], label='S')
        ax.plot(np.arange(1,self.T), self.data[1:, 1], label='R')
        ax.plot(np.arange(1,self.T), self.data[1:, 2] * 100, label='Therapy')
        ax.set_xlabel('Time')
        ax.set_ylabel('Density')
        ax.legend()
        return ax


if __name__ == "__main__":

    # set up parameters
    N = 200
    T = 400
    R0 = 10
    grS = 0.028
    grR = 0.023
    drS = 0.013
    drR = 0.013
    divrS = 0.75
    therapy = 'adaptive'
    initial_condition_type = 'cluster'
    S0 = 200

    # set up model
    model = AMB_model(N, T, S0, R0, grS, grR, drS, drR, divrS)
    # set up initial condition
    model.set_initial_condition(initial_condition_type)
    # show grid of initial conditions
    model.print_grid()
    # run simulation
    model.run(therapy)
    # plot data
    fig, ax = plt.subplots(1, 1)
    ax = model.plot_celltypes_density(ax)
    plt.show()



    # # do a parameter sweep
    # therapies = ['notherapy', 'continuous', 'adaptive']
    # initial_conditions_types = ['random', 'cluster', 'two_clusters']
    # S0s = [50, 100, 200]
    # for initial_condition_type in initial_conditions_types:
    #     for S0 in S0s:
    #             # set up model
    #         model = AMB_model(N, T, S0, R0, grS, grR, drS, drR, divrS)
    #             # set up initial condition
    #         model.set_initial_condition(initial_condition_type)
    #             # show grid of initial conditions
    #             # model.print_grid()
    #             # run simulation
    #         model.run(therapy)
    #             # # plot data
    #         fig, ax = plt.subplots(1, 1)
    #         ax = model.plot_celltypes_density(ax)
    #         ax.set_title('Therapy: {}, Initial condition: {}, S0: {}'.format(therapy, initial_condition_type, S0))
    #             # # save figure
    #         fig.savefig('elene_{}_{}_{}.png'.format(therapy, initial_condition_type, S0))
    #         plt.close(fig)
