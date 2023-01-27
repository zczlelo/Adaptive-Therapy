import numpy as np
import matplotlib.pyplot as plt


def initial_conditions(parameters):

    S0 = parameters['S0']
    R0 = parameters['R0']
    S0_distribution = parameters['S0_distribution']
    R0_distribution = parameters['R0_distribution']
    space_points = parameters['space_points']

    if S0_distribution == 'uniform':
        S = np.random.uniform(0, 1, space_points)
    elif S0_distribution == 'normal':
        S = np.random.normal(1, 0, space_points)
    else:
        raise ValueError('S0_distribution must be uniform or normal')

    if R0_distribution == 'uniform':
        R = np.random.uniform(0, 1, space_points)
    elif R0_distribution == 'normal':
        R = np.random.normal(1, 0, space_points)
    else:
        raise ValueError('R0_distribution must be uniform or normal')

    S = S0 * S
    R = R0 * R
    N = S + R

    return S, R, N


def one_step()



def pde_2D_model(parameters):

    time_start = parameters['time_start']
    time_end = parameters['time_end']
    time_points = parameters['time_points']

    space_start = parameters['space_start']
    space_end = parameters['space_end']
    space_points = parameters['space_points']

    X = np.linspace(space_start, space_end, space_points)
    T = np.linspace(time_start, time_end, time_points)

    S = np.zeros((space_points, time_points))
    R = np.zeros((space_points, time_points))
    N = np.zeros((space_points, time_points))
    D = np.zeros(time_points)

    index = 0

    S[:, index], R[:, index], N[:, index] = initial_conditions(parameters)

    for t in T[1:]:
        S[:, index + 1], R[:, index + 1], N[:, index + 1], D[index + 1] = one_step(S[:, index], R[:, index], X, parameters)
        index += 1
    
    return S, R, N, D, X, T




if __name__ == '__main__':

    parameters = {
        'time_start': 0,                                  
        'time_end': 2000,
        'time_points': 1000,
        'space_start': 0,
        'space_end': 10,
        'space_points': 1000,
        'tolerance': 100,
        'S0': 1.4,
        'R0': 0.01,
        'S0_distribution': 'uniform',
        'R0_distribution': 'uniform',
        'growth_rate_S': 0.03,
        'growth_rate_R': 0.03,
        'carrying_capacity': 10,
        'diffusion_coefficient_S': 0.1,
        'diffusion_coefficient_R': 0.1,
        'maximum_tollerated_dose': 0.8,
        'death_rate_S': 0.03,
        'death_rate_R': 0.02,
        'division_rate': 0.04,
        'therapy_type': 'continuous'
    }

