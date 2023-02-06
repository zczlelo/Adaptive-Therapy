import numpy as np
from initial_conditions import *
from broyden_methods import broyden_method_good

def unpack_parameters(parameters):

    separated_parameters = [
        {
            'type': 0,
            'extra_parameters': parameters['S0_extra_parameters'],
            'growth_rate': parameters['growth_rate_S'],
            'diffusion_coefficient': parameters['diffusion_coefficient_S'],
            'death_rate': parameters['death_rate_S'],
            'division_rate': parameters['division_rate_S'],
            'carrying_capacity': parameters['carrying_capacity'],
            'diffusion_type': parameters['diffusion_type'],
            'maximum_tolerated_dose': parameters['maximum_tolerated_dose'],
            'space_start': parameters['space_start'],
            'space_end': parameters['space_end'],
            'space_points': parameters['space_points']
        }, 
        {
            'type': 1,
            'extra_parameters': parameters['R0_extra_parameters'],
            'growth_rate': parameters['growth_rate_R'],
            'diffusion_coefficient': parameters['diffusion_coefficient_R'],
            'death_rate': parameters['death_rate_R'],
            'division_rate': 0,
            'carrying_capacity': parameters['carrying_capacity'],
            'diffusion_type': parameters['diffusion_type'],
            'maximum_tolerated_dose': parameters['maximum_tolerated_dose'],
            'space_start': parameters['space_start'],
            'space_end': parameters['space_end'],
            'space_points': parameters['space_points']
        },
        {
            'type': 2,
            'growth_rate': parameters['growth_rate_N'],
            'diffusion_coefficient': parameters['diffusion_coefficient_N'],
            'death_rate': parameters['death_rate_N'],
            'division_rate': parameters['division_rate_N'],
            'carrying_capacity': parameters['carrying_capacity'],
            'diffusion_type': parameters['diffusion_type'],
            'maximum_tolerated_dose': parameters['maximum_tolerated_dose'],
            'space_start': parameters['space_start'],
            'space_end': parameters['space_end'],
            'space_points': parameters['space_points']
        }
    ]

    return separated_parameters

def compute_denisity_grid(U, parameters):
    x = int(np.sqrt(len(U)//3))
    return np.reshape(U[:x**2], (x,x)) + np.reshape(U[x**2:2*x**2], (x,x)) + np.reshape(U[2*x**2:], (x,x))

def unpack_index(k, parameters):
    
    space_points = parameters['space_points']

    type = k // (space_points**2)
    i = (k - type * space_points**2) // space_points
    j = k - type * space_points**2 - i * space_points

    return i, j, type

def get_neighbourhood(k, space_points):

    return_array = np.zeros(5)
    type = k // (space_points**2)
    return_array[0] = k
    return_array[1] = (k + 1) % (space_points**2) + type * space_points**2
    return_array[2] = (k - 1) % (space_points**2) + type * space_points**2
    return_array[3] = (k + space_points) % (space_points**2) + type * space_points**2
    return_array[4] = (k - space_points) % (space_points**2) + type * space_points**2
    
    return return_array
    
def compute_diffusion_coefficient(value, parameters):
    if parameters['diffusion_type'] == 'standard':
        return parameters['diffusion_coefficient']
    else:
        raise ValueError('Invalid diffusion type')

def compute_scheme(parameters, density, neighbourhood):

    point_value = neighbourhood[0]
    drug_concentration = parameters['drug_concentration']
    diffusion_coefficient = compute_diffusion_coefficient(point_value, parameters)
    maximum_tolerated_dose = parameters['maximum_tolerated_dose']
    carrying_capacity = parameters['carrying_capacity']
    growth_rate = parameters['growth_rate']
    death_rate = parameters['death_rate']
    space_end = parameters['space_end']
    space_start = parameters['space_start']
    space_points = parameters['space_points']
    space_step = (space_end - space_start) / space_points
    division_rate = parameters['division_rate']

    effectivie_growth_rate = growth_rate * (1 - density/carrying_capacity)*(1 - 2*division_rate*(drug_concentration/maximum_tolerated_dose)) - death_rate
    effective_diffusion_coefficient = diffusion_coefficient/(space_step**2)

    return effectivie_growth_rate * point_value + effective_diffusion_coefficient * (sum(neighbourhood[1:]) - 4 * point_value)

def therapy_drug_concentration(density, parameters):

    therapy_type = parameters['therapy_type']
    maximum_tolerated_dose = parameters['maximum_tolerated_dose']

    if therapy_type == 'continuous':
        return maximum_tolerated_dose
    if therapy_type == 'notherapy':
        return 0
    elif therapy_type == 'adaptive':
        initial_size = parameters['S0'] + parameters['R0'] + parameters['N0']
        current_size = np.sum(density)
        current_state = parameters['current_state']
        # on therapy
        if current_state == 1:
            # if shrunk sufficiently, turn off therapy
            if current_size < initial_size/2:
                parameters['current_state'] = 0
                return 0
            else:
            # else, keep on therapy
                return maximum_tolerated_dose
        # off therapy
        else:
            # if grown sufficiently, turn on therapy
            if current_size > initial_size:
                parameters['current_state'] = 1
                return maximum_tolerated_dose
            else:
            # else, keep off therapy
                return 0

def construct_F(U, parameters):

    density_grid = compute_denisity_grid(U, parameters)
    parameters_by_type = unpack_parameters(parameters)
    time_step = parameters['time_step']
    space_points = parameters['space_points']
    drug_concentration = therapy_drug_concentration(density_grid, parameters)

    for dic in parameters_by_type:
        dic['drug_concentration'] = drug_concentration

    def F(solution):

        result = np.zeros_like(solution)
        

        for k in range(len(solution)):

            time_derivative = (solution[k] - U[k]) / time_step

            i, j, type = unpack_index(k, parameters)

            neighbourhood_coordinates = get_neighbourhood(k, space_points)
            neighbourhood = [solution[int(h)] for h in neighbourhood_coordinates]
            result[k] = compute_scheme(parameters_by_type[type], density_grid[i][j], neighbourhood) - time_derivative

        return result

    return F

def pde_model(parameters):

    time_step = parameters['time_step']
    time_start = parameters['time_start']
    time_end = parameters['time_end']

    space_points = parameters['space_points']
    space_start = parameters['space_start']
    space_end = parameters['space_end']

    S0 = parameters['S0']
    S0_distribution = parameters['S0_distribution']
    S0_extra_parameters = parameters['S0_extra_parameters']
    initial_S = set_up_initial_condition(S0, S0_distribution, space_points, space_start, space_end, S0_extra_parameters)

    R0 = parameters['R0']
    R0_distribution = parameters['R0_distribution']
    R0_extra_parameters = parameters['R0_extra_parameters']
    initial_R = set_up_initial_condition(R0, R0_distribution, space_points, space_start, space_end, R0_extra_parameters)

    N0 = parameters['N0']
    N0_distribution = parameters['N0_distribution']
    N0_extra_parameters = parameters['N0_extra_parameters']
    initial_N = set_up_initial_condition(N0, N0_distribution, space_points, space_start, space_end, N0_extra_parameters)
    
    U = np.concatenate((initial_S, initial_R, initial_N))

    current_time = time_start
    while current_time + time_step < time_end:
        F = construct_F(U, parameters)
        U_new = broyden_method_good(F, U)[0]
        print(U_new)
        U = U_new
        current_time += time_step

    return 0

parameters = {
    'time_start': 0,
    'time_end': 0.3,
    'time_step': 0.1,
    'space_start': 0,
    'space_end': 0.5,
    'space_points': 3,
    'S0': 1.4,
    'R0': 0.02,
    'N0': 1.42,
    'S0_distribution': 'uniform',
    'R0_distribution': 'uniform',
    'N0_distribution': 'uniform',
    'S0_extra_parameters': [0.1, 0.1],
    'R0_extra_parameters': [0.1, 0.1],
    'N0_extra_parameters': [0.1, 0.1],
    'growth_rate_S': 0.03,
    'growth_rate_R': 0.03,
    'growth_rate_N': 0.03,
    'carrying_capacity': 10,
    'diffusion_coefficient_S': 0.0001,
    'diffusion_coefficient_R': 0.0001,
    'diffusion_coefficient_N': 0.0001,
    'standard_deviation_S': 0.01,
    'standard_deviation_R': 0.01,
    'standard_deviation_N': 0.01,
    'maximum_tolerated_dose': 1,
    'death_rate_S': 0.03,
    'death_rate_R': 0.03,
    'death_rate_N': 0.03,
    'division_rate_S': 0.4,
    'division_rate_N': 0.4,
    'therapy_type': 'continuous',
    'time_boundary_conditions': 'Periodic',
    'S0_left': 0,
    'R0_left': 0,
    'S0_right': 0,
    'R0_right': 0,
    'diffusion_type': 'standard'
}

pde_model(parameters)