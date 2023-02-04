import numpy as np

def unpack_parameters(parameters):

    array = [{}, {}, {}]

    return array

def compute_denisity_grid(U, parameters):

    space_points = parameters['space_points']
    return_grid = np.zeros((space_points, space_points))
    
    for i in range(space_points):
        for j in range(space_points):
            k1 = i * space_points + j
            k2 = k1 + space_points**2
            k3 = k2 + space_points**2
            return_grid[i][j] = U[k1] + U[k2] + U[k3]
    
    return return_grid

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

def compute_scheme(parameters, density, neighbourhood):

    return value

def construct_F(U, parameters):

    density_grid = compute_denisity_grid(U, parameters)
    parameters_by_type = unpack_parameters(parameters)
    time_step = parameters['time_step']
    space_points = parameters['space_points']

    # TODO drug concentration calculation

    def F(solution):

        result = np.zeros_like(solution)
        

        for k in range(len(solution)):

            time_derivative = (solution[k] - U[k]) / time_step

            i, j, type = unpack_index(k, parameters)

            neighbourhood_coordinates = get_neighbourhood(k, space_points)
            neighbourhood = [solution[h] for h in neighbourhood_coordinates]
            result[k] = compute_scheme(parameters_by_type[type], density_grid[i][j], neighbourhood) - time_derivative

        return result

    return F