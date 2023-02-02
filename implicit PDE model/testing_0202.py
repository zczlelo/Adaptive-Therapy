import numpy as np

def unpack_parameters(parameters):

    array = [{}, {}, {}]

    return array

def compute_denisity_grid(U, parameters):
    
    return grid

def unpack_index(k, parameters):
    
    space_points = parameters['space_points']

    type = k // (space_points**2)
    i = (k - type * space_points**2) // space_points
    j = k - type * space_points**2 - i * space_points

    return i, j, type

def get_neighbourhood(i, j, type):

    return array

def compute_scheme(i, j, parameters, density, neighbourhood):

    return value

def construct_F(U, parameters):

    density_grid = compute_density_grid(U, parameters)
    parameters_by_type = unpack_parameters(parameters)
    time_step = parameters['time_step']

    def F(solution):

        result = np.zeros_like(solution)
        

        for k in range(len(solution)):

            time_derivative = (solution[k] - U[k]) / time_step

            i, j, type = unpack_index(k, parameters)

            if type == 0:
                neighbourhood_coordinates = get_neighbourhood(i, j, type)
                neighbourhood = [solution[h] for h in neighbourhood_coordinates]
                solution[k] = compute_scheme(i,j, parameters_by_type[0], density_grid[i][j], neighbourhood) - time_derivative
            elif type == 1:
                neighbourhood_coordinates = get_neighbourhood(i, j, type)
                neighbourhood = [solution[h] for h in neighbourhood_coordinates]
                solution[k] = compute_scheme(i,j, parameters_by_type[1], density_grid[i][j], neighbourhood) - time_derivative
            elif type == 2:
                neighbourhood_coordinates = get_neighbourhood(i, j, type)
                neighbourhood = [solution[h] for h in neighbourhood_coordinates]
                solution[k] = compute_scheme(i,j, parameters_by_type[2], density_grid[i][j], neighbourhood) - time_derivative
            else:
                raise ValueError("Type must be 0, 1 or 2")
        return result

    return F