from ABM_model import ABM_model
from ode_model import ode_model
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # set up parameter of test
    test_count = 1
    # dt in times of fraction of day
    dt = 1/24
    # progress threshold as factor of initial tumor size
    threshold = 1.2

    ABM_parameters = {

        "domain_size": 40,
        "T": int(400 * (1/dt)),
        "S0": 200,
        "R0": 10,
        "N0": 0,
        "grS": 0.023 * dt,
        "grR": 0.023 * dt,
        "grN": 0,
        "drS": 0.013 * dt,
        "drR": 0.013 * dt,
        "drN": 0,
        "divrS": 0.75,
        "divrN": 0,
        "dimension": 2,
        "therapy": 'adaptive',
        "save_locations": False,
        "initial_condition_type": "uniform"
    }
    
    parameters = {"domain_size" : 40,
    "T" : 400,
    "S0" : 200,
    "R0" : 10,
    "N0" : 0,
    "grS" : 0.028,
    "grR" : 0.023,
    "grN" : 0.005,
    "drS" : 0.013,
    "drR" : 0.013,
    "drN" : 0.00,
    "divrS" : 0.75,
    "divrN" : 0.5,
    "therapy" : "adaptive",
    "initial_condition_type" : "uniform",
    "save_locations" : False,
    "dimension" : 2}
    # set up parameter of ODE model
    ode_parameters = {
        'time_start': 0,
        'time_end': 1000,
        'time_step': 0.01,
        'tolerance': 0.0001,
        'S0': 2,
        'R0': 0.2,
        'growth_rate_S': 0.028,
        'growth_rate_R': 0.023,
        'carrying_capacity': 10,
        'maximum_tollerated_dose': 1,
        'death_rate_S': 0.013,
        'death_rate_R': 0.013,
        'division_rate': 0.75,
        'therapy_type': 'adaptive',
        'current_state': 1
    }

    # calculate time to progression for the ODE model
    S_ode, R_ode, N_ode, T_ode, D_ode = ode_model(ode_parameters, verbose=False)
    # calculate initial tumor size
    initial_tumor_size = S_ode[0] + R_ode[0]
    # calculate time to progression
    time_to_progression_ODE = 0
    for i in range(len(S_ode)):
        if S_ode[i] + R_ode[i] > threshold * initial_tumor_size:
            time_to_progression_ODE = T_ode[i]
            break
    
    # calculate time to progression for the ABM model
    model = ABM_model(ABM_parameters)
    model.set_initial_condition(parameters["initial_condition_type"])
    model.plot_grid()

    times_to_progressions = []
    model.run(parameters["therapy"])
    fig,ax = plt.subplots()
    print("Plotting first simulation result")
    model.plot_celltypes_density(ax)
    plt.show()

    for i in range(test_count):
        if i == 5:
            print('Halfway there!')
        model.run(parameters["therapy"])
        times_to_progressions.append(model.time_to_progression(threshold))
        # reset grid
        model.reset_grid()

    # calculate how many times the ABM model did not progress
    not_progressed = times_to_progressions.count(-1)
    # remove -1 from list
    times_to_progressions = [x for x in times_to_progressions if x != -1]
    
    # if times_to_progressions is empty, set average time to progression to -1
    if not times_to_progressions:
        average_time_to_progression = -1
    else:
        average_time_to_progression = sum(times_to_progressions)/len(times_to_progressions)
        # normalize to dt
        average_time_to_progression = average_time_to_progression * dt

    print('Number of tests: ', test_count)
    print('Number of times ABM model did not progress: ', not_progressed)
    print('Average time to progression for ABM model: ', average_time_to_progression)
    print('Standard deviation of time to progression for ABM model: ', np.std(times_to_progressions))   
    print('Time to progression for ODE model: ', time_to_progression_ODE)

    # draw plot
    data = model.data
    S = data[1:, 0]
    R = data[1:, 1]
    total = S + R
    T = np.arange(0, len(S) * dt, dt)

    # plot S and R of ODE and ABM model against time
    # plt.plot(T_ode, S_ode, label='S ODE')
    # plt.plot(T_ode, R_ode, label='R ODE')
    plt.plot(T, S, label='S ABM')
    plt.plot(T, R, label='R ABM')
    plt.plot(T, total, label='Total ABM')
    plt.xlabel('Time')
    plt.ylabel('Number of cells')
    plt.legend()
    plt.show()


    


