from AMB_model import AMB_model
from ode_model import ode_model

if __name__ == "__main__":

    # set up parameter of test
    test_count = 1
    # dt in times of fraction of day
    dt = 1/24
    # progress threshold as factor of initial tumor size
    threshold = 2

    # set up parameters of model
    N = 20
    T = 800
    R0 = 1
    S0 = 3
    grS = 0.028 * dt
    grR = 0.23 * dt
    drS = 0.013 * dt
    drR = 0.013 * dt
    divrS = 0.75
    therapy = 'adaptive'
    initial_condition_type = 'random'

    # set up parameter of ODE model
    ode_parameters = {
        'time_start': 0,
        'time_end': T * (1/dt),
        'time_step': 0.01,
        'tolerance': 0.0001,
        'S0': 2,
        'R0': 1/20,
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
    
    # calculate time to progression for the AMB model
    model = AMB_model(N, T, S0, R0, grS, grR, drS, drR, divrS)
    model.set_initial_condition(initial_condition_type)

    times_to_progressions = []

    for i in range(test_count):
        model.run(therapy)
        times_to_progressions.append(model.time_to_progression(threshold))
        # reset grid
        model.reset_grid()
    
    # calculate average time to progression
    average_time_to_progression = sum(times_to_progressions) / len(times_to_progressions)

    print('Average time to progression for AMB model: ', average_time_to_progression)
    print('Time to progression for ODE model: ', time_to_progression_ODE)
