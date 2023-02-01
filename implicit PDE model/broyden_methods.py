import numpy as np

def broyden_method_sparse(F, x0, parameters, zeros, maximum_iterations = 100, tolerance = 1e-6, verbose = False):



    return x, error, i

def broyden_method_bad(F, x0, parameters, maximum_iterations = 100, tolerance = 1e-6, verbose = False):  
    # initialize
    y0 = F(x0)
    # compute inverse of J
    J_inverse = np.eye(len(x0))
    # compute the error
    error = np.linalg.norm(y0)

    # main loop
    for i in range(maximum_iterations):
        d = -J_inverse*y0
        x = x0 + d
        y = F(x)
        dF = y - y0
        J_inverse = J_inverse + ((d - J_inverse*dF)/(dF.T*dF))*dF.T
        x0 = x
        y0 = y
        error = np.linalg.norm(y)
        if error < tolerance:
            return x, error, i
    
    return x, error, i