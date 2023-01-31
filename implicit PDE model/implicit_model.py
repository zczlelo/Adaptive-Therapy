import numpy as np


def set_rhs(parameters):
    diffusion = set_diffusion(parameters)
    def rhs_f(u, t):
        N = len(u) - 1
        rhs = zeros(N+1)
        rhs[0] = dsdt(t)
        for i in range(1, N):
            rhs[i] = (beta/dx**2)*(u[i+1] - 2*u[i] + u[i-1]) + \
                     g(x[i], t)
        rhs[N] = (beta/dx**2)*(2*u[i-1] + 2*dx*dudx(t) -
                               2*u[i]) + g(x[N], t)
        return rhs
    return rhs_f


def set_jacobian(parameters):

    return None

def rhs(u, t):
    N = len(u) - 1
    rhs = zeros(N+1)
    rhs[0] = dsdt(t)
    for i in range(1, N):
        rhs[i] = (beta/dx**2)*(u[i+1] - 2*u[i] + u[i-1]) + \
                 g(x[i], t)
    rhs[N] = (beta/dx**2)*(2*u[i-1] + 2*dx*dudx(t) -
                           2*u[i]) + g(x[N], t)
    return rhs

def K(u, t):
    N = len(u) - 1
    K = zeros((N+1,N+1))
    K[0,0] = 0
    for i in range(1, N):
        K[i,i-1] = beta/dx**2
        K[i,i] = -2*beta/dx**2
        K[i,i+1] = beta/dx**2
    K[N,N-1] = (beta/dx**2)*2
    K[N,N] = (beta/dx**2)*(-2)
    return K