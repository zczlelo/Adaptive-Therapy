from broyden_methods import broyden_method_bad
import numpy as np



def func_1(x):
    return x

def func_2(x):
    return x**2

def func_3(x):
    return np.sin(x)

def func_4(x):
    return np.sin(x) + x**2

def func_5(x):
    return 2 * x - np.roll(x, 1) - np.roll(x, -1) + 1

if __name__ == "__main__":

    result_1 = broyden_method_bad(func_1, np.array([[1, 1, 1]]).T, None, 3)
    #result_2 = broyden_method_bad(func_2, np.array([1, 1, 1]))
    #result_3 = broyden_method_bad(func_3, np.array([1, 1, 1]))
    #result_4 = broyden_method_bad(func_4, np.array([1, 1, 1]))

    # print results
    print(result_1)
    #print(result_2)
    #print(result_3)
    #print(result_4)
