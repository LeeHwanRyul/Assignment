import numpy as np

def EulerMethod(func, t, initialCondition):
    x = np.zeros_like(t)

    x[0] = initialCondition

    for k in range(len(t) - 1):
        t_k = t[k]
        t_k1 = t[k + 1]

        dt = t_k1 - t_k

        x[k+1] = func(0, x[k]) * dt + x[k]
    return x

def HeunMethod():
    return 0

def RungeKutta2ndMethod(func, t, initialCondition):
    return