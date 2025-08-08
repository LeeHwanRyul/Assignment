import numpy as np

def Space(t, initialCondition):
    try:
        x = np.zeros_like(t)

        x[0] = initialCondition
    except:
        n = len(t)
        dim = len(initialCondition)
        x = np.zeros((n, dim))
        x[0] = initialCondition

    return x

def EulerMethod(func, t, initialCondition):
    x = Space(t, initialCondition)

    for k in range(len(t) - 1):
        t_k = t[k]
        t_k1 = t[k + 1]

        dt = t_k1 - t_k

        x[k+1] = func(t_k, x[k]) * dt + x[k]
    return x

def HeunMethod(func, t, initialCondition):
    x = Space(t, initialCondition)

    for k in range(len(t) - 1):
        t_k = t[k]
        t_k1 = t[k + 1]
        dt = t_k1 - t_k

        k1 = func(t_k, x[k])
        x_predict = x[k] + dt * k1

        k2 = func(t_k1, x_predict)
        x[k + 1] = x[k] + dt / 2 * (k1 + k2)

    return x

def RungeKutta2ndMethod(func, t, initialCondition):
    x = Space(t, initialCondition)

    for k in range(len(t) - 1):
        t_k = t[k]
        t_k1 = t[k + 1]

        dt = t_k1 - t_k

        k1 = func(t_k, x[k])
        k2 = func(t_k + dt / 2, x[k] + dt / 2 * k1)

        x[k + 1] = x[k] + dt * k2
    return x