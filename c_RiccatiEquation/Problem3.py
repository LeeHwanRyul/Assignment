import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from NumericalAnalysis import EulerMethod, HeunMethod, RungeKutta2ndMethod
def HighODE(t, X):
    x, x1, x2, x3 = X
    dx1 = x1
    dx2 = x2
    dx3 = x3
    dx4 = -(x3 + 5 * x2 + 7 * x1 + 9 * x + 6) / 2
    return np.array([dx1, dx2, dx3, dx4])

if __name__ == "__main__":
    t = np.linspace(0, 1, 1000)

    x_0 = [0.1, 0.2, 0.3, 0.5]

    x_euler = EulerMethod(HighODE, t, x_0)
    x_heun = HeunMethod(HighODE, t, x_0)
    x_RK2 = RungeKutta2ndMethod(HighODE, t, x_0)

    sol = solve_ivp(HighODE, [t[0], t[-1]], x_0, t_eval=t)

    validLen = len(sol.t)

    for i in range(4):
        x_ref = sol.y[i]
        tValid = sol.t

        x_euler = x_euler[:validLen]
        x_heun = x_heun[:validLen]
        x_RK2 = x_RK2[:validLen]

        errorEuler = np.abs(x_ref - x_euler[:, i])
        errorHeun = np.abs(x_ref - x_heun[:, i])
        errorRK2 = np.abs(x_ref - x_RK2[:, i])

        print(f"{i+1}th Max Error (Euler) :", np.max(errorEuler))
        print(f"{i+1}th Max Error (Heun) :", np.max(errorHeun))
        print(f"{i+1}th Max Error (RK2) :", np.max(errorRK2))
        print()
