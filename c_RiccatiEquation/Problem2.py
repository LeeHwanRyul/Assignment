import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from NumericalAnalysis import EulerMethod, HeunMethod, RungeKutta2ndMethod
def RiccatiEquation(t, x):
    result = 2 * x + (x ** 2) / 2 + 1
    return result

if __name__ == "__main__":
    t = np.linspace(0, 0.7, 1000)

    x_0 = 1

    x_euler = EulerMethod(RiccatiEquation, t, x_0)
    x_heun = HeunMethod(RiccatiEquation, t, x_0)
    x_RK2 = RungeKutta2ndMethod(RiccatiEquation, t, x_0)

    sol = solve_ivp(RiccatiEquation, [t[0], t[-1]], [x_0], t_eval=t)

    validLen = len(sol.t)

    x_ref = sol.y[0]
    tValid = sol.t

    x_euler = x_euler[:validLen]
    x_heun = x_heun[:validLen]
    x_RK2 = x_RK2[:validLen]

    errorEuler = np.abs(x_ref - x_euler)
    errorHeun = np.abs(x_ref - x_heun)
    errorRK2 = np.abs(x_ref - x_RK2)

    print("Max Error (Euler) :", np.max(errorEuler))
    print("Max Error (Heun) :", np.max(errorHeun))
    print("Max Error (RK2) :", np.max(errorRK2))
