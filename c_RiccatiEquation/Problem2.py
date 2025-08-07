import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from NumericalAnalysis import EulerMethod, HeunMethod, RungeKutta2ndMethod
def RiccatiEquation(t, x):
    result = 2 * x - (x ** 2) / 2 + 1
    return result

if __name__ == "__main__":
    t = np.linspace(0, 1, 1000)

    x_0 = 1

    x_euler = EulerMethod(RiccatiEquation, t, x_0)
    x_heun = HeunMethod(RiccatiEquation, t, x_0)
    x_RK2 = RungeKutta2ndMethod(RiccatiEquation, t, x_0)

    sol = solve_ivp(RiccatiEquation, [t[0], t[-1]], [x_0], t_eval=t)
    x_ref = sol.y[0]

    errorEuler = np.abs(x_ref - x_euler)
    errorHeun = np.abs(x_ref - x_heun)
    errorRK2 = np.abs(x_ref - x_RK2)

    plt.plot(t, x_ref, label='solve_ivp (accurate)')
    plt.plot(t, x_euler, '--', label='Euler Method')
    plt.plot(t, x_heun, '--', label='Heun Method')
    plt.plot(t, x_RK2, '--', label='RK2 Method')
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("Max Error (Euler) :", np.max(errorEuler))
    print("Max Error (Heun) :", np.max(errorHeun))
    print("Max Error (RK2) :", np.max(errorRK2))
