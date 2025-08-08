import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from NumericalAnalysis import EulerMethod, HeunMethod, RungeKutta2ndMethod
def RiccatiEquation(t, x):
    result = 2 * x - (x ** 2) / 2 + 1
    return result

if __name__ == "__main__":
    t = np.linspace(0, 10, 1000)

    x_0 = 1

    x_euler_trim = EulerMethod(RiccatiEquation, t, x_0)
    x_heun_trim = HeunMethod(RiccatiEquation, t, x_0)
    x_RK2_trim = RungeKutta2ndMethod(RiccatiEquation, t, x_0)

    sol = solve_ivp(RiccatiEquation, [t[0], t[-1]], [x_0], t_eval=t)
    x_ref = sol.y[0]

    plt.figure(figsize=(10, 5))
    plt.plot(t, x_ref, label='solve_ivp (True)', linewidth=2)
    plt.plot(t, x_euler_trim, '--', label='Euler')
    plt.plot(t, x_heun_trim, '--', label='Heun')
    plt.plot(t, x_RK2_trim, '--', label='RK2')
    plt.title(f'Method Comparison')
    plt.xlabel('Time')
    plt.ylabel(f'x')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    errorEuler = np.abs(x_ref - x_euler_trim)
    errorHeun = np.abs(x_ref - x_heun_trim)
    errorRK2 = np.abs(x_ref - x_RK2_trim)

    print("Max Error (Euler) :", np.max(errorEuler))
    print("Max Error (Heun) :", np.max(errorHeun))
    print("Max Error (RK2) :", np.max(errorRK2))
