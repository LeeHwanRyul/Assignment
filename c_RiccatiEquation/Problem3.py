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
    t = np.linspace(0, 10, 1000)

    x_0 = [0.1, 0.2, 0.3, 0.5]

    x_euler = EulerMethod(HighODE, t, x_0)
    x_heun = HeunMethod(HighODE, t, x_0)
    x_RK2 = RungeKutta2ndMethod(HighODE, t, x_0)

    sol = solve_ivp(HighODE, [t[0], t[-1]], x_0, t_eval=t)

    validLen = len(sol.t)

    for i in range(4):
        x_ref = sol.y[i]

        x_euler_trim = x_euler[:validLen, i]
        x_heun_trim = x_heun[:validLen, i]
        x_RK2_trim = x_RK2[:validLen, i]

        plt.figure(figsize=(10, 5))
        plt.plot(t, x_ref, label='solve_ivp (True)', linewidth=2)
        plt.plot(t, x_euler_trim, '--', label='Euler')
        plt.plot(t, x_heun_trim, '--', label='Heun')
        plt.plot(t, x_RK2_trim, '--', label='RK2')
        plt.title(f'{i + 1}th Variable Comparison')
        plt.xlabel('Time')
        plt.ylabel(f'x[{i}]')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        errorEuler = np.abs(x_ref - x_euler_trim)
        errorHeun = np.abs(x_ref - x_heun_trim)
        errorRK2 = np.abs(x_ref - x_RK2_trim)

        print(f"{i+1}th Max Error (Euler) :", np.max(errorEuler))
        print(f"{i+1}th Max Error (Heun) :", np.max(errorHeun))
        print(f"{i+1}th Max Error (RK2) :", np.max(errorRK2))
        print()
