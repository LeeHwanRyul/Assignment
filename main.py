import numpy as np
from FourierSeries import FourierSeries
import matplotlib.pyplot as plt

def SawtoothWave(t, amplitude, period):
    amplitude *= 2
    return amplitude * ((t % period) / period) - (amplitude / 2)

def PeriodicWave(t, amplitude, period):
    try:
        y = np.zeros_like(t)
        for i in range(len(t)):
            if (t[i] % period) < (period / 2):
                y[i] = amplitude
            else:
                y[i] = -amplitude
    except:
        y = 0
        if (t % period) < (period / 2):
            y = amplitude
        else:
            y = -amplitude
    return y

if __name__ == "__main__":
    """
    T = 0.02
    dx = 0.0001
    t = np.arange(0, T * 5, dx)
    amp = 1

    x = SawtoothWave(t, amp, T)
    """

    T = 20
    dx = 0.0001
    t = np.arange(0, T * 5, dx)
    amp = 2

    x = PeriodicWave(t, amp, T)

    NList = [4, 5, 6, 7]
    eulerList = [5, 7, 10, 20]

    fig, axs = plt.subplots(len(NList), len(eulerList), figsize=(20, 12))
    fig.suptitle("Fourier Series")

    xFourier = [[0 for col in range(len(eulerList))] for row in range(len(NList))]

    for j in range(len(NList)):
        for i in range(len(eulerList)):
            ax = axs[j][i]
            xFourier[j][i] = FourierSeries(t, NList[j], PeriodicWave, amp, T, eulerList[i])

            ax.plot(t, x, label="Original")
            ax.plot(t, xFourier[j][i], label="Fourier", linestyle='--')

            ax.set_title(f"N={NList[j]}, eulerCoef={eulerList[i]}", fontsize=10)
            ax.set_xlim(0, T * 5)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2)
    plt.show()