import numpy as np
from FourierSeries import FourierSeries
import matplotlib.pyplot as plt

def SawtoothWave(t, amplitude, period=1):
    amplitude *= 2
    return amplitude * ((t % period) / period) - (amplitude / 2)

def PeriodicWave(t, amplitude, period=1):
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

def CustomFunction(t, amplitude, period=1):
    return 3*np.sin(2*np.pi*t) + 2*np.cos(6*np.pi*t) - 0.5*np.sin(10*np.pi*t)

if __name__ == "__main__":
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
    """

    NList = [10, 100]
    eulerList = [125, 150, 200]

    xFourier = [[0 for col in range(len(eulerList))] for row in range(len(NList))]
    CList = [[0 for col in range(len(eulerList))] for row in range(len(NList))]

    for j in range(len(NList)):
        for i in range(len(eulerList)):
            # xFourier: 퓨리에 시리즈로 근사화된 x(t)
            # CList: Fourier Coefficient
            xFourier[j][i], CList[j][i] = FourierSeries(t, NList[j], SawtoothWave, amp, T, eulerList[i])

    # plot task1, task2, task3
    fig_magnitude, axs_mag = plt.subplots(len(NList), len(eulerList), figsize=(18, 10))
    fig_phase, axs_phase = plt.subplots(len(NList), len(eulerList), figsize=(18, 10))
    fig_reconstruct, axs_recon = plt.subplots(len(NList), len(eulerList), figsize=(18, 10))

    fig_magnitude.suptitle("Amplitude of Fourier Coefficients")
    fig_phase.suptitle("Angle of Fourier Coefficients")
    fig_reconstruct.suptitle("Fourier Series Approximation vs Original")

    for j in range(len(NList)):
        for i in range(len(eulerList)):
            n_vals = np.arange(-NList[j], NList[j] + 1)

            # Magnitude plot
            axs_mag[j, i].stem(n_vals, np.abs(CList[j][i]))

            # Phase plot
            axs_phase[j, i].stem(n_vals, np.angle(CList[j][i]))

            # Approximation plot
            axs_recon[j, i].plot(t, x, label="Original")
            axs_recon[j, i].plot(t, xFourier[j][i], label="Fourier", linestyle='--')
            axs_recon[j, i].set_xlim(0, T * 5)
            axs_recon[j, i].legend()

    plt.tight_layout()
    plt.show()