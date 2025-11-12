import numpy as np
from scipy.fft import fft, ifft, fftshift

def hR(x):
    if x < 0:
        return 0
    else:
        tau11 = 0.0122  # ps
        tau22 = 0.032  # ps
        return ((tau11**2 + tau22**2) / (tau11 * tau22**2)) * np.exp(-x / tau22) * np.sin(x / tau11)

def hR_new(x):
    if x < 0:
        return 0
    else:
        f_b = 0.21
        tau_b = 0.096  # ps
        return (1 - f_b) * hR(x) + f_b * ((2 * tau_b - x) / (tau_b**2)) * np.exp(-x / tau_b)

def NLT(u, omega, gama, Tshock, f_R):
    conR_A2 = fftshift(ifft(f_R * fft(fftshift(np.abs(u)**2))))
    return -1j * gama * u * conR_A2 - gama * Tshock * fftshift(ifft(1j * omega * fft(fftshift(u * conR_A2))))

def A_sech(P0, T0, tau):
    return np.sqrt(P0) / np.cosh(tau / T0)

