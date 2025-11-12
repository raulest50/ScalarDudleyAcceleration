import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftshift
from  NL_functions import hR_new, NLT, hR
import math  # Use standard library math module
import time  # For measuring execution time

# Para ejecutarse por celda 
# %matplotlib inline 

# Define a moving average function with edge handling
def moving_average(x, w):
    smoothed = np.convolve(x, np.ones(w), 'valid') / w
    # Pad the result to match the original length
    pad_left = (len(x) - len(smoothed)) // 2
    pad_right = len(x) - len(smoothed) - pad_left
    return np.pad(smoothed, (pad_left, pad_right), mode='edge')


# Define constants and parameters
c = 3 * 10**8  # Speed of light in m/s
l0 = 835 * 10**-9  # Wavelength in meters
f0 = c / l0 * 10**-12  # Frequency in pHz
w0 = 2 * np.pi * c / l0  # Angular frequency in rad/s
w0_shock = w0 * 1.0e-12  # Angular frequency in 1/ps
k0 = 2 * np.pi / l0  # Wave number in m^-1
gama = 0.11  # Nonlinear parameter in W^-1*m^-1
T0 = 0.0284  # Pulse width in ps
z = 0.15  # Distance in meters
fR = 0.18  # Raman response coefficient
Tshock = 1 / w0_shock  # Shock time in ps
P0 = 10000  # Initial power in Watts
nt = 2**13  # FFT points and window size
step_num = 1500  # Number of z steps
distance = z  # Propagation distance
t_mult = 150  # Time window multiple

# Dispersion parameters
beta = np.zeros(11)
beta[1] = -11.830 * 10**-3
beta[2] = 8.1038 * 10**-5
beta[3] = -9.5205 * 10**-8
beta[4] = 2.0737 * 10**-10
beta[5] = -5.3943 * 10**-13
beta[6] = 1.3486 * 10**-15
beta[7] = -2.5495 * 10**-18
beta[8] = 3.0524 * 10**-21
beta[9] = -1.7140 * 10**-24

# Pulse shape parameter
mshape = 0
chirp0 = 0
m = mshape
c0 = chirp0

# Pulse width
if mshape == 0:
    t0_FWHM = 2 * np.log(1 + np.sqrt(2)) * T0
else:
    t0_FWHM = 2 * np.sqrt(np.log(2)) * T0

# Set simulation parameters
t_range = 0.5 * t_mult * t0_FWHM
tau = np.linspace(-t_range, t_range, nt + 1)
tau = tau[:-1]
dtau = tau[1] - tau[0]
fs = 1 / dtau
deltaz = distance / step_num
domega = 2 * np.pi / nt * fs
omega = np.concatenate([-2 * np.pi * fs + np.arange(nt//2, nt) * domega,
                        np.arange(nt//2) * domega])
f = omega / (2 * np.pi)

# Shift the frequency domain
omega = np.fft.fftshift(omega)

# Input Field profile, initial pulse
if mshape == 0:
    A = np.sqrt(P0) * np.cosh(tau / T0)**-1 * np.exp(-0.5j * chirp0 * (tau / T0)**2)
else:
    A = np.sqrt(P0) * np.exp(-0.5 * (1 + 1j * chirp0) * (tau / T0)**(2 * mshape))

A0 = A

# Plot input pulse shape and spectrum
plt.figure()
plt.plot(tau, np.abs(A)**2 / P0, '-r')
plt.axis([-.4, .4, 0, None])
plt.xlabel('Time (ps)')
plt.ylabel('Normalized Power')
plt.title('Input Pulse Shape')
plt.show()

# save initial pulse
plt.savefig('Input.png')
plt.close()



# Dispersion effects
dispbeta = 0
for v in range(2, 11):
    dispbeta += (beta[v-1] / math.factorial(v)) * omega**v

dispersion_half = np.exp(-1j * dispbeta * deltaz / 2)

# Raman Response in Fourier Domain
hR1 = np.array([hR(t) for t in tau])
f_R = ((1 - fR) + dtau * fR * fft(fftshift(hR1)))

# Simulation loop
A_shot = np.zeros((len(tau), step_num), dtype=complex)
P_t = np.zeros_like(A_shot)
P_w = np.zeros_like(A_shot)
distan = np.zeros(step_num)

# Start measuring propagation time
propagation_start_time = time.time()

for n in range(step_num):
    # First half dispersion
    A = ifft(fft(A) * dispersion_half)

    # Non-linear step using Runge-Kutta method
    k1 = NLT(A, omega, gama, Tshock, f_R)
    A_half2 = A + k1 * deltaz / 2
    k2 = NLT(A_half2, omega, gama, Tshock, f_R)
    A_half3 = A + k2 * deltaz / 2
    k3 = NLT(A_half3, omega, gama, Tshock, f_R)
    A_full = A + k3 * deltaz
    k4 = NLT(A_full, omega, gama, Tshock, f_R)
    A = A + deltaz / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    # Second half dispersion
    A = ifft(fft(A) * dispersion_half)

    # Storing results
    A_shot[:, n] = A
    P_t[:, n] = np.abs(A)**2 / P0
    P_w[:, n] = (dtau**2 / P0) * np.abs(fftshift(fft(A)))**2
    distan[n] = deltaz * n

# End measuring propagation time
propagation_end_time = time.time()
propagation_elapsed_time = propagation_end_time - propagation_start_time
print(f"Propagation execution time: {propagation_elapsed_time:.4f} seconds")

# Wavelength for plotting
l_lambda = c * 1e-3 / (f0 + f)


# Smooth the data using a moving average filter
window_size = 10  # Adjust this size for more or less smoothing

P_t_smoothed = np.zeros_like(P_t)
P_w_smoothed = np.zeros_like(P_w)

for i in range(P_t.shape[0]):
    P_t_smoothed[i, :] = moving_average(P_t[i, :], window_size)
    P_w_smoothed[i, :] = moving_average(P_w[i, :], window_size)

# Plotting the final pulse
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(l_lambda, 10 * np.log10((dtau**2 / P0) * np.abs(fftshift(fft(A)))**2 + 1e-10), '-b')  # Adding a small constant
plt.xlabel('Wavelength (nm)')
plt.ylabel('Power (dB)')
plt.title('Output - Spectrum')
plt.axis([450, 1500, -100, 10])

plt.subplot(2, 1, 2)
Pt = np.abs(A)**2 / P0
plt.plot(tau, Pt, '-b')
plt.xlabel('Time (ps)')
plt.ylabel('Normalized Power')
plt.title('Output - Shape')
plt.show()

# save final step plots
plt.savefig('final_step_scalar.png')
plt.close()

# Time Evolution Plot
plt.figure()
max_Pt_s = np.max(np.abs(P_t_smoothed))
plt.pcolormesh(distan, tau, 10 * np.log10(np.abs(P_t_smoothed) + 1e-10), shading='nearest', cmap='jet')  # Adding a small constant
plt.colorbar()
plt.clim(max_Pt_s - 28.0, max_Pt_s)
plt.ylim([-1, 3])
plt.xlabel('Distance (m)')
plt.ylabel('Time (ps)')
plt.title('Time Evolution')
# plt.show()

# save time evolution
plt.savefig('time_evolution_scalar.png')
plt.close()



# Spectral Evolution Plot
plt.figure()
max_Pw_s = np.max(np.abs(P_w_smoothed))
plt.pcolormesh(distan, l_lambda, 10 * np.log10(np.abs(P_w_smoothed) + 1e-10), shading='nearest', cmap='jet')  # Adding a small constant
plt.colorbar()
plt.clim(max_Pw_s - 100.0, max_Pw_s)
plt.ylim([320, 1400])
plt.xlabel('Distance (m)')
plt.ylabel('Wavelength (nm)')
plt.title('Spectral Evolution')
plt.show()

# Save Spectral evolution
plt.savefig('spectral_evolution_scalar.png')
plt.close()
