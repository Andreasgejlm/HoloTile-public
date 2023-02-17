import numpy as np
import scipy.special as sp
from scipy.signal import convolve2d


def tile2(im, n):
    return np.tile(im, (n, n))

def beamshapegaussian2(xa, ya, R, D, wl, f):
    beta = 2 * np.sqrt(2 * np.pi) * R * D / (wl * f)
    R = R / np.sqrt(2)
    xi = xa / (R / np.sqrt(2))
    eta = ya / (R*6)
    phix = np.sqrt(np.pi) / 2 * xi * sp.erf(xi) + 1 / 2 * np.exp(-xi ** 2) - 1 / 2
    phiy = np.sqrt(np.pi) / 2 * eta * sp.erf(eta) + 1 / 2 * np.exp(-eta ** 2) - 1 / 2
    phi = beta * (phix + phiy)
    return phi

def gaussiansource2(x, y, R=1, ux=0, uy=0):
    R = R / np.sqrt(2)
    xi = (x - ux) / R
    eta = (y - uy) / R
    return np.exp(-(xi ** 2 + eta ** 2) / 2)

def norm(field: np.ndarray):
    f = field - np.min(field)
    f /= np.max(f)
    return f

def awgs(target, source, padding, n, dx, wl, f, iterations=50):
    M_old, N_old = np.shape(target)
    signalarea = slice(padding, padding + M_old - 1)
    padded_target = np.pad(target, padding)
    [M_new, N_new] = np.shape(padded_target)
    p, q = np.meshgrid(np.linspace(-M_old // 2, M_old // 2, M_old), np.linspace(-N_old // 2, N_old // 2, N_old))
    At = norm(padded_target)
    fmax = M_old * dx / (2 * wl * f)
    del_X = 1 / (fmax * 2)
    fx, fy = np.meshgrid(np.repeat(np.linspace(-fmax, fmax, int(n)), M_old // n), np.repeat(np.linspace(-fmax, fmax, int(n)), M_old // n))

    initial_phase = 2 * np.pi * (del_X * p * fx + del_X * q * fy)
    padded_initial_phase = np.pad(initial_phase, padding)
    padded_initial_phase = np.exp(1j * padded_initial_phase)

    mask = np.ones([2, 2])
    mask /= np.sum(mask)

    A = At * padded_initial_phase
    A = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(A)))
    for ii in range(iterations):
        A = convolve2d(np.angle(A), mask, mode='same')
        B = source * np.exp(1j * A)
        C = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(B)) / (M_new * N_new))
        Ar = norm(abs(C))
        wpro = np.exp((At[signalarea, signalarea] - Ar[signalarea, signalarea]))
        Acon = At[signalarea, signalarea] * wpro
        D = Ar * np.exp(1j * np.angle(C))
        D[signalarea, signalarea] = Acon * np.exp(1j * np.angle(C[signalarea, signalarea]))
        A = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(D)))
    return np.angle(A)