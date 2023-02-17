from abc import ABC
import numpy as np
import scipy.special as sp


class BeamShaper(ABC):
    def __init__(self, wavelength: float, focal_length: float, slm_coords: tuple[np.ndarray, np.ndarray]):
        self.wavelength = wavelength
        self.focal_length = focal_length
        self.slm_coords = slm_coords
        self.angle = 0.0
        self.x_offset = 0.0
        self.y_offset = 0.0
        self.z_offset = 0.0
        self.slm_angle = 0.0
        self.axis_rotation_angle = 0.0
        self.rotation_axis = 'y'

    def update(self, R: float, D: float) -> np.ndarray:
        self.R = R
        self.D = D
        return self._update()

    def _update(self) -> np.ndarray:
        return np.zeros(1)

    def __add__(self, other):
        return self._update() + other._update()


class SquareTopHatShaper(BeamShaper):
    def _update(self) -> np.ndarray:
        beta = 2 * np.sqrt(2 * np.pi) * self.R * self.D / (self.wavelength * self.focal_length)
        R = self.R / np.sqrt(2)
        x, y = self.slm_coords
        x *= np.cos(self.slm_angle)
        xi = x / R
        eta = y / R
        phix = np.sqrt(np.pi) / 2 * xi * sp.erf(xi) + 1 / 2 * np.exp(-xi ** 2) - 1 / 2
        phiy = np.sqrt(np.pi) / 2 * eta * sp.erf(eta) + 1 / 2 * np.exp(-eta ** 2) - 1 / 2
        phi = beta * (phix + phiy)
        return phi