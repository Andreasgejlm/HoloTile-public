from typing import Literal
import cv2
import numpy as np
from .propagate import propagate
from .utils import awgs, tile2, gaussiansource2
from .beamshaper import *


class HoloTile:
    def __init__(self, wavelength: float, focal_length: float, input_beam_half_width: float, slm_coords: tuple[np.ndarray, np.ndarray], beam_shaping: BeamShaper):
        self.gs_padding = 5
        self.wavelength = wavelength
        self.focal_length = focal_length
        self.slm_x, self.slm_y = slm_coords
        self.slm_N, self.slm_M = np.shape(self.slm_x)
        self.slm_height, self.slm_width = np.max(self.slm_y[:, 0]*2), np.max(self.slm_x[0, :]*2)
        self.slm_pixel_size = self.slm_height / self.slm_N
        self._R = input_beam_half_width
        self._D = None
        self.beam_shaper = beam_shaping
        self.x_offset = self.y_offset = self.z_offset = 0


    # ------------- Convenience Initializers ------------- #
    @classmethod
    def from_output_width(cls, wavelength: float, focal_length: float, input_beam_half_width: float, slm_coords: tuple[np.ndarray, np.ndarray], half_width: float, beam_shaping: BeamShaper):
        c = cls(wavelength, focal_length, input_beam_half_width, slm_coords, beam_shaping)
        c.D = half_width
        return c

    @classmethod
    def from_tile_number(cls, wavelength: float, focal_length: float, input_beam_half_width: float, slm_coords: tuple[np.ndarray, np.ndarray], n_tiles: int, beam_shaping: BeamShaper):
        c = cls(wavelength, focal_length, input_beam_half_width, slm_coords, beam_shaping)
        c.Nt = n_tiles
        return c

    @classmethod
    def from_output_pixels(cls, wavelength: float, focal_length: float, input_beam_half_width: float, slm_coords: tuple[np.ndarray, np.ndarray], n_pixels: int, beam_shaping: BeamShaper):
        c = cls(wavelength, focal_length, input_beam_half_width, slm_coords, beam_shaping)
        image_res = n_pixels + 1
        c.Nt = int(np.floor(slm_coords[0].shape[0]/(image_res + 20)))
        return c

    # ------------- Computed Properties and Setters ------------- #
    @property
    def beam_shaper(self) -> BeamShaper:
        """
        :return: The chosen BeamShaper
        """
        return self._beam_shaper

    @beam_shaper.setter
    def beam_shaper(self, value: BeamShaper):
        """
        Set the BeamShaper by calling:
        holotile = HoloTile(...)
        holotile.beam_shaper = BeamShaper(...)
        Updates the associated values when called
        :param value: Instance of BeamShaper class
        :return: None
        """
        self._beam_shaper = value
        self.update_beamshaping()

    @property
    def D(self):
        """
        :return: The chosen output pixel width
        """
        return self._D

    @D.setter
    def D(self, D: float):
        """
        Set the output pixel width by calling:
        holotile = HoloTile(...)
        holotile.D = 2.3E-6 (or other)
        Updates the associated values when called
        :param value: Output pixel half width [mm, float]
        :return: None
        """
        self._D = D
        self._Nt = int(self.D * (2 * self.slm_height) / (self.wavelength * self.focal_length))
        self.holo_res = int(np.floor(self.slm_N / self.Nt))
        self.update_beamshaping()

    @property
    def Nt(self):
        """
        :return: The chosen number of tiles on SLM
        """
        return self._Nt

    @Nt.setter
    def Nt(self, Nt: int):
        """
        Set the tile number by:
        holotile = HoloTile(...)
        holotile.Nt = 12 (or other)
        Updates the associated values when called
        :param value: Tile number [int]
        :return: None
        """
        self._Nt = Nt
        self.holo_res = int(np.floor(self.slm_N / self.Nt))
        self._D = self.Nt * (self.wavelength * self.focal_length) / (2 * self.slm_height)
        self.update_beamshaping()

    @property
    def R(self):
        """
        :return: The input beam width
        """
        return self._R

    @R.setter
    def R(self, R: float):
        """
        Set the input beam width by:
        holotile = HoloTile(...)
        holotile.R = 1.2E-3 (or other)
        Updates the associated values when called
        :param value: Input beam 1/e^2 half width [float]
        :return: None
        """
        self._R = R
        self.update_beamshaping()

    @property
    def holo_res(self):
        return self._holo_res

    @holo_res.setter
    def holo_res(self, res: int):
        if np.mod(res, 2) != 0:
            res -= 1
        self._holo_res = res

    @property
    def image_res(self):
        return int(self.holo_res - self.gs_padding * 2), int(self.holo_res - self.gs_padding * 2)


    # ------------- Class Methods ------------- #

    def update_beamshaping(self):
        """
        Updates the beam shaping phase according to newly set Nt, D, or R.
        :return: None
        """
        if self.D is not None and self.R is not None and self.beam_shaper is not None:
            self.psf_phi = self.beam_shaper.update(self.R, self.D)

    def generate(self, target_img, iterations=50, axial_offset_mm: float = 0.0, angle: float = 0.0, precalculated_tiled_hologram: np.ndarray=None):
        """
        Main function of HoloTile. Generates HoloTile hologram from target_img image.
        :return: Stacked HoloTile hologram, tiled hologram (w/o PSF shaping), and the PSF shaping phase
        """
        target_hologram = None
        if precalculated_tiled_hologram is None:
            target_img = cv2.resize(target_img, self.image_res).astype(float)
            target_hologram = self._gs(target_img, iterations)
            target_hologram += np.pi
            tiled_hologram = tile2(target_hologram, int(self.Nt))
        else:
            tiled_hologram = precalculated_tiled_hologram

        while np.shape(tiled_hologram) != np.shape(self.psf_phi):
            dx = self.slm_M - tiled_hologram.shape[1]
            dy = self.slm_N - tiled_hologram.shape[0]
            if dx > 1 and dy > 1:
                tiled_hologram = np.pad(tiled_hologram, ((int(dy / 2), int(dy / 2)), (int(dx / 2), int(dx / 2))))
            else:
                if dx == 1:
                    tiled_hologram = np.pad(tiled_hologram, ((0, 0), (0, dx)))
                if dy == 1:
                    tiled_hologram = np.pad(tiled_hologram, ((0, dy), (0, 0)))
        stacked_hologram = np.mod(tiled_hologram + self.psf_phi, np.pi * 2)
        return self._translate(stacked_hologram), self._translate(tiled_hologram), self._translate(self.psf_phi), target_hologram

    def reconstruct(self, hologram, source=None, pad=True):
        """
        Reconstructs hologram in the reconstruction plane to create ground truth. Source illumination can be specified.
        Default is a Gaussian Source of self.R input beam width.
        :return: Reconstruction [np.ndarray[float]]
        """
        if source is None:
            source = gaussiansource2(self.slm_x, self.slm_y, self.R)
        else:
            assert source.shape == (self.slm_N, self.slm_M), "Source should have same dimensions as SLM"
        if not pad:
            slm = source.astype(np.complex128)
            slm *= np.exp(1j * hologram)
            M, N, width, height = self.slm_M, self.slm_N, self.slm_width, self.slm_height
        else:
            slm = np.pad(source, int(self.slm_N / 2)).astype(np.complex128)
            slm *= np.exp(1j * np.pad(hologram, int(self.slm_N / 2)))
            M, N, width, height = 2*self.slm_M, 2*self.slm_N, 2*self.slm_width, 2*self.slm_height

        recon = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(slm)))

        recon = np.abs(recon) ** 2
        recon -= np.min(recon)
        recon /= np.max(recon)
        return recon

    def translate(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x_offset = -x
        self.y_offset = -y
        self.z_offset = z


    # ------------- Private Class Methods ------------- #

    def _gs(self, target_img, iterations):
        n = (self.holo_res - self.gs_padding * 2) / 2
        holo_source = np.ones([self.holo_res, self.holo_res])
        return awgs(target_img,
                    holo_source,
                    self.gs_padding,
                    n,
                    self.slm_pixel_size,
                    self.wavelength,
                    self.focal_length,
                    iterations)

    def _translate(self, hologram: np.ndarray) -> np.ndarray:
        phi_x = phi_y = phi_z = 0
        x, y = self.slm_x, self.slm_y
        if np.abs(self.x_offset) > 0:
            lx = self.wavelength * self.focal_length / self.x_offset
            phi_x = 2 * np.pi * x / lx
        if np.abs(self.y_offset) > 0:
            ly = self.wavelength * self.focal_length / self.y_offset
            phi_y = 2 * np.pi * y / ly
        if np.abs(self.z_offset) > 0:
            lz = self.focal_length ** 2 / self.z_offset
            phi_z = 2 * np.pi * (x ** 2 + y ** 2) / (2 * lz * self.wavelength)
        return np.mod(hologram + phi_x + phi_y + phi_z, 2 * np.pi)