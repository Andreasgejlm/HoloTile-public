import numpy as np
import cv2
from holotile import *
from pathlib import Path
import matplotlib.pyplot as plt
import time
from PIL import Image

# PATHS
LOGO_PATH = Path() / "demo targets" / "logo.png"

WAVELENGTH = 532E-9
FOCAL_LENGTH = 150E-3
SLM_PIXEL_SIZE = 8.0E-6
SLM_N = 1080
SLM_DIM = SLM_PIXEL_SIZE * SLM_N
INPUT_BEAM_HALF_WIDTH = 3E-3

slm_enable = False
cam_enable = False

if slm_enable:
    from slm import SLM
    slm = SLM()

if cam_enable:
    from camera.edmundcam import EdmundCamera
    cam = EdmundCamera(exposure_time=17.308)

slm_x, slm_y = np.meshgrid(np.linspace(-SLM_DIM / 2, SLM_DIM / 2, SLM_N),
                               np.linspace(-SLM_DIM / 2, SLM_DIM / 2, SLM_N))

def sdu_logo_demo():
    beamshaper = SquareTopHatShaper(WAVELENGTH, FOCAL_LENGTH, (slm_x, slm_y))

    holotile = HoloTile.from_tile_number(WAVELENGTH, FOCAL_LENGTH, INPUT_BEAM_HALF_WIDTH, (slm_x, slm_y), beam_shaping=beamshaper, n_tiles=18)
    holotile.translate(-3E-3, 0, 0)
    logo_image = cv2.imread(str(LOGO_PATH), -1)
    logo_image = logo_image[:, :, 0]
    logo_image = logo_image.astype(np.float64) / 255.0
    hologram, tiled, psf, target = holotile.generate(logo_image)
    recon = holotile.reconstruct(hologram)

    if slm_enable:
        slm.show_phase(hologram)
        if cam_enable:
            fig, ax = plt.subplots(1,2, figsize=(10,6))
            ax = ax.flatten()
            ax[0].imshow(recon, cmap='gray')
            ax[1].imshow(cam.capture())
            plt.show()
        else:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(recon, cmap='gray')
            plt.show()
            plt.waitforbuttonpress()



def find_R():
    beamshaper = SquareTopHatShaper(WAVELENGTH, FOCAL_LENGTH, (slm_x, slm_y))
    fig, ax = plt.subplots(3, 3, sharex=True, sharey=True)
    ax = ax.flatten()
    for i, R in enumerate(np.linspace(1.5E-3, 6E-3, 9)):
        holotile = HoloTile.from_tile_number(WAVELENGTH, FOCAL_LENGTH, INPUT_BEAM_HALF_WIDTH, (slm_x, slm_y), beam_shaping=beamshaper, n_tiles=18)
        holotile.translate(-3E-3, 0, 0)
        holotile.R = R
        logo_image = cv2.imread(str(LOGO_PATH), -1)
        logo_image = logo_image[:, :, 0]
        logo_image = logo_image.astype(np.float64) / 255.0
        hologram, tiled, psf, target = holotile.generate(logo_image)
        if slm_enable:
            slm.show_phase(hologram)
        ax[i].imshow(cam.capture(), cmap='gray')
        ax[i].set_title(f"{R*1E3: .2f}")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # SDU Logo
    sdu_logo_demo()
    #find_R()