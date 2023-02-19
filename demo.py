import numpy as np
import cv2
from datetime import datetime
from holotile import *
from pathlib import Path
import matplotlib.pyplot as plt

# PATHS
LOGO_PATH = Path() / "demo targets" / "logo.png"


#------------ USER SETTINGS ------------#
WAVELENGTH = 532E-9
FOCAL_LENGTH = 150E-3
SLM_PIXEL_SIZE = 8.0E-6
SLM_N = 1080
SLM_DIM = SLM_PIXEL_SIZE * SLM_N
INPUT_BEAM_HALF_WIDTH = 2.5E-3
#------------ USER SETTINGS ------------#


slm_x, slm_y = np.meshgrid(np.linspace(-SLM_DIM / 2, SLM_DIM / 2, SLM_N),
                               np.linspace(-SLM_DIM / 2, SLM_DIM / 2, SLM_N))

def logo_demo():
    beamshaper = SquareTopHatShaper(WAVELENGTH, FOCAL_LENGTH, (slm_x, slm_y))

    holotile = HoloTile.from_tile_number(WAVELENGTH, FOCAL_LENGTH, INPUT_BEAM_HALF_WIDTH, (slm_x, slm_y), beam_shaping=beamshaper, n_tiles=18)
    #holotile.translate(-3E-3, 0, 0)
    logo_image = cv2.imread(str(LOGO_PATH), -1)
    logo_image = logo_image[:, :, 0]
    logo_image = logo_image.astype(np.float64) / 255.0
    hologram, tiled, psf, target = holotile.generate(logo_image)
    recon = holotile.reconstruct(hologram)

    fig, ax = plt.subplots(2,2, figsize=(6,6))
    ax = ax.flatten()

    ax[0].imshow(tiled, cmap='gray')
    ax[1].imshow(psf, cmap='gray')
    ax[2].imshow(hologram, cmap='gray')
    ax[3].imshow(recon, cmap='gray')

    plt.show()

    # Uncomment to save hologram as phase values [0-2pi]
    #np.save(f"saved_holograms/HoloTile_hologram_{datetime.now().strftime('%d-%m-%Y %H:%M:%S')}.npy", hologram, allow_pickle=False)


def tiling_demo():
    beamshaper = SquareTopHatShaper(WAVELENGTH, FOCAL_LENGTH, (slm_x, slm_y))
    logo_image = cv2.imread(str(LOGO_PATH), -1)
    logo_image = logo_image[:, :, 0]
    logo_image = logo_image.astype(np.float64) / 255.0
    Nts = [15, 18, 22, 30]

    fig, ax = plt.subplots(2, 2, figsize=(6, 6))
    ax = ax.flatten()
    for i, Nt in enumerate(Nts):
        holotile = HoloTile.from_tile_number(WAVELENGTH, FOCAL_LENGTH, INPUT_BEAM_HALF_WIDTH, (slm_x, slm_y), beam_shaping=beamshaper, n_tiles=Nt)
        #holotile.translate(-3E-3, 0, 0)

        hologram, tiled, psf, target = holotile.generate(logo_image)
        recon = holotile.reconstruct(hologram)
        ax[i].imshow(recon, cmap='gray')
    plt.show()




if __name__ == '__main__':
    logo_demo()

    #tiling_demo()
