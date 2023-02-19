import numpy as np
import cv2
from holotile import *
from pathlib import Path
import matplotlib.pyplot as plt

# PATHS
LOGO_PATH = Path() / "demo targets" / "logo.png"
WAVELENGTH = 532E-9
FOCAL_LENGTH = 150E-3
SLM_PIXEL_SIZE = 8.0E-6
SLM_N = 1080
SLM_DIM = SLM_PIXEL_SIZE * SLM_N
INPUT_BEAM_HALF_WIDTH = 3.5E-3

slm_enable = False
cam_enable = False

if slm_enable:
    from slm import SLM
    slm = SLM()

if cam_enable:
    from camera.edmundcam import EdmundCamera
    cam = EdmundCamera(exposure_time=.508)

slm_x, slm_y = np.meshgrid(np.linspace(-SLM_DIM / 2, SLM_DIM / 2, SLM_N),
                               np.linspace(-SLM_DIM / 2, SLM_DIM / 2, SLM_N))

def sdu_logo_demo():
    beamshaper = SquareTopHatShaper(WAVELENGTH, FOCAL_LENGTH, (slm_x, slm_y))

    holotile = HoloTile.from_tile_number(WAVELENGTH, FOCAL_LENGTH, INPUT_BEAM_HALF_WIDTH, (slm_x, slm_y), beam_shaping=beamshaper, n_tiles=18)
    logo_image = cv2.imread(str(LOGO_PATH), -1)
    logo_image = logo_image[:, :, 0]
    logo_image = logo_image.astype(np.float64) / 255.0
    hologram, tiled, psf, target = holotile.generate(logo_image)

    fig, ax = plt.subplots(3,3, figsize=(6,6), sharex="all", sharey="all")
    source_cut_fig, source_cut_ax = plt.subplots(3,3, figsize=(6,6), sharex='all', sharey='all')
    source_cut_ax = source_cut_ax.flatten()
    ax = ax.flatten()
    s_slm_x, s_slm_y = np.meshgrid(np.linspace(-SLM_DIM, SLM_DIM, SLM_N*2),
                               np.linspace(-SLM_DIM, SLM_DIM, SLM_N*2))
    r = np.sqrt(s_slm_x**2 + s_slm_y**2)
    noise = np.random.random(s_slm_x.shape)/5 * gaussiansource2(s_slm_x, s_slm_y, R=3.5E-3)
    for cut_off, axis, source_cut_axis in zip(np.linspace(1E-3, 8E-3, 9), ax, source_cut_ax):
        source = gaussiansource2(s_slm_x, s_slm_y, R=3.5E-3) + noise
        source[np.sqrt((s_slm_x + 0.4E-3) ** 2 + s_slm_y ** 2) < 0.1E-3] = 0.2
        source[np.sqrt(s_slm_x ** 2 + s_slm_y**2) > cut_off] = 0
        source_cut_axis.plot(s_slm_x[0, :], source[SLM_N, :])

        recon = holotile.reconstruct(hologram, method="ASM", source=source)

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
        else:
            #fig, ax = plt.subplots(figsize=(6, 6))
            axis.imshow(recon, cmap='hot', vmax=200)
            axis.set_title(f"{cut_off*1E3: .2f}")
    plt.show()


if __name__ == '__main__':
    # SDU Logo
    sdu_logo_demo()
