import numpy as np
try:
    import pyfftw
except:
    pass
try:
    import torch
except:
    pass

backend = np
if torch.cuda.is_available():
    cuda = torch.device('cuda')
    backend = torch
    fft_backend = torch.fft
    print("Using torch backend")
else:
    try:
        fft_backend = pyfftw.interfaces.numpy_fft
        print("Using pyfftw backend")
    except:
        print("Tried to use pyFFTW for fast Fourier transforms, but library could not be found. Using Numpy instead.")
        fft_backend = np.fft


def propagate(source: np.ndarray, wl, z, nx, ny, width, height):
    """
        :param source: 2D image to be propagated. If 3D propagation is desired, simply change z to an array of z distances
        :param wl: Wavelength used in meters
        :param z: Propagation distance(s) in meters
        :return: Complex field after Propagator distance z
    """
    gpu = torch.cuda.is_available()
    return ASM(source, wl, z, nx, ny, width, height, acc=gpu)

def ASM(source: np.ndarray, wl, z, nx, ny, width, height, acc=False):
    if acc:
        return ASM_cuda(source, wl, z, nx, ny, width, height)

    dx = width / nx
    dy = height / ny

    k = 2*backend.pi / wl
    # compute angular spectrum
    fft_c = fft_backend.fft2(source)
    c = fft_backend.fftshift(fft_c)

    fx = fft_backend.fftshift(fft_backend.fftfreq(nx, d=dx))
    fy = fft_backend.fftshift(fft_backend.fftfreq(ny, d=dy))
    #fx = wl/width * np.linspace(-nx/2, nx/2, nx)
    #fy = wl/height * np.linspace(-ny/2, ny/2, ny)
    fxx, fyy = backend.meshgrid(fx, fy, indexing='xy')
    argument = (2 * backend.pi) ** 2 * ((1. / wl) ** 2 - fxx ** 2 - fyy ** 2)

    # Calculate the propagating and the evanescent (complex) modes
    tmp = backend.sqrt(backend.abs(argument))
    kz = backend.where(argument >= 0, tmp, 1j * tmp)
    E = fft_backend.ifft2(fft_backend.ifftshift(c * backend.exp(1j * kz * z)))
    return E * np.exp(-1j * k * z)

def ASM_cuda(source, wl, z, nx, ny, width, height):
    source = torch.from_numpy(source).to(cuda)
    if isinstance(z, float):
        z = np.array([z])

    dx = width / nx
    dy = height / ny

    # compute angular spectrum
    fft_c = fft_backend.fft2(source)
    c = fft_backend.fftshift(fft_c)

    fx = fft_backend.fftshift(fft_backend.fftfreq(nx, d=dx)).type(torch.DoubleTensor)
    fy = fft_backend.fftshift(fft_backend.fftfreq(ny, d=dy)).type(torch.DoubleTensor)
    fxx, fyy = backend.meshgrid(fx, fy, indexing='xy')

    argument = (2 * backend.pi) ** 2 * ((1. / wl) ** 2 - fxx ** 2 - fyy ** 2)

    # Calculate the propagating and the evanescent (complex) modes
    tmp = backend.sqrt(backend.abs(argument))
    kz = backend.where(argument >= 0, tmp, 1j * tmp)

    kzs = backend.zeros_like(kz)
    kzs = backend.unsqueeze(kzs, -1)
    kzs = backend.repeat_interleave(kzs, len(z), dim=-1)
    for i in range(z.size):
        kzs[:, :, i] = kz * z[i]
    c = c[:, :, None]
    cs = c * backend.exp(1j * kzs.to(cuda))
    E = fft_backend.ifft2(fft_backend.ifftshift(cs, dim=(0, 1)), dim=(0,1))
    E = E.cpu().numpy() * np.exp(1j * 2*np.pi/wl * z)
    if len(E.shape) == 3:
        return E.squeeze()
    return E