"""FFT backend.

The original code used pyFFTW builders with 12 threads. The default backend
reproduces those calls exactly so results match the archived runs to round-off.
If pyFFTW is not installed, NumPy's FFT is used instead; results then differ at
machine-precision level only.
"""

import numpy as np

try:
    import pyfftw
except ImportError:  # pragma: no cover - exercised only without pyfftw
    pyfftw = None

AXES = (0, 1, 2)


class FFT:
    """3-D FFTs over the (x, y, z) axes of the simulation fields.

    Parameters
    ----------
    shape : tuple of int
        Grid shape ``(Nx, Ny, Nz)``.
    threads : int
        pyFFTW threads (12 in the original runs).
    backend : {"pyfftw", "numpy", "auto"}
        ``"auto"`` selects pyFFTW when available.
    """

    def __init__(self, shape, threads=12, backend="auto"):
        if backend == "auto":
            backend = "pyfftw" if pyfftw is not None else "numpy"
        if backend == "pyfftw" and pyfftw is None:
            raise ImportError("pyfftw is not installed; use backend='numpy'")
        if backend not in ("pyfftw", "numpy"):
            raise ValueError(f"unknown FFT backend {backend!r}")
        self.backend = backend
        self.threads = threads
        if backend == "pyfftw":
            # Pre-planned transforms, used by the elasticity solver.
            a = pyfftw.empty_aligned(shape, dtype="complex128")
            b = pyfftw.empty_aligned(shape, dtype="complex128")
            self._fft_planned = pyfftw.builders.fftn(a, axes=AXES, threads=threads)
            self._ifft_planned = pyfftw.builders.ifftn(b, axes=AXES, threads=threads)

    def fft(self, a):
        """Forward transform, planned for the input array (original ``fft_``)."""
        if self.backend == "numpy":
            return np.fft.fftn(a, axes=AXES)
        return pyfftw.builders.fftn(a, axes=AXES, threads=self.threads)()

    def ifft(self, a):
        """Inverse transform, planned for the input array (original ``ifft_``)."""
        if self.backend == "numpy":
            return np.fft.ifftn(a, axes=AXES)
        return pyfftw.builders.ifftn(a, axes=AXES, threads=self.threads)()

    def fft_planned(self, a):
        """Forward transform through the pre-planned object (elasticity solver).

        The returned array may be pyFFTW's internal buffer: copy it before the
        next call.
        """
        if self.backend == "numpy":
            return np.fft.fftn(a, axes=AXES)
        return self._fft_planned(a)

    def ifft_planned(self, a):
        """Inverse transform through the pre-planned object (elasticity solver)."""
        if self.backend == "numpy":
            return np.fft.ifftn(a, axes=AXES)
        return self._ifft_planned(a)
