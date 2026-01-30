"""
FPCS - Feature-Preserving Compensated Sampling

A high-performance time series downsampling algorithm that preserves
important visual features (peaks, valleys) while reducing data size.
"""

__version__ = "1.0.0"

# Try to import optimized Cython implementation first
try:
    from .fpcs_cy import downsample, downsample_into, downsample_batch
    _BACKEND = "cython"
except ImportError:
    # Fall back to pure Python implementation
    from .fpcs_pure import downsample
    _BACKEND = "python"

    # These are not available in pure Python
    downsample_into = None
    downsample_batch = None

# Always available from pure Python
from .fpcs_pure import FPCSDownsampler

__all__ = [
    "downsample",
    "downsample_into",
    "downsample_batch",
    "FPCSDownsampler",
    "_BACKEND",
    "__version__",
]


def get_backend():
    """Return the active backend: 'cython' or 'python'."""
    return _BACKEND
