# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: overflowcheck=False
# cython: infer_types=True
"""
FPCS (Feature-Preserving Compensated Sampling) Time Series Downsampling Algorithm

Highly optimized Cython implementation for maximum performance.

Optimizations applied:
- All Cython safety checks disabled (boundscheck, wraparound, etc.)
- GIL-free core loop with nogil
- Direct pointer arithmetic instead of memory views
- Inline helper functions
- memcpy for bulk data transfer
- Minimized branching in hot path
"""

cimport cython
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from libc.math cimport isnan
from cpython.mem cimport PyMem_Malloc, PyMem_Free
import numpy as np
cimport numpy as np

np.import_array()

# Type definitions for clarity
ctypedef double float64_t


cdef tuple _downsample_float64(
    const double[::1] x,
    const double[::1] y,
    int ratio,
):
    """Internal float64 implementation for downsample()."""
    cdef Py_ssize_t n = x.shape[0]

    if y.shape[0] != n:
        raise ValueError(f"x and y must have the same length, got {n} and {y.shape[0]}")

    if ratio < 1:
        raise ValueError(f"Sampling ratio must be >= 1, got {ratio}")

    if n == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    if n == 1:
        return np.asarray(x).copy(), np.asarray(y).copy()

    if ratio == 1:
        return np.asarray(x).copy(), np.asarray(y).copy()

    cdef Py_ssize_t capacity = _output_capacity(n, ratio)
    cdef double* out_x = <double*>malloc(capacity * sizeof(double))
    cdef double* out_y = <double*>malloc(capacity * sizeof(double))

    if out_x == NULL or out_y == NULL:
        if out_x != NULL:
            free(out_x)
        if out_y != NULL:
            free(out_y)
        raise MemoryError("Failed to allocate output buffers")

    cdef Py_ssize_t out_count
    cdef const double* x_ptr = &x[0]
    cdef const double* y_ptr = &y[0]
    cdef np.ndarray[double, ndim=1, mode='c'] result_x, result_y

    try:
        with nogil:
            out_count = _downsample_core_ptr(x_ptr, y_ptr, n, ratio, out_x, out_y)

        result_x = np.empty(out_count, dtype=np.float64)
        result_y = np.empty(out_count, dtype=np.float64)
        memcpy(<void*>result_x.data, <void*>out_x, out_count * sizeof(double))
        memcpy(<void*>result_y.data, <void*>out_y, out_count * sizeof(double))
    finally:
        free(out_x)
        free(out_y)

    return result_x, result_y


def downsample(
    x,
    y,
    int ratio,
):
    """
    Downsample a time series using the FPCS algorithm (Cython implementation).

    Fixed windows count all input samples. Mixed windows ignore NaNs when
    selecting extrema; all-NaN windows retain one gap marker and reset
    compensation. Global endpoints are retained, even if NaN. With ratio=1,
    all points are retained.

    Parameters
    ----------
    x : array-like
        The x-coordinates (e.g., timestamps or indices). Integer inputs are supported.
    y : array-like
        The y-coordinates (e.g., values). Integer inputs are supported.
    ratio : int
        The window size R, including NaNs. Must be >= 1. Output is bounded by
        min(n, 2 * ceil(n / R) + 2), including global endpoints.

    Returns
    -------
    x_downsampled : np.ndarray
        The x-coordinates of retained points (float64).
    y_downsampled : np.ndarray
        The y-coordinates of retained points (float64).
    """
    cdef np.ndarray x_arr = np.ascontiguousarray(x, dtype=np.float64)
    cdef np.ndarray y_arr = np.ascontiguousarray(y, dtype=np.float64)
    cdef const double[::1] x_view = x_arr
    cdef const double[::1] y_view = y_arr

    return _downsample_float64(x_view, y_view, ratio)


cdef inline void _retain_index(
    const double* x, const double* y,
    Py_ssize_t index,
    double* out_x, double* out_y,
    Py_ssize_t* out_idx, Py_ssize_t* last_retained,
) noexcept nogil:
    """Retain each source sample at most once, in input order."""
    if index > last_retained[0]:
        out_x[out_idx[0]] = x[index]
        out_y[out_idx[0]] = y[index]
        out_idx[0] += 1
        last_retained[0] = index


cdef inline void _process_window_state(
    const double* x, const double* y,
    Py_ssize_t* min_idx, Py_ssize_t* max_idx, Py_ssize_t* potential_idx,
    int* previous_min_flag,
    double* out_x, double* out_y,
    Py_ssize_t* out_idx, Py_ssize_t* last_retained,
) noexcept nogil:
    """Retain the earlier extremum and carry the later one for compensation."""
    if min_idx[0] < max_idx[0]:
        if previous_min_flag[0] == 1 and potential_idx[0] >= 0:
            _retain_index(x, y, potential_idx[0], out_x, out_y, out_idx, last_retained)
        _retain_index(x, y, min_idx[0], out_x, out_y, out_idx, last_retained)
        potential_idx[0] = max_idx[0]
        min_idx[0] = max_idx[0]
        previous_min_flag[0] = 1
    else:
        if previous_min_flag[0] == 0 and potential_idx[0] >= 0:
            _retain_index(x, y, potential_idx[0], out_x, out_y, out_idx, last_retained)
        _retain_index(x, y, max_idx[0], out_x, out_y, out_idx, last_retained)
        potential_idx[0] = min_idx[0]
        max_idx[0] = min_idx[0]
        previous_min_flag[0] = 0


cdef inline Py_ssize_t _downsample_core_ptr(
    const double* x,
    const double* y,
    Py_ssize_t n,
    int ratio,
    double* out_x,
    double* out_y,
) noexcept nogil:
    """
    Downsample fixed windows, ignoring NaNs unless the whole window is missing.
    Handles ±Inf values as valid extrema in min/max comparison.
    """
    cdef:
        Py_ssize_t min_idx = -1
        Py_ssize_t max_idx = -1
        Py_ssize_t potential_idx = -1
        Py_ssize_t last_retained = -1
        Py_ssize_t out_idx = 0
        Py_ssize_t i, start, end
        double py
        double min_y, max_y
        int previous_min_flag = -1
        bint has_valid

    _retain_index(x, y, 0, out_x, out_y, &out_idx, &last_retained)
    start = 0
    while start < n:
        end = start + ratio
        if end > n:
            end = n
        has_valid = False
        for i in range(start, end):
            py = y[i]
            if isnan(py):
                continue
            has_valid = True
            if min_idx < 0:
                min_idx = max_idx = i
                min_y = max_y = py
            elif py >= max_y:
                max_idx = i
                max_y = py
            elif py < min_y:
                min_idx = i
                min_y = py

        if has_valid:
            _process_window_state(
                x, y, &min_idx, &max_idx, &potential_idx, &previous_min_flag,
                out_x, out_y, &out_idx, &last_retained,
            )
            min_y = y[min_idx]
            max_y = y[max_idx]
        else:
            if potential_idx >= 0:
                _retain_index(x, y, potential_idx, out_x, out_y, &out_idx, &last_retained)
            _retain_index(x, y, start, out_x, out_y, &out_idx, &last_retained)
            min_idx = max_idx = potential_idx = -1
            previous_min_flag = -1
        start = end

    if potential_idx >= 0:
        _retain_index(x, y, potential_idx, out_x, out_y, &out_idx, &last_retained)
    _retain_index(x, y, n - 1, out_x, out_y, &out_idx, &last_retained)

    return out_idx


cdef inline Py_ssize_t _output_capacity(Py_ssize_t n, int ratio) noexcept nogil:
    """Bound output size before writing to caller-owned buffers."""
    cdef Py_ssize_t capacity = 2 * ((n - 1) // ratio + 1) + 2
    return n if n < capacity else capacity


# =============================================================================
# In-place variant for pre-allocated output buffers
# =============================================================================

def downsample_into(
    x,
    y,
    int ratio,
    double[::1] out_x not None,
    double[::1] out_y not None,
):
    """
    Downsample into pre-allocated output buffers.

    This avoids memory allocation overhead when called repeatedly.
    Output buffers must not overlap the inputs or each other. A size of len(x)
    is always sufficient. Smaller buffers must cover
    min(len(x), 2 * ceil(len(x) / ratio) + 2). With ratio=1, len(x) is required.
    The written prefix matches downsample(x, y, ratio).

    Parameters
    ----------
    x, y : array-like
        Input coordinates. Integer inputs are supported and converted to float64.
    ratio : int
        Sampling ratio R >= 1.
    out_x, out_y : array-like
        Pre-allocated output buffers (contiguous float64).

    Returns
    -------
    count : int
        Number of points written to output buffers.
    """
    cdef np.ndarray x_arr = np.ascontiguousarray(x, dtype=np.float64)
    cdef np.ndarray y_arr = np.ascontiguousarray(y, dtype=np.float64)
    cdef const double[::1] x_view = x_arr
    cdef const double[::1] y_view = y_arr
    cdef Py_ssize_t n = x_view.shape[0]

    if y_view.shape[0] != n:
        raise ValueError(f"x and y must have the same length")

    if ratio < 1:
        raise ValueError(f"Sampling ratio must be >= 1, got {ratio}")

    if n == 0:
        return 0

    cdef np.ndarray out_x_arr = np.asarray(out_x)
    cdef np.ndarray out_y_arr = np.asarray(out_y)
    if (np.shares_memory(out_x_arr, x) or np.shares_memory(out_x_arr, y)
            or np.shares_memory(out_y_arr, x) or np.shares_memory(out_y_arr, y)
            or np.shares_memory(out_x_arr, out_y_arr)):
        raise ValueError("Output buffers must not overlap the input arrays or each other.")

    cdef Py_ssize_t max_output = _output_capacity(n, ratio)

    if out_x.shape[0] < max_output or out_y.shape[0] < max_output:
        raise ValueError(f"Output buffers too small. Need at least {max_output} elements.")

    if ratio == 1:
        memcpy(&out_x[0], &x_view[0], n * sizeof(double))
        memcpy(&out_y[0], &y_view[0], n * sizeof(double))
        return n

    cdef const double* x_ptr = &x_view[0]
    cdef const double* y_ptr = &y_view[0]
    cdef double* ox_ptr = &out_x[0]
    cdef double* oy_ptr = &out_y[0]
    cdef Py_ssize_t count

    with nogil:
        count = _downsample_core_ptr(x_ptr, y_ptr, n, ratio, ox_ptr, oy_ptr)

    return count


# =============================================================================
# Batch processing for multiple series
# =============================================================================

def downsample_batch(
    list x_arrays,
    list y_arrays,
    int ratio,
):
    """
    Downsample multiple time series in batch.

    More efficient than calling downsample() in a loop due to reduced
    Python overhead.

    Parameters
    ----------
    x_arrays : list of array-like
        List of x-coordinate arrays. Integer inputs are supported.
    y_arrays : list of array-like
        List of y-coordinate arrays. Integer inputs are supported.
    ratio : int
        Sampling ratio R >= 1.

    Returns
    -------
    results : list of tuples
        List of (x_downsampled, y_downsampled) tuples.
    """
    cdef Py_ssize_t num_series = len(x_arrays)

    if len(y_arrays) != num_series:
        raise ValueError("x_arrays and y_arrays must have the same length")

    cdef list results = []
    cdef Py_ssize_t i
    cdef np.ndarray x_arr, y_arr
    cdef const double[::1] x_view, y_view

    for i in range(num_series):
        x_arr = np.ascontiguousarray(x_arrays[i], dtype=np.float64)
        y_arr = np.ascontiguousarray(y_arrays[i], dtype=np.float64)
        x_view = x_arr
        y_view = y_arr
        results.append(_downsample_float64(x_view, y_view, ratio))

    return results
