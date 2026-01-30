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
from libc.string cimport memcpy, memmove
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

    # Store first and last points
    cdef double first_x = x[0]
    cdef double first_y = y[0]
    cdef double last_x = x[n - 1]
    cdef double last_y = y[n - 1]

    # Maximum possible output size: 
    # - All NaN values are retained (worst case: n NaN points)
    # - Plus 2 points per window for non-NaN extrema
    # - Plus 2 for first/last point retention
    cdef Py_ssize_t max_output = n + ((n + ratio - 1) // ratio) * 2 + 4

    # Allocate output buffers using C malloc for performance
    cdef double* out_x = <double*>malloc(max_output * sizeof(double))
    cdef double* out_y = <double*>malloc(max_output * sizeof(double))

    if out_x == NULL or out_y == NULL:
        if out_x != NULL:
            free(out_x)
        if out_y != NULL:
            free(out_y)
        raise MemoryError("Failed to allocate output buffers")

    cdef Py_ssize_t out_count
    cdef Py_ssize_t final_count
    cdef Py_ssize_t i
    cdef const double* x_ptr = &x[0]
    cdef const double* y_ptr = &y[0]

    # Run the core algorithm without GIL
    with nogil:
        out_count = _downsample_core_ptr(x_ptr, y_ptr, n, ratio, out_x, out_y)

        # Ensure first point is included
        final_count = out_count
        if out_count == 0 or out_x[0] != first_x:
            # Use memmove for efficient bulk shift (handles overlapping memory)
            memmove(&out_x[1], &out_x[0], out_count * sizeof(double))
            memmove(&out_y[1], &out_y[0], out_count * sizeof(double))
            out_x[0] = first_x
            out_y[0] = first_y
            final_count += 1

        # Ensure last point is included (always O(1) - just append)
        if out_x[final_count - 1] != last_x:
            out_x[final_count] = last_x
            out_y[final_count] = last_y
            final_count += 1

    # Create output arrays and copy data
    cdef np.ndarray[double, ndim=1, mode='c'] result_x = np.empty(final_count, dtype=np.float64)
    cdef np.ndarray[double, ndim=1, mode='c'] result_y = np.empty(final_count, dtype=np.float64)

    # Use memcpy for fast bulk copy
    memcpy(<void*>result_x.data, <void*>out_x, final_count * sizeof(double))
    memcpy(<void*>result_y.data, <void*>out_y, final_count * sizeof(double))

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

    The first and last data points are always retained. This ensures the output
    spans the same domain as the input.

    Parameters
    ----------
    x : array-like
        The x-coordinates (e.g., timestamps or indices). Integer inputs are supported.
    y : array-like
        The y-coordinates (e.g., values). Integer inputs are supported.
    ratio : int
        The sampling ratio R. Must be >= 1. For every R points,
        approximately 1-2 points will be retained.

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


cdef inline void _process_window_state(
    double min_x, double min_y,
    double max_x, double max_y,
    double potential_x, double potential_y,
    int* previous_min_flag,
    bint* has_potential,
    double* out_x, double* out_y,
    Py_ssize_t* out_idx
) noexcept nogil:
    """Inline helper to process a completed window and write retained points."""
    # Process window based on temporal order of extrema
    if min_x < max_x:
        # Min came before Max - retain Min
        if previous_min_flag[0] == 1 and has_potential[0]:
            if min_x != potential_x or min_y != potential_y:
                out_x[out_idx[0]] = potential_x
                out_y[out_idx[0]] = potential_y
                out_idx[0] += 1

        out_x[out_idx[0]] = min_x
        out_y[out_idx[0]] = min_y
        out_idx[0] += 1

        # Update state for next window (potential point becomes current max)
        # Note: We don't update potential_* variables here because they are
        # output-only parameters in this context. The caller must update them.
        previous_min_flag[0] = 1
    else:
        # Max came before Min (or same) - retain Max
        if previous_min_flag[0] == 0 and has_potential[0]:
            if max_x != potential_x or max_y != potential_y:
                out_x[out_idx[0]] = potential_x
                out_y[out_idx[0]] = potential_y
                out_idx[0] += 1

        out_x[out_idx[0]] = max_x
        out_y[out_idx[0]] = max_y
        out_idx[0] += 1

        previous_min_flag[0] = 0

    # Caller must update potential point and has_potential flag


cdef inline Py_ssize_t _downsample_core_ptr(
    const double* x,
    const double* y,
    Py_ssize_t n,
    int ratio,
    double* out_x,
    double* out_y,
) noexcept nogil:
    """
    Core downsampling logic using raw pointers for maximum performance.

    Handles NaN values by always retaining them (to preserve data gaps).
    Handles ±Inf values as valid extrema in min/max comparison.
    """
    cdef:
        # Current window extrema
        double max_x, max_y
        double min_x, min_y
        # Potential point for compensation
        double potential_x, potential_y
        # Current point being processed
        double px, py

        # State flags
        # previous_min_flag: -1 = no data, 0 = retained max, 1 = retained min
        int previous_min_flag = -1
        int counter = 0
        bint has_potential = False
        bint initialized = False

        # Indices
        Py_ssize_t out_idx = 0

        # Precompute for inner loop
        const double* x_end = x + n
        const double* x_ptr = x
        const double* y_ptr = y

    # Main processing loop
    while x_ptr < x_end:
        px = x_ptr[0]
        py = y_ptr[0]
        x_ptr += 1
        y_ptr += 1

        # Handle NaN: always retain but don't use for min/max
        if isnan(py):
            out_x[out_idx] = px
            out_y[out_idx] = py
            out_idx += 1
            counter += 1
            # Check window threshold (but NaN doesn't reset min/max)
            if counter >= ratio:
                # Process window if we have valid data
                if initialized:
                    _process_window_state(
                        min_x, min_y, max_x, max_y, potential_x, potential_y,
                        &previous_min_flag, &has_potential, out_x, out_y, &out_idx
                    )

                    # Update potential point based on what was retained
                    if previous_min_flag == 1: # Retained Min
                        potential_x = max_x
                        potential_y = max_y
                        min_x = max_x
                        min_y = max_y
                    else: # Retained Max
                        potential_x = min_x
                        potential_y = min_y
                        max_x = min_x
                        max_y = min_y
                    has_potential = True

                counter = 0
            continue

        # First valid (non-NaN) point initialization
        if not initialized:
            max_x = px
            max_y = py
            min_x = px
            min_y = py
            initialized = True
            counter = 1
            continue

        counter += 1

        # Update max/min - Inf values work correctly with comparisons
        if py >= max_y:
            max_x = px
            max_y = py
        elif py < min_y:
            min_x = px
            min_y = py

        # Window complete check
        if counter >= ratio:
            _process_window_state(
                min_x, min_y, max_x, max_y, potential_x, potential_y,
                &previous_min_flag, &has_potential, out_x, out_y, &out_idx
            )

            # Update potential point based on what was retained
            if previous_min_flag == 1: # Retained Min
                potential_x = max_x
                potential_y = max_y
                min_x = max_x
                min_y = max_y
            else: # Retained Max
                potential_x = min_x
                potential_y = min_y
                max_x = min_x
                max_y = min_y
            has_potential = True

            counter = 0

    # Flush remaining incomplete window
    if counter > 0 and initialized:
        _process_window_state(
            min_x, min_y, max_x, max_y, potential_x, potential_y,
            &previous_min_flag, &has_potential, out_x, out_y, &out_idx
        )

    return out_idx


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
    The output buffers must be large enough to hold the results.
    Recommended size: ((len(x) + ratio - 1) // ratio) * 2 + 2

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

    if n == 0:
        return 0

    if ratio < 1:
        raise ValueError(f"Sampling ratio must be >= 1, got {ratio}")

    cdef Py_ssize_t max_output = ((n + ratio - 1) // ratio) * 2 + 2

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
    cdef double[::1] x_view, y_view

    for i in range(num_series):
        x_arr = np.ascontiguousarray(x_arrays[i], dtype=np.float64)
        y_arr = np.ascontiguousarray(y_arrays[i], dtype=np.float64)
        x_view = x_arr
        y_view = y_arr
        results.append(_downsample_float64(x_view, y_view, ratio))

    return results
