"""
FPCS (Feature-Preserving Compensated Sampling) Time Series Downsampling Algorithm

This module provides two interfaces for the FPCS algorithm:
1. Batch processing: `downsample(x, y, ratio)` for complete datasets
2. Streaming: `FPCSDownsampler` class for incremental data processing
"""

from dataclasses import dataclass
from numbers import Number
from typing import Iterator
import numpy as np
from numpy.typing import ArrayLike


def _isfinite_scalar(value) -> bool:
    """Check numeric scalars without narrowing their precision or range."""
    try:
        return np.isfinite(value)
    except TypeError:
        if not isinstance(value, Number):
            raise
        # NumPy does not handle all Python numbers (e.g. large ints or Decimal).
        # Comparisons avoid float conversion and its rounding/overflow limits.
        return value == value and value != np.inf and value != -np.inf


@dataclass
class Point:
    """A 2D point with x (time) and y (value) coordinates."""
    x: float
    y: float


class FPCSDownsampler:
    """
    Streaming FPCS downsampler for time series data.

    This class implements the FPCS algorithm for streaming data, allowing
    points to be added one at a time and retained points to be yielded.

    To match batch processing with the same input precision, flush only at the
    end. Each flush resets window and compensation state, so frequent flushing
    changes compression. Integer comparisons are
    retained; Cython batch APIs convert inputs to float64 and can select different
    extrema for integers outside [-2**53, 2**53].

    Parameters
    ----------
    ratio : int
        The window size R, counting all original input positions. Must be >= 1.
        Non-finite points are discarded, without gap markers. Each flushed
        segment is bounded by min(n, 2 * ceil(n / R) + 2), including endpoints,
        where n is its input point count. Sum these bounds across segments.

    Example
    -------
    >>> downsampler = FPCSDownsampler(ratio=10)
    >>> for x, y in data_stream:
    ...     for retained_x, retained_y in downsampler.add(x, y):
    ...         process(retained_x, retained_y)
    >>> # Don't forget to flush at the end
    >>> for retained_x, retained_y in downsampler.flush():
    ...     process(retained_x, retained_y)
    """

    def __init__(self, ratio: int):
        if ratio < 1:
            raise ValueError(f"Sampling ratio must be >= 1, got {ratio}")

        self.ratio = ratio
        self._reset()

    def _reset(self):
        """Reset the internal state."""
        self._index: int = -1
        self._last_retained_index: int = -1
        self._last_point: Point | None = None
        self._last_point_index: int = -1
        self._clear_window_state()

    def _clear_window_state(self):
        """Clear extrema and compensation after a flush."""
        self._window_has_valid = False
        self._potential_point: Point | None = None
        self._potential_index: int = -1
        self._previous_min_flag: int = -1  # -1: no data, 0: retained max, 1: retained min
        self._counter: int = 0
        self._max_point: Point | None = None
        self._max_index: int = -1
        self._min_point: Point | None = None
        self._min_index: int = -1

    def _retain(self, index: int, point: Point) -> Iterator[tuple[float, float]]:
        """Yield each source point at most once, in input order."""
        if index > self._last_retained_index:
            self._last_retained_index = index
            yield (point.x, point.y)

    def add(self, x: float, y: float) -> Iterator[tuple[float, float]]:
        """
        Add a new data point to the downsampler.

        Parameters
        ----------
        x : float
            The x-coordinate (typically time/index).
        y : float
            The y-coordinate (the value). Points with non-finite x or y are
            ignored as extrema candidates but count toward window length.
            With ratio=1, all finite points are retained.

        Yields
        ------
        tuple[float, float]
            Retained points as (x, y) tuples.
        """
        index = self._index + 1
        self._index = index
        self._counter += 1
        if not (_isfinite_scalar(x) and _isfinite_scalar(y)):
            if self._counter >= self.ratio:
                if self._window_has_valid:
                    yield from self._process_window()
                else:
                    self._counter = 0
            return

        if self.ratio == 1:
            yield (x, y)
            return

        p = Point(x, y)
        if self._last_point is None:
            yield from self._retain(index, p)
        self._last_point = p
        self._last_point_index = index
        self._window_has_valid = True

        if self._max_point is None:
            self._max_point = self._min_point = p
            self._max_index = self._min_index = index
        elif p.y >= self._max_point.y:
            self._max_point = p
            self._max_index = index
        elif p.y < self._min_point.y:
            self._min_point = p
            self._min_index = index

        # Check if we've reached the sampling threshold
        if self._counter >= self.ratio:
            yield from self._process_window()

    def _process_window(self) -> Iterator[tuple[float, float]]:
        """Process the current window and yield retained points."""
        if self._min_index < self._max_index:
            if self._previous_min_flag == 1 and self._potential_point is not None:
                yield from self._retain(self._potential_index, self._potential_point)
            yield from self._retain(self._min_index, self._min_point)
            self._potential_point = self._max_point
            self._potential_index = self._max_index
            self._min_point = self._max_point
            self._min_index = self._max_index
            self._previous_min_flag = 1
        else:
            if self._previous_min_flag == 0 and self._potential_point is not None:
                yield from self._retain(self._potential_index, self._potential_point)
            yield from self._retain(self._max_index, self._max_point)
            self._potential_point = self._min_point
            self._potential_index = self._min_index
            self._max_point = self._min_point
            self._max_index = self._min_index
            self._previous_min_flag = 0

        self._counter = 0
        self._window_has_valid = False

    def flush(self) -> Iterator[tuple[float, float]]:
        """
        Finish the partial window, pending extremum, and global endpoint.

        Call this at the end of the stream to retain pending extrema and the
        last endpoint. Repeated calls yield nothing until new points are added.
        New points start a fresh segment with reset windows and compensation;
        its output bound applies separately. Intermediate flushes change
        compression.

        Yields
        ------
        tuple[float, float]
            Any remaining retained points as (x, y) tuples.
        """
        if self._last_point is None:
            self._clear_window_state()
            return
        if self._window_has_valid:
            yield from self._process_window()
        if self._potential_point is not None:
            yield from self._retain(self._potential_index, self._potential_point)
        yield from self._retain(self._last_point_index, self._last_point)
        self._clear_window_state()
        self._last_point = None

    def reset(self):
        """Reset the downsampler to its initial state."""
        self._reset()


def downsample(
    x: ArrayLike,
    y: ArrayLike,
    ratio: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Downsample a time series using the FPCS algorithm.

    Points with non-finite x or y (NaN or either infinity) are ignored as extrema
    candidates. Windows count original input positions, including invalid pairs.
    The first and last finite points of the series are retained. With ratio=1, all finite points are retained.
    No gap markers are emitted; traces connect across discarded points. All-invalid input
    produces empty arrays.

    Integer comparisons are retained. Cython batch APIs convert inputs to
    float64, so exact values and selected extrema can differ for integers
    outside [-2**53, 2**53]. For large integer timestamps, subtract an integer
    origin before conversion to preserve plotting resolution.

    Parameters
    ----------
    x : array-like
        The x-coordinates (e.g., timestamps or indices).
    y : array-like
        The y-coordinates (e.g., values).
    ratio : int
        The window size R in original input positions. Must be >= 1.
        Output is bounded by min(n, 2 * ceil(n / R) + 2), where n is the input
        point count (and output cannot exceed the finite-pair count).

    Returns
    -------
    x_downsampled : np.ndarray
        The x-coordinates of retained points.
    y_downsampled : np.ndarray
        The y-coordinates of retained points.

    Example
    -------
    >>> import numpy as np
    >>> x = np.arange(1000)
    >>> y = np.sin(x * 0.1) + np.random.randn(1000) * 0.1
    >>> x_down, y_down = downsample(x, y, ratio=10)
    >>> print(f"Reduced from {len(x)} to {len(x_down)} points")
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) != len(y):
        raise ValueError(f"x and y must have the same length, got {len(x)} and {len(y)}")

    if ratio < 1:
        raise ValueError(f"Sampling ratio must be >= 1, got {ratio}")

    if len(x) == 0:
        return np.array([]), np.array([])

    if len(x) == 1 or ratio == 1:
        if x.dtype.hasobject or y.dtype.hasobject:
            finite = np.fromiter(
                (_isfinite_scalar(xi) and _isfinite_scalar(yi) for xi, yi in zip(x, y)),
                dtype=bool, count=len(x),
            )
        else:
            finite = np.isfinite(x) & np.isfinite(y)
        return x[finite].copy(), y[finite].copy()

    downsampler = FPCSDownsampler(ratio)

    retained_x = []
    retained_y = []

    for xi, yi in zip(x, y):
        for rx, ry in downsampler.add(xi, yi):
            retained_x.append(rx)
            retained_y.append(ry)

    # Flush any remaining points
    for rx, ry in downsampler.flush():
        retained_x.append(rx)
        retained_y.append(ry)

    return np.array(retained_x), np.array(retained_y)


if __name__ == "__main__":
    # Demo usage
    import matplotlib.pyplot as plt

    # Generate sample data
    np.random.seed(42)
    n_points = 1000
    x = np.arange(n_points)
    y = np.sin(x * 0.05) * 50 + np.random.randn(n_points) * 10

    # Downsample with ratio 10
    ratio = 10
    x_down, y_down = downsample(x, y, ratio)

    print(f"Original: {len(x)} points")
    print(f"Downsampled: {len(x_down)} points")
    print(f"Compression ratio: {len(x) / len(x_down):.2f}x")

    # Plot comparison
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axes[0].plot(x, y, 'b-', alpha=0.7, linewidth=0.5)
    axes[0].set_title(f'Original ({len(x)} points)')
    axes[0].set_ylabel('Value')

    axes[1].plot(x_down, y_down, 'r-', alpha=0.7, linewidth=0.5)
    axes[1].scatter(x_down, y_down, c='red', s=5)
    axes[1].set_title(f'Downsampled with FPCS (ratio={ratio}, {len(x_down)} points)')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Value')

    plt.tight_layout()
    plt.savefig('fpcs_demo.png', dpi=150)
    plt.show()
