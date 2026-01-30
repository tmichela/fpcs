"""
FPCS (Feature-Preserving Compensated Sampling) Time Series Downsampling Algorithm

This module provides two interfaces for the FPCS algorithm:
1. Batch processing: `downsample(x, y, ratio)` for complete datasets
2. Streaming: `FPCSDownsampler` class for incremental data processing
"""

from dataclasses import dataclass
from typing import Iterator
import math
import numpy as np
from numpy.typing import ArrayLike


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

    Parameters
    ----------
    ratio : int
        The sampling ratio R. Must be >= 1. For every R points received,
        approximately 1-2 points will be retained.

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
        self._potential_point: Point | None = None
        self._previous_min_flag: int = -1  # -1: no data, 0: retained max, 1: retained min
        self._counter: int = 0
        self._max_point: Point | None = None
        self._min_point: Point | None = None
        self._initialized: bool = False
        self._pending_nans: list[Point] = []  # NaN points to yield before next retained point

    def add(self, x: float, y: float) -> Iterator[tuple[float, float]]:
        """
        Add a new data point to the downsampler.

        Parameters
        ----------
        x : float
            The x-coordinate (typically time/index).
        y : float
            The y-coordinate (the value). NaN values are always retained.
            ±Inf values are treated as extrema.

        Yields
        ------
        tuple[float, float]
            Retained points as (x, y) tuples.
        """
        p = Point(x, y)

        # Handle NaN: always retain NaN points (to preserve data gaps)
        # but don't use them for min/max comparison
        if math.isnan(y):
            self._pending_nans.append(p)
            self._counter += 1
            # Check if we've reached the sampling threshold
            if self._counter >= self.ratio:
                yield from self._process_window()
            return

        # First point initialization (skip NaN for initialization)
        if not self._initialized:
            self._max_point = p
            self._min_point = p
            self._counter = 1
            self._initialized = True
            return

        self._counter += 1

        # Update max/min points (Inf values are valid extrema)
        # Note: comparisons with Inf work correctly in Python
        if p.y >= self._max_point.y:
            self._max_point = p
        elif p.y < self._min_point.y:
            self._min_point = p

        # Check if we've reached the sampling threshold
        if self._counter >= self.ratio:
            yield from self._process_window()

    def _process_window(self) -> Iterator[tuple[float, float]]:
        """Process the current window and yield retained points."""
        # Yield any pending NaN points first (in order)
        for nan_point in self._pending_nans:
            yield (nan_point.x, nan_point.y)
        self._pending_nans.clear()

        # Handle case where we only had NaN points (no valid min/max)
        if not self._initialized:
            self._counter = 0
            return

        if self._min_point.x < self._max_point.x:
            # MinPoint received before MaxPoint
            if (self._previous_min_flag == 1 and 
                self._potential_point is not None and
                self._min_point != self._potential_point):
                # Compensation: both adjacent samplings retain MinPoint
                yield (self._potential_point.x, self._potential_point.y)

            # Retain MinPoint
            yield (self._min_point.x, self._min_point.y)

            # Update state
            self._potential_point = self._max_point
            self._min_point = self._max_point
            self._previous_min_flag = 1
        else:
            # MaxPoint received before MinPoint (or same position)
            if (self._previous_min_flag == 0 and
                self._potential_point is not None and
                self._max_point != self._potential_point):
                # Compensation: both adjacent samplings retain MaxPoint
                yield (self._potential_point.x, self._potential_point.y)

            # Retain MaxPoint
            yield (self._max_point.x, self._max_point.y)

            # Update state
            self._potential_point = self._min_point
            self._max_point = self._min_point
            self._previous_min_flag = 0

        self._counter = 0

    def flush(self) -> Iterator[tuple[float, float]]:
        """
        Flush any remaining points from an incomplete window.

        Call this after all data has been processed to ensure no points
        are left in the buffer.

        Yields
        ------
        tuple[float, float]
            Any remaining retained points as (x, y) tuples.
        """
        # Yield any pending NaN points
        for nan_point in self._pending_nans:
            yield (nan_point.x, nan_point.y)
        self._pending_nans.clear()

        if self._counter > 0 and self._initialized:
            # Process the incomplete final window
            yield from self._process_window()

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

    The first and last data points are always retained. This ensures the output
    spans the same domain as the input.

    Parameters
    ----------
    x : array-like
        The x-coordinates (e.g., timestamps or indices).
    y : array-like
        The y-coordinates (e.g., values).
    ratio : int
        The sampling ratio R. Must be >= 1. For every R points,
        approximately 1-2 points will be retained.

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

    if len(x) == 1:
        return x.copy(), y.copy()

    # If ratio is 1, return a copy of the input
    if ratio == 1:
        return x.copy(), y.copy()

    # Store first and last points
    first_x, first_y = x[0], y[0]
    last_x, last_y = x[-1], y[-1]

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

    # Ensure first point is included
    if len(retained_x) == 0 or retained_x[0] != first_x:
        retained_x.insert(0, first_x)
        retained_y.insert(0, first_y)

    # Ensure last point is included
    if retained_x[-1] != last_x:
        retained_x.append(last_x)
        retained_y.append(last_y)

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
