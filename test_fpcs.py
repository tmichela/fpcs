"""
Unit tests and benchmarks for FPCS downsampling implementations.

Run with: pytest test_fpcs.py -v
Run benchmarks: pytest test_fpcs.py -v -k benchmark --benchmark-only
"""
from packaging import version

import importlib
import importlib.metadata

import numpy as np
import pytest

import fpcs
from fpcs import downsample, FPCSDownsampler, get_backend, downsample_into, downsample_batch
from fpcs import fpcs_pure


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_data():
    """Simple test data with known min/max pattern."""
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float64)
    y = np.array([5, 2, 8, 1, 9, 3, 7, 0, 6, 4], dtype=np.float64)
    return x, y


@pytest.fixture
def sinusoidal_data():
    """Sinusoidal data for visual verification."""
    x = np.arange(1000, dtype=np.float64)
    y = np.sin(x * 0.05) * 50 + np.random.default_rng(42).standard_normal(1000) * 5
    return x, y


@pytest.fixture
def large_data():
    """Large dataset for performance testing."""
    n = 100_000
    x = np.arange(n, dtype=np.float64)
    y = np.sin(x * 0.001) * 100 + np.random.default_rng(42).standard_normal(n) * 10
    return x, y


def _stream_downsample(x, y, ratio):
    downsampler = FPCSDownsampler(ratio)
    retained = []
    for xi, yi in zip(x, y):
        retained.extend(downsampler.add(xi, yi))
    retained.extend(downsampler.flush())
    if not retained:
        return np.array([]), np.array([])
    return np.asarray(retained).T


@pytest.fixture(params=["python", "streaming", "main", "into", "batch"])
def finite_downsample(request):
    """Exercise the same finite-only contract through each public entry point."""
    if request.param == "python":
        return fpcs_pure.downsample
    if request.param == "streaming":
        return _stream_downsample
    if request.param == "main":
        return downsample
    if get_backend() != "cython":
        pytest.skip("Cython not available")
    if request.param == "batch":
        return lambda x, y, ratio: downsample_batch([x], [y], ratio)[0]

    def into(x, y, ratio):
        out_x = np.full(len(x) + 2, -12345.0)
        out_y = np.full(len(x) + 2, -67890.0)
        count = downsample_into(x, y, ratio, out_x[1:-1], out_y[1:-1])
        assert 0 <= count <= len(x)
        np.testing.assert_array_equal(out_x[[0, -1]], [-12345.0, -12345.0])
        np.testing.assert_array_equal(out_y[[0, -1]], [-67890.0, -67890.0])
        return out_x[1:count + 1], out_y[1:count + 1]

    return into


# =============================================================================
# Backend Info
# =============================================================================

class TestBackend:
    """Test backend detection."""

    def test_backend_is_valid(self):
        """Backend should be 'cython' or 'python'."""
        assert get_backend() in ("cython", "python")


@pytest.mark.parametrize("installed", [True, False])
def test_package_version(monkeypatch, installed):
    def metadata_version(name):
        assert name == "fpcs"
        if not installed:
            raise importlib.metadata.PackageNotFoundError(name)
        return "1.2.3.dev4+g1234567"

    with monkeypatch.context() as patch:
        patch.setattr(importlib.metadata, "version", metadata_version)
        importlib.reload(fpcs)
        assert fpcs.__version__ == (
            "1.2.3.dev4+g1234567" if installed else "0+unknown"
        )
    importlib.reload(fpcs)


# =============================================================================
# Python Implementation Tests
# =============================================================================

class TestPythonDownsample:
    """Tests for the pure Python downsample function."""

    def test_empty_input(self):
        """Empty arrays should return empty arrays."""
        x, y = np.array([]), np.array([])
        x_out, y_out = fpcs_pure.downsample(x, y, ratio=10)
        assert len(x_out) == 0
        assert len(y_out) == 0

    def test_single_point(self):
        """Single point should be retained."""
        x, y = np.array([1.0]), np.array([5.0])
        x_out, y_out = fpcs_pure.downsample(x, y, ratio=10)
        assert len(x_out) == 1
        assert x_out[0] == 1.0
        assert y_out[0] == 5.0

    def test_ratio_one(self):
        """Ratio 1 should return a copy of input."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])
        x_out, y_out = fpcs_pure.downsample(x, y, ratio=1)
        np.testing.assert_array_equal(x_out, x)
        np.testing.assert_array_equal(y_out, y)

    def test_invalid_ratio(self):
        """Ratio < 1 should raise ValueError."""
        x, y = np.array([1.0]), np.array([1.0])
        with pytest.raises(ValueError, match="ratio must be >= 1"):
            fpcs_pure.downsample(x, y, ratio=0)

    def test_mismatched_lengths(self):
        """Mismatched x and y lengths should raise ValueError."""
        x = np.array([1.0, 2.0])
        y = np.array([1.0])
        with pytest.raises(ValueError, match="same length"):
            fpcs_pure.downsample(x, y, ratio=10)

    def test_preserves_extrema(self, simple_data):
        """Downsampling should preserve local extrema within windows."""
        x, y = simple_data
        x_out, y_out = fpcs_pure.downsample(x, y, ratio=5)

        # With ratio=5, we have 2 windows: [0-4] and [5-9]
        # Window 1: y=[5,2,8,1,9] -> min=1, max=9
        # Window 2: y=[3,7,0,6,4] -> min=0, max=7
        # Each window retains min/max based on which came first
        assert len(x_out) >= 2  # At least one point per window
        # Check that at least one extrema per window is retained
        assert 9.0 in y_out or 1.0 in y_out  # Window 1 extrema

    def test_output_smaller_than_input(self, sinusoidal_data):
        """Output should be smaller than input for ratio > 1."""
        x, y = sinusoidal_data
        x_out, y_out = fpcs_pure.downsample(x, y, ratio=10)
        assert len(x_out) < len(x)
        assert len(y_out) < len(y)

    def test_output_x_is_sorted(self, sinusoidal_data):
        """Output x values should be in ascending order."""
        x, y = sinusoidal_data
        x_out, y_out = fpcs_pure.downsample(x, y, ratio=10)
        assert np.all(np.diff(x_out) >= 0)

    def test_output_values_from_input(self, sinusoidal_data):
        """All output values should come from input."""
        x, y = sinusoidal_data
        x_out, y_out = fpcs_pure.downsample(x, y, ratio=10)

        for xi, yi in zip(x_out, y_out):
            idx = np.where(x == xi)[0]
            assert len(idx) > 0
            assert y[idx[0]] == yi


class TestPythonStreaming:
    """Tests for the streaming FPCSDownsampler class."""

    def test_streaming_produces_output(self, sinusoidal_data):
        """Streaming should produce output points."""
        x, y = sinusoidal_data
        ratio = 10

        # Streaming preserves the same endpoints as batch processing.
        downsampler = FPCSDownsampler(ratio)
        x_stream = []
        y_stream = []

        for xi, yi in zip(x, y):
            for rx, ry in downsampler.add(xi, yi):
                x_stream.append(rx)
                y_stream.append(ry)

        for rx, ry in downsampler.flush():
            x_stream.append(rx)
            y_stream.append(ry)

        # Should have produced some output
        assert len(x_stream) > 0
        assert len(x_stream) < len(x)  # Downsampled

    @pytest.mark.parametrize("values", [
        [3, 1, 9, 2],
        [np.nan, np.nan],
        [3, 1, 9, 2, np.nan, np.nan],
    ])
    def test_reset(self, values):
        """Reset should clear internal state."""
        downsampler = FPCSDownsampler(ratio=4)
        for i, yi in enumerate(values):
            list(downsampler.add(float(i), yi))
        downsampler.reset()

        x = np.arange(100, 108, dtype=np.float64)
        y = np.array([np.nan, np.nan, 4, -2, 8, 0, np.nan, np.nan])
        retained = []
        for xi, yi in zip(x, y):
            retained.extend(downsampler.add(xi, yi))
        retained.extend(downsampler.flush())
        expected_x, expected_y = _stream_downsample(x, y, ratio=4)
        actual_x, actual_y = np.asarray(retained).reshape(-1, 2).T
        np.testing.assert_array_equal(actual_x, expected_x)
        np.testing.assert_array_equal(actual_y, expected_y)

    @pytest.mark.parametrize("values", [
        [np.nan] * 20,
        [3, 1, 9, 2],
        [np.nan, 3, 1, 9, 2, np.nan, np.nan],
    ])
    def test_flush_is_idempotent(self, values):
        downsampler = FPCSDownsampler(ratio=4)
        retained = []
        for i, yi in enumerate(values):
            retained.extend(downsampler.add(float(i), yi))
        retained.extend(downsampler.flush())
        assert list(downsampler.flush()) == []
        expected_x, expected_y = fpcs_pure.downsample(np.arange(len(values)), values, 4)
        actual_x, actual_y = np.asarray(retained).reshape(-1, 2).T
        np.testing.assert_array_equal(actual_x, expected_x)
        np.testing.assert_array_equal(actual_y, expected_y)

    def test_invalid_ratio(self):
        """Ratio < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="ratio must be >= 1"):
            FPCSDownsampler(ratio=0)

    @pytest.mark.parametrize("prefix", [[3, 1, 9, 2], [3, np.nan], [np.nan] * 6])
    def test_new_data_after_flush_starts_fresh_windows(self, prefix):
        downsampler = FPCSDownsampler(ratio=4)
        for i, yi in enumerate(prefix):
            list(downsampler.add(float(i), yi))
        list(downsampler.flush())
        assert list(downsampler.flush()) == []

        values = [np.nan, 3, 1, 9, 2, np.nan]
        retained = []
        for i, yi in enumerate(values, start=len(prefix)):
            retained.extend(downsampler.add(float(i), yi))
        retained.extend(downsampler.flush())
        indices = np.array([1, 2, 3, 4])
        actual_x, actual_y = np.asarray(retained).reshape(-1, 2).T
        np.testing.assert_array_equal(actual_x, indices + len(prefix))
        np.testing.assert_array_equal(actual_y, np.asarray(values)[indices])


# =============================================================================
# Main API Tests (uses Cython if available)
# =============================================================================

class TestMainAPI:
    """Tests for the main downsample API (Cython or Python)."""

    def test_downsample_works(self, sinusoidal_data):
        """Main downsample function should work."""
        x, y = sinusoidal_data
        x_out, y_out = downsample(x, y, ratio=10)
        assert len(x_out) < len(x)
        assert len(x_out) == len(y_out)

    def test_downsample_integer_input(self):
        """Main downsample should accept integer inputs."""
        x = np.arange(100, dtype=np.int64)
        y = np.arange(100, dtype=np.int64)

        x_out, y_out = downsample(x, y, ratio=10)

        assert len(x_out) < len(x)
        assert len(x_out) == len(y_out)
        assert np.all(np.isin(x_out, x))
        assert np.all(np.isin(y_out, y))

    @pytest.mark.skipif(get_backend() != "cython", reason="Cython not available")
    def test_downsample_into_works(self, sinusoidal_data):
        """downsample_into should work with pre-allocated buffers."""
        x, y = sinusoidal_data
        ratio = 10
        max_output = ((len(x) + ratio - 1) // ratio) * 2 + 2

        out_x = np.empty(max_output, dtype=np.float64)
        out_y = np.empty(max_output, dtype=np.float64)

        count = downsample_into(x, y, ratio, out_x, out_y)

        assert count > 0
        assert count < len(x)

    @pytest.mark.skipif(get_backend() != "cython", reason="Cython not available")
    def test_downsample_into_integer_input(self):
        """downsample_into should accept integer inputs."""
        x = np.arange(100, dtype=np.int64)
        y = np.arange(100, dtype=np.int64)
        ratio = 10
        max_output = ((len(x) + ratio - 1) // ratio) * 2 + 2

        out_x = np.empty(max_output, dtype=np.float64)
        out_y = np.empty(max_output, dtype=np.float64)

        count = downsample_into(x, y, ratio, out_x, out_y)

        assert count > 0
        assert count < len(x)
        assert np.all(np.isin(out_x[:count], x))
        assert np.all(np.isin(out_y[:count], y))

    @pytest.mark.skipif(get_backend() != "cython", reason="Cython not available")
    def test_downsample_batch_works(self, sinusoidal_data):
        """downsample_batch should process multiple series."""
        x, y = sinusoidal_data

        results = downsample_batch([x, x], [y, y], ratio=10)

        assert len(results) == 2
        assert len(results[0][0]) == len(results[1][0])


# =============================================================================
# First/Last Point Retention Tests
# =============================================================================

class TestFirstLastPointRetention:
    """Tests for all interfaces retaining first and last data points."""

    def test_first_point_retained(self, sinusoidal_data):
        """First point should always be retained in batch mode."""
        x, y = sinusoidal_data
        x_out, y_out = downsample(x, y, ratio=10)

        assert x_out[0] == x[0]
        assert y_out[0] == y[0]

    def test_last_point_retained(self, sinusoidal_data):
        """Last point should always be retained in batch mode."""
        x, y = sinusoidal_data
        x_out, y_out = downsample(x, y, ratio=10)

        assert x_out[-1] == x[-1]
        assert y_out[-1] == y[-1]

    def test_first_last_with_large_ratio(self):
        """First and last should be retained even with very large ratio."""
        x = np.arange(10, dtype=np.float64)
        y = np.ones(10, dtype=np.float64)

        # Ratio larger than data length
        x_out, y_out = downsample(x, y, ratio=100)

        assert x_out[0] == x[0]
        assert x_out[-1] == x[-1]

    def test_streaming_retains_first_last(self):
        """Streaming preserves endpoints even before a window completes."""
        x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float64)
        y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float64)  # monotonic

        x_out, y_out = _stream_downsample(x, y, ratio=100)
        np.testing.assert_array_equal(x_out, x[[0, -1]])
        np.testing.assert_array_equal(y_out, y[[0, -1]])

    def test_python_and_cython_first_last_match(self, sinusoidal_data):
        """Python and Cython should produce same first/last points."""
        x, y = sinusoidal_data

        x_py, y_py = fpcs_pure.downsample(x, y, ratio=10)
        x_main, y_main = downsample(x, y, ratio=10)

        # First point
        assert x_py[0] == x_main[0] == x[0]
        assert y_py[0] == y_main[0] == y[0]

        # Last point
        assert x_py[-1] == x_main[-1] == x[-1]
        assert y_py[-1] == y_main[-1] == y[-1]


# =============================================================================
# Cross-Implementation Tests
# =============================================================================

class TestImplementationEquivalence:
    """Tests that Python and Cython implementations produce identical results."""

    @pytest.mark.skipif(get_backend() != "cython", reason="Cython not available")
    @pytest.mark.parametrize("ratio", [2, 5, 10, 50, 100])
    def test_same_output(self, sinusoidal_data, ratio):
        """Python and Cython should produce identical results."""
        x, y = sinusoidal_data

        from fpcs import fpcs_cy

        x_py, y_py = fpcs_pure.downsample(x, y, ratio)
        x_cy, y_cy = fpcs_cy.downsample(x, y, ratio)

        np.testing.assert_array_equal(x_py, x_cy)
        np.testing.assert_array_equal(y_py, y_cy)

    @pytest.mark.skipif(get_backend() != "cython", reason="Cython not available")
    def test_same_output_large(self, large_data):
        """Python and Cython should match on large datasets."""
        x, y = large_data
        ratio = 100

        from fpcs import fpcs_cy

        x_py, y_py = fpcs_pure.downsample(x, y, ratio)
        x_cy, y_cy = fpcs_cy.downsample(x, y, ratio)

        np.testing.assert_array_equal(x_py, x_cy)
        np.testing.assert_array_equal(y_py, y_cy)


# =============================================================================
# Benchmarks
# =============================================================================

class TestBenchmarks:
    """Benchmark tests using pytest-benchmark."""

    @pytest.fixture
    def benchmark_data(self):
        """1M points for benchmarking."""
        n = 1_000_000
        x = np.arange(n, dtype=np.float64)
        y = np.sin(x * 0.001) * 100 + np.random.default_rng(42).standard_normal(n) * 10
        ratio = 100
        return x, y, ratio

    @pytest.mark.benchmark(group="downsample")
    def test_benchmark_python(self, benchmark, benchmark_data):
        """Benchmark pure Python implementation."""
        x, y, ratio = benchmark_data
        result = benchmark(fpcs_pure.downsample, x, y, ratio)
        assert len(result[0]) > 0

    @pytest.mark.benchmark(group="downsample")
    def test_benchmark_cython(self, benchmark, benchmark_data):
        """Benchmark Cython implementation (or Python fallback)."""
        x, y, ratio = benchmark_data
        result = benchmark(downsample, x, y, ratio)
        assert len(result[0]) > 0

    @pytest.mark.skipif(version.parse(np.__version__) >= version.parse("2"), reason="lttbc needs numpy <= 2")
    @pytest.mark.benchmark(group="downsample")
    def test_benchmark_lttbc(self, benchmark, benchmark_data):
        """Benchmark LTTB at different ratios."""
        from lttbc import downsample as lttbc_downsample
        x, y, ratio = benchmark_data
        # practically the fpcs algo picks ~1.25 points per ratio
        n_out = int(len(x) / (0.8 * ratio))
        result = benchmark(lttbc_downsample, x, y, n_out)
        assert len(result[0]) > 0

    @pytest.mark.skipif(get_backend() != "cython", reason="Cython not available")
    @pytest.mark.parametrize("ratio", [10, 50, 100, 500])
    @pytest.mark.benchmark(group="ratio-scaling")
    def test_benchmark_cython_ratios(self, benchmark, benchmark_data, ratio):
        """Benchmark Cython at different ratios."""
        x, y, _ = benchmark_data
        result = benchmark(downsample, x, y, ratio)
        assert len(result[0]) > 0


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and corner cases."""

    def test_all_same_values(self):
        """All identical y values."""
        x = np.arange(100, dtype=np.float64)
        y = np.ones(100, dtype=np.float64) * 5.0

        x_py, y_py = fpcs_pure.downsample(x, y, ratio=10)
        x_main, y_main = downsample(x, y, ratio=10)

        np.testing.assert_array_equal(x_py, x_main)
        np.testing.assert_array_equal(y_py, y_main)

    def test_monotonically_increasing(self):
        """Monotonically increasing y values."""
        x = np.arange(100, dtype=np.float64)
        y = np.arange(100, dtype=np.float64)

        x_py, y_py = fpcs_pure.downsample(x, y, ratio=10)
        x_main, y_main = downsample(x, y, ratio=10)

        np.testing.assert_array_equal(x_py, x_main)
        np.testing.assert_array_equal(y_py, y_main)

    def test_monotonically_decreasing(self):
        """Monotonically decreasing y values."""
        x = np.arange(100, dtype=np.float64)
        y = np.arange(99, -1, -1, dtype=np.float64)

        x_py, y_py = fpcs_pure.downsample(x, y, ratio=10)
        x_main, y_main = downsample(x, y, ratio=10)

        np.testing.assert_array_equal(x_py, x_main)
        np.testing.assert_array_equal(y_py, y_main)

    def test_alternating_values(self):
        """Alternating high/low values."""
        x = np.arange(100, dtype=np.float64)
        y = np.array([0.0 if i % 2 == 0 else 100.0 for i in range(100)], dtype=np.float64)

        x_py, y_py = fpcs_pure.downsample(x, y, ratio=10)
        x_main, y_main = downsample(x, y, ratio=10)

        np.testing.assert_array_equal(x_py, x_main)
        np.testing.assert_array_equal(y_py, y_main)

    def test_exact_window_size(self):
        """Input length exactly divisible by ratio."""
        x = np.arange(100, dtype=np.float64)
        y = np.random.default_rng(42).standard_normal(100)

        x_py, y_py = fpcs_pure.downsample(x, y, ratio=10)  # 100 / 10 = 10 windows
        x_main, y_main = downsample(x, y, ratio=10)

        np.testing.assert_array_equal(x_py, x_main)
        np.testing.assert_array_equal(y_py, y_main)

    def test_nearly_exact_window_size(self):
        """Input length one less than divisible by ratio."""
        x = np.arange(99, dtype=np.float64)
        y = np.random.default_rng(42).standard_normal(99)

        x_py, y_py = fpcs_pure.downsample(x, y, ratio=10)
        x_main, y_main = downsample(x, y, ratio=10)

        np.testing.assert_array_equal(x_py, x_main)
        np.testing.assert_array_equal(y_py, y_main)

    def test_negative_values(self):
        """Negative y values."""
        x = np.arange(100, dtype=np.float64)
        y = np.random.default_rng(42).standard_normal(100) - 100

        x_py, y_py = fpcs_pure.downsample(x, y, ratio=10)
        x_main, y_main = downsample(x, y, ratio=10)

        np.testing.assert_array_equal(x_py, x_main)
        np.testing.assert_array_equal(y_py, y_main)

    def test_very_large_values(self):
        """Very large y values to check for overflow."""
        x = np.arange(100, dtype=np.float64)
        y = np.random.default_rng(42).standard_normal(100) * 1e15

        x_py, y_py = fpcs_pure.downsample(x, y, ratio=10)
        x_main, y_main = downsample(x, y, ratio=10)

        np.testing.assert_array_equal(x_py, x_main)
        np.testing.assert_array_equal(y_py, y_main)


# =============================================================================
# NaN and Inf Handling Tests
# =============================================================================

class TestNonFiniteHandling:
    @pytest.mark.parametrize("ratio", [1, 2, 4, 100])
    @pytest.mark.parametrize("coordinate", ["x", "y"])
    @pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
    def test_invalid_pairs_match_filtered_input(self, finite_downsample, ratio, coordinate, invalid):
        x = np.arange(19, dtype=np.float64)
        y = np.array([3, 1, 9, 2, 8, 0, 7, 4, 5, -2, 1, 0, 9, 5, 3, 8, 2, 1, 6.])
        (x if coordinate == "x" else y)[[0, 2, 3, 4, 8, 13, 18]] = invalid
        mask = np.isfinite(x) & np.isfinite(y)
        expected = fpcs_pure.downsample(x[mask], y[mask], ratio)
        actual = finite_downsample(x, y, ratio)
        for result, reference in zip(actual, expected):
            np.testing.assert_array_equal(result, reference)
            assert np.isfinite(result).all()

    @pytest.mark.parametrize("ratio", [1, 3, 100])
    @pytest.mark.parametrize("values", [[], [np.nan], [np.inf], [-np.inf], [np.nan, np.inf, -np.inf]])
    def test_no_finite_points(self, finite_downsample, ratio, values):
        x_out, y_out = finite_downsample(np.arange(len(values)), values, ratio)
        assert len(x_out) == len(y_out) == 0

    @pytest.mark.parametrize("ratio", [1, 4, 100])
    def test_one_finite_point(self, finite_downsample, ratio):
        x_out, y_out = finite_downsample([0, 1, 2, 3], [np.nan, 7, np.inf, -np.inf], ratio)
        np.testing.assert_array_equal(x_out, [1])
        np.testing.assert_array_equal(y_out, [7])

    @pytest.mark.parametrize("ratio", [1, 2, 9, 100, 1000])
    def test_sparse_order_bound_and_parity(self, finite_downsample, ratio):
        rng = np.random.default_rng(731)
        for size in [0, 1, 2, 17, 300]:
            x = np.arange(size, dtype=np.float64)
            y = rng.standard_normal(size)
            y[rng.random(size) < 0.4] = np.nan
            y[rng.random(size) < 0.2] = np.inf
            x[rng.random(size) < 0.2] = -np.inf
            mask = np.isfinite(x) & np.isfinite(y)
            n = np.count_nonzero(mask)
            x_out, y_out = finite_downsample(x, y, ratio)
            expected = fpcs_pure.downsample(x[mask], y[mask], ratio)
            np.testing.assert_array_equal(x_out, expected[0])
            np.testing.assert_array_equal(y_out, expected[1])
            assert len(x_out) <= min(n, 2 * ((n + ratio - 1) // ratio) + 2)
            assert np.all(np.diff(x_out) > 0)
            if n:
                assert x_out[0] == x[mask][0]
                assert x_out[-1] == x[mask][-1]

    @pytest.mark.parametrize("ratio", [1, 4, 100])
    def test_readonly_strided_repeated_timestamps(self, finite_downsample, ratio):
        x = np.zeros(18)[::-2]
        y = np.array([np.nan, 4, 3, np.inf, 1, 9, -np.inf, 2, np.nan])[::-1]
        x.flags.writeable = y.flags.writeable = False
        mask = np.isfinite(x) & np.isfinite(y)
        actual = finite_downsample(x, y, ratio)
        expected = fpcs_pure.downsample(x[mask], y[mask], ratio)
        np.testing.assert_array_equal(actual, expected)

    @pytest.mark.parametrize("sample", [fpcs_pure.downsample, _stream_downsample])
    @pytest.mark.parametrize("ratio", [1, 2, 10])
    def test_python_preserves_extended_precision_finite_points(self, sample, ratio):
        if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
            pytest.skip("longdouble has no extended range on this platform")
        x = np.arange(3, dtype=np.longdouble)
        y = np.array(["1e400", "2e400", "3e400"], dtype=np.longdouble)
        x_out, y_out = sample(x, y, ratio)
        indices = [0, 2] if ratio > len(x) else [0, 1, 2]
        np.testing.assert_array_equal(x_out, x[indices])
        np.testing.assert_array_equal(y_out, y[indices])

    def test_million_fragmented_samples_have_bounded_finite_output(self):
        y = np.tile([np.nan, 1.0, np.inf, 2.0], 250_000)
        x = np.arange(len(y), dtype=np.float64)
        ratio = max(1, int(len(y) / (0.8 * 150)))
        x_out, y_out = downsample(x, y, ratio)
        n = np.count_nonzero(np.isfinite(y))
        assert len(x_out) <= 2 * ((n + ratio - 1) // ratio) + 2
        assert x_out[0] == 1
        assert x_out[-1] == len(y) - 1
        assert np.isfinite(y_out).all()
        np.testing.assert_array_equal(y_out, y[x_out.astype(np.intp)])


@pytest.mark.skipif(get_backend() != "cython", reason="Cython not available")
@pytest.mark.parametrize("values,ratio,capacity", [
    pytest.param([], 10, 0, id="empty"),
    pytest.param([7], 10, 1, id="one-valid-point"),
    pytest.param(np.full(1000, np.nan), 10, 202, id="all-nan"),
    pytest.param(np.tile([np.nan, 2.0], 32), 20, 10, id="fragmented-gaps"),
    pytest.param(np.r_[np.nan, np.arange(25), np.full(4, np.nan)], 10, 8, id="mixed-runs"),
    pytest.param([np.nan, 2, np.nan, 3], 1, 4, id="ratio-one"),
    pytest.param(np.arange(40), 10, 10, id="dense-conservative-bound"),
])
class TestOutputBufferCapacity:
    def test_documented_capacity_is_sufficient(self, values, ratio, capacity):
        y = np.asarray(values, dtype=np.float64)
        x = np.arange(len(y), dtype=np.float64)
        out_x = np.full(capacity + 2, -12345.0)
        out_y = np.full(capacity + 2, -67890.0)
        count = downsample_into(x, y, ratio, out_x[1:-1], out_y[1:-1])
        expected_x, expected_y = downsample(x, y, ratio)
        np.testing.assert_array_equal(out_x[1:count + 1], expected_x)
        np.testing.assert_array_equal(out_y[1:count + 1], expected_y)
        np.testing.assert_array_equal(out_x[[0, -1]], [-12345.0, -12345.0])
        np.testing.assert_array_equal(out_y[[0, -1]], [-67890.0, -67890.0])

    @pytest.mark.parametrize("short_buffer", ["x", "y"])
    def test_undersized_buffer_is_rejected_before_writing(self, values, ratio, capacity, short_buffer):
        if capacity == 0:
            pytest.skip("An empty result needs no buffer space")
        y = np.asarray(values, dtype=np.float64)
        x = np.arange(len(y), dtype=np.float64)
        out_x = np.full(2 * len(y) + 8, -12345.0)
        out_y = np.full(2 * len(y) + 8, -67890.0)
        size_x = capacity - (short_buffer == "x")
        size_y = capacity - (short_buffer == "y")
        with pytest.raises(ValueError, match="too small"):
            downsample_into(x, y, ratio, out_x[2:2 + size_x], out_y[2:2 + size_y])
        np.testing.assert_array_equal(out_x, np.full(out_x.shape, -12345.0))
        np.testing.assert_array_equal(out_y, np.full(out_y.shape, -67890.0))


@pytest.mark.skipif(get_backend() != "cython", reason="Cython not available")
class TestOutputBufferOverlap:
    @pytest.mark.parametrize("input_step", [1, 2, -2])
    @pytest.mark.parametrize("input_dtype", [np.float64, np.int64])
    @pytest.mark.parametrize("overlap", ["input-x", "input-y", "input-x-to-y", "input-y-to-x", "outputs"])
    def test_overlap_is_rejected_before_writing(self, overlap, input_dtype, input_step):
        x = np.arange(24, dtype=input_dtype)
        y = np.arange(24, dtype=input_dtype) ** 2
        out_x = np.full(12, -12345.0)
        out_y = np.full(12, -67890.0)
        if overlap == "input-x":
            out_x = x[6:18].view(np.float64)
        elif overlap == "input-y":
            out_y = y[6:18].view(np.float64)
        elif overlap == "input-x-to-y":
            out_y = x[6:18].view(np.float64)
        elif overlap == "input-y-to-x":
            out_x = y[6:18].view(np.float64)
        else:
            shared = np.full(24, -12345.0)
            out_x, out_y = shared[:12], shared[6:18]
        arrays = (x, y, out_x, out_y)
        before = [array.copy() for array in arrays]
        with pytest.raises(ValueError, match="overlap"):
            downsample_into(x[::input_step][:12], y[::input_step][:12], 4, out_x, out_y)
        for actual, expected in zip(arrays, before):
            np.testing.assert_array_equal(actual, expected)

    @pytest.mark.parametrize("input_step", [1, 2, -2])
    def test_disjoint_slices_of_one_allocation_are_allowed(self, input_step):
        x_storage = np.arange(36, dtype=np.float64)
        y_storage = x_storage ** 2
        x, out_x = x_storage[:12][::input_step], x_storage[12:24]
        y, out_y = y_storage[:12][::input_step], y_storage[12:24]
        expected_x, expected_y = downsample(x, y, 4)
        count = downsample_into(x, y, 4, out_x, out_y)
        np.testing.assert_array_equal(out_x[:count], expected_x)
        np.testing.assert_array_equal(out_y[:count], expected_y)
        np.testing.assert_array_equal(x, np.arange(12)[::input_step])
        np.testing.assert_array_equal(y, (np.arange(12) ** 2)[::input_step])
