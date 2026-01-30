"""
Unit tests and benchmarks for FPCS downsampling implementations.

Run with: pytest test_fpcs.py -v
Run benchmarks: pytest test_fpcs.py -v -k benchmark --benchmark-only
"""
from packaging import version

import numpy as np
import pytest

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


# =============================================================================
# Backend Info
# =============================================================================

class TestBackend:
    """Test backend detection."""

    def test_backend_is_valid(self):
        """Backend should be 'cython' or 'python'."""
        assert get_backend() in ("cython", "python")


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

        # Streaming (note: streaming does NOT auto-include first/last points)
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

    def test_reset(self):
        """Reset should clear internal state."""
        downsampler = FPCSDownsampler(ratio=5)

        # Add some points
        for i in range(10):
            list(downsampler.add(float(i), float(i)))

        # Reset
        downsampler.reset()

        # State should be cleared
        assert downsampler._counter == 0
        assert not downsampler._initialized

    def test_invalid_ratio(self):
        """Ratio < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="ratio must be >= 1"):
            FPCSDownsampler(ratio=0)


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
    """Tests for batch mode always retaining first and last data points."""

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

    def test_streaming_does_not_force_first_last(self):
        """Streaming mode should NOT auto-include first/last points."""
        x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float64)
        y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float64)  # monotonic

        downsampler = FPCSDownsampler(ratio=100)  # Large ratio, won't trigger window
        results = []
        for xi, yi in zip(x, y):
            results.extend(list(downsampler.add(xi, yi)))
        results.extend(list(downsampler.flush()))

        # With ratio=100 on 10 points, the streaming algorithm processes
        # all points in one window and flushes the result
        assert len(results) > 0

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
        x, y, ratio = benchmark_data
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

class TestNaNHandling:
    """Tests for NaN value handling."""

    def test_nan_values_retained(self):
        """NaN values should be retained in output."""
        x = np.arange(20, dtype=np.float64)
        y = np.sin(x * 0.5)
        y[5] = np.nan
        y[12] = np.nan

        x_out, y_out = downsample(x, y, ratio=5)

        nan_count = np.isnan(y_out).sum()
        assert nan_count == 2, f"Expected 2 NaN values, got {nan_count}"

    def test_nan_at_start(self):
        """NaN at start of data should be handled."""
        x = np.arange(20, dtype=np.float64)
        y = np.sin(x * 0.5)
        y[0] = np.nan

        x_out, y_out = downsample(x, y, ratio=5)

        # First point is always retained, NaN should be preserved
        assert np.isnan(y_out).sum() >= 1

    def test_nan_at_end(self):
        """NaN at end of data should be handled."""
        x = np.arange(20, dtype=np.float64)
        y = np.sin(x * 0.5)
        y[-1] = np.nan

        x_out, y_out = downsample(x, y, ratio=5)

        # Last point is always retained
        assert np.isnan(y_out[-1])

    def test_all_nan(self):
        """All NaN data should return all NaN."""
        x = np.arange(10, dtype=np.float64)
        y = np.full(10, np.nan)

        x_out, y_out = downsample(x, y, ratio=5)

        assert np.all(np.isnan(y_out))

    def test_python_cython_nan_equivalence(self):
        """Python and Cython should handle NaN identically."""
        x = np.arange(50, dtype=np.float64)
        y = np.sin(x * 0.2)
        y[10] = np.nan
        y[25] = np.nan
        y[40] = np.nan

        x_py, y_py = fpcs_pure.downsample(x, y, ratio=10)
        x_main, y_main = downsample(x, y, ratio=10)

        # Check same number of NaN values
        assert np.isnan(y_py).sum() == np.isnan(y_main).sum()


class TestInfHandling:
    """Tests for Inf value handling."""

    def test_positive_inf_retained(self):
        """+Inf should be retained as maximum."""
        x = np.arange(20, dtype=np.float64)
        y = np.sin(x * 0.5)
        y[7] = np.inf

        x_out, y_out = downsample(x, y, ratio=10)

        assert np.any(np.isinf(y_out) & (y_out > 0)), "+Inf not found in output"

    def test_negative_inf_retained(self):
        """-Inf should be retained as minimum."""
        x = np.arange(20, dtype=np.float64)
        y = np.sin(x * 0.5)
        y[7] = -np.inf

        x_out, y_out = downsample(x, y, ratio=10)

        assert np.any(np.isinf(y_out) & (y_out < 0)), "-Inf not found in output"

    def test_both_inf_in_window(self):
        """Both +Inf and -Inf in same window."""
        x = np.arange(20, dtype=np.float64)
        y = np.zeros(20)
        y[2] = np.inf
        y[4] = -np.inf

        x_out, y_out = downsample(x, y, ratio=10)

        # Both should be retained as they are the extrema
        assert np.any(y_out == np.inf), "+Inf not found"
        assert np.any(y_out == -np.inf), "-Inf not found"

    def test_python_cython_inf_equivalence(self):
        """Python and Cython should handle Inf identically."""
        x = np.arange(50, dtype=np.float64)
        y = np.sin(x * 0.2)
        y[15] = np.inf
        y[35] = -np.inf

        x_py, y_py = fpcs_pure.downsample(x, y, ratio=10)
        x_main, y_main = downsample(x, y, ratio=10)

        # Check same Inf positions
        assert np.isinf(y_py).sum() == np.isinf(y_main).sum()
