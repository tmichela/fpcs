import sys

import numpy as np


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


def main() -> int:
    try:
        import fpcs
    except Exception as exc:  # pragma: no cover - best effort in a tiny smoke test
        return _fail(f"could not import fpcs: {exc}")

    x = np.arange(0, 20, dtype=np.float64)
    y = np.array(
        [0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 4, 0, 3, 0, 2, 0, 1, 0, 0],
        dtype=np.float64,
    )
    ratio = 5

    try:
        x_out, y_out = fpcs.downsample(x, y, ratio=ratio)
    except Exception as exc:
        return _fail(f"downsample failed: {exc}")

    if len(x_out) == 0 or len(y_out) == 0:
        return _fail("downsample returned empty output")
    if len(x_out) != len(y_out):
        return _fail("downsample returned mismatched output lengths")
    if x_out[0] != x[0] or x_out[-1] != x[-1]:
        return _fail("endpoints were not preserved")
    if np.any(np.diff(x_out) < 0):
        return _fail("output x is not sorted")
    if len(x_out) >= len(x):
        return _fail("output was not downsampled")

    input_pairs = set(zip(x.tolist(), y.tolist()))
    for xi, yi in zip(x_out, y_out):
        if (float(xi), float(yi)) not in input_pairs:
            return _fail("output contains a point not present in input")

    try:
        downsampler = fpcs.FPCSDownsampler(ratio=ratio)
        stream_out = []
        for xi, yi in zip(x, y):
            stream_out.extend(list(downsampler.add(float(xi), float(yi))))
        stream_out.extend(list(downsampler.flush()))
    except Exception as exc:
        return _fail(f"streaming API failed: {exc}")

    if len(stream_out) == 0:
        return _fail("streaming API produced no output")

    backend = getattr(fpcs, "get_backend", lambda: "unknown")()
    version = getattr(fpcs, "__version__", "unknown")
    print(f"OK: fpcs {version} backend={backend} output={len(x_out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

