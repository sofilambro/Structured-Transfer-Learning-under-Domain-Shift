"""
TinyImageNet-C-style corruption suite (paper, Appendix A.1).

Synthetic robustness stress test for Experiment 1, built with implementations
matching the ImageNet-C corruption functions of Hendrycks & Dietterich (2019).
The paper uses six corruptions, each with its own severity ceiling ::

    brightness    severities {1, 2, 3, 4}
    contrast      severities {1, 2, 3}
    defocus blur  severities {1, 2}
    fog           severities {1, 2, 3, 4, 5}
    saturate      severities {1, 2, 3}
    snow          severities {1, 2, 3, 4}

The ceilings differ because the corruptions are not equally destructive at
64x64: defocus blur at high severity erases a small image entirely, whereas fog
remains legible across its full range.

Construction: for each target subset the 5,000 validation images (100 classes x
50) are shuffled with a fixed seed and partitioned into **six non-overlapping
groups**, one per corruption. Each image then draws its severity uniformly from
that corruption's range. So every image is corrupted exactly once and the six
corruptions are equally represented -- unlike ImageNet-C, which materializes
every image at every severity. This keeps the evaluation set the same size as
the clean one, so corruption accuracy and clean accuracy are directly comparable.

Reported as top-1 accuracy averaged over the corrupted set. The paper finds it
tracks clean accuracy almost perfectly (Pearson r = 0.980 across the 56
source-dependent runs), i.e. no separate clean/robust trade-off appears here.

Severity constants follow the ImageNet-C reference implementation. Because that
implementation is tuned for 224x224 inputs, a few parameters are rescaled for
64x64 and are marked individually below.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image

#: Corruption name -> maximum severity, per Appendix A.1.
CORRUPTIONS: dict[str, int] = {
    "brightness":   4,
    "contrast":     3,
    "defocus_blur": 2,
    "fog":          5,
    "saturate":     3,
    "snow":         4,
}


def _as_float(img: Image.Image) -> np.ndarray:
    """PIL image -> float array in [0, 1], shape (H, W, 3)."""
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def _as_image(arr: np.ndarray) -> Image.Image:
    """float array in [0, 1] -> PIL image, clipped."""
    return Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8))


def _rgb_to_hsv(arr: np.ndarray) -> np.ndarray:
    import colorsys
    flat = arr.reshape(-1, 3)
    out = np.array([colorsys.rgb_to_hsv(*px) for px in flat], dtype=np.float32)
    return out.reshape(arr.shape)


def _hsv_to_rgb(arr: np.ndarray) -> np.ndarray:
    import colorsys
    flat = arr.reshape(-1, 3)
    out = np.array([colorsys.hsv_to_rgb(*px) for px in flat], dtype=np.float32)
    return out.reshape(arr.shape)


# Every corruption takes the same signature ``(img, severity, rng)``. Only the
# stochastic ones (fog, snow) use ``rng``; the rest accept and ignore it so the
# builder can dispatch uniformly. The builder hands each image its own generator,
# so the fog field and snow speckle pattern differ per image while staying fully
# reproducible from the corruption seed.


def brightness(img: Image.Image, severity: int, rng=None) -> Image.Image:
    """Additive shift in HSV value. ImageNet-C constants. Deterministic."""
    c = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
    hsv = _rgb_to_hsv(_as_float(img))
    hsv[..., 2] = np.clip(hsv[..., 2] + c, 0, 1)
    return _as_image(_hsv_to_rgb(hsv))


def contrast(img: Image.Image, severity: int, rng=None) -> Image.Image:
    """Scale deviation from the per-image mean. ImageNet-C constants. Deterministic."""
    c = [0.4, 0.3, 0.2, 0.1, 0.05][severity - 1]
    arr  = _as_float(img)
    mean = arr.mean(axis=(0, 1), keepdims=True)
    return _as_image((arr - mean) * c + mean)


def defocus_blur(img: Image.Image, severity: int, rng=None) -> Image.Image:
    """
    Convolution with a disc kernel. Deterministic.

    Radii are scaled down for 64x64 input: the ImageNet-C values (radius 3-10)
    assume 224x224 and would obliterate a TinyImageNet image. This is one of the
    parameters the paper's severity ceiling of 2 also guards against.
    """
    radius, alias_blur = [(1, 0.1), (2, 0.1), (3, 0.1), (4, 0.5), (5, 0.5)][severity - 1]
    kernel = _disk_kernel(radius=radius, alias_blur=alias_blur)

    arr = _as_float(img)
    channels = [_convolve2d(arr[..., ch], kernel) for ch in range(3)]
    return _as_image(np.stack(channels, axis=-1))


def fog(img: Image.Image, severity: int, rng=None) -> Image.Image:
    """Additive plasma-fractal haze, then renormalize to the original max."""
    rng = _as_rng(rng)
    c = [(1.5, 2), (2.0, 2), (2.5, 1.7), (2.5, 1.5), (3.0, 1.4)][severity - 1]
    arr = _as_float(img)
    max_val = arr.max()
    h, w = arr.shape[:2]
    haze = _plasma_fractal(size=_next_power_of_two(max(h, w)),
                           wibbledecay=c[1], rng=rng)[:h, :w][..., np.newaxis]
    arr = arr + c[0] * haze
    return _as_image(arr * max_val / max(max_val + c[0], 1e-6))


def saturate(img: Image.Image, severity: int, rng=None) -> Image.Image:
    """Scale and shift HSV saturation. ImageNet-C constants. Deterministic."""
    c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity - 1]
    hsv = _rgb_to_hsv(_as_float(img))
    hsv[..., 1] = np.clip(hsv[..., 1] * c[0] + c[1], 0, 1)
    return _as_image(_hsv_to_rgb(hsv))


def snow(img: Image.Image, severity: int, rng=None) -> Image.Image:
    """
    Motion-blurred sparse bright speckles composited over a desaturated image.

    The ImageNet-C original applies a Wand/ImageMagick motion blur; this uses a
    directional box filter instead, to keep the dependency footprint to NumPy and
    Pillow. Visually equivalent at 64x64.
    """
    rng = _as_rng(rng)
    c = [
        (0.1, 0.2, 1, 0.6, 8, 3, 0.95),
        (0.1, 0.2, 1, 0.5, 10, 4, 0.9),
        (0.15, 0.3, 1.75, 0.55, 10, 4, 0.9),
        (0.25, 0.3, 2.25, 0.6, 12, 6, 0.85),
        (0.3, 0.3, 1.25, 0.65, 14, 12, 0.8),
    ][severity - 1]

    arr = _as_float(img)
    h, w = arr.shape[:2]

    layer = rng.normal(size=(h, w), loc=c[0], scale=c[1])
    layer = _zoom_nearest(layer, c[2])[:h, :w]
    layer[layer < c[3]] = 0
    layer = _motion_blur(np.clip(layer, 0, 1), radius=c[4],
                         angle_deg=rng.uniform(-135, -45))

    # Composite over a partially desaturated base, so snow reads as overcast
    # rather than as coloured noise.
    grey = arr.mean(axis=2, keepdims=True)
    base = c[6] * arr + (1 - c[6]) * np.maximum(arr, grey * 1.5 + 0.5)
    return _as_image(base + layer[..., np.newaxis] + np.rot90(layer, k=2)[..., np.newaxis])


def _as_rng(rng) -> np.random.Generator:
    """Accept a Generator, an int seed, or None."""
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


#: Dispatch table used by the builder.
CORRUPTION_FNS = {
    "brightness":   brightness,
    "contrast":     contrast,
    "defocus_blur": defocus_blur,
    "fog":          fog,
    "saturate":     saturate,
    "snow":         snow,
}


def build_corrupted_set(
    images: list[tuple[str, int]],
    load_image,
    out_dir: str | Path,
    seed: int = 0,
    corruptions: dict[str, int] | None = None,
) -> Path:
    """
    Materialize a corrupted validation set and its manifest.

    Args:
        images:      ``[(filename, class_index), ...]`` for the clean split.
        load_image:  ``filename -> PIL.Image``.
        out_dir:     destination directory; images and ``manifest.csv`` land here.
        seed:        controls both the shuffle and the per-image severity draw,
                     so the corrupted set is reproducible.
        corruptions: override the ``{name: max_severity}`` map; defaults to
                     :data:`CORRUPTIONS`.

    Returns:
        Path to the written ``manifest.csv``.

    The manifest records the original filename, class, corruption, sampled
    severity and that corruption's maximum severity -- enough to recompute
    per-corruption or per-severity breakdowns after the fact without rebuilding.
    """
    corruptions = corruptions or CORRUPTIONS
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Shuffle once, then cut into len(corruptions) contiguous, non-overlapping
    # groups. Every image is corrupted exactly once.
    ordered = list(images)
    random.Random(seed).shuffle(ordered)

    names  = list(corruptions.keys())
    groups = _partition_evenly(ordered, len(names))
    rng    = np.random.default_rng(seed)

    manifest_path = out_dir / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["corrupted_filename", "original_filename", "class",
             "corruption", "severity", "max_severity"]
        )

        for name, group in zip(names, groups):
            max_severity = corruptions[name]
            fn = CORRUPTION_FNS[name]
            for filename, class_idx in group:
                severity = int(rng.integers(1, max_severity + 1))
                # Per-image generator, so fog fields and snow speckle differ
                # across images while remaining a pure function of `seed`.
                image_rng = np.random.default_rng(rng.integers(0, 2 ** 32))
                corrupted = fn(load_image(filename), severity, image_rng)

                stem = Path(filename).stem
                # Encode corruption and severity in the filename so a directory
                # listing alone is enough to audit the set.
                out_name = f"{stem}__{name}_s{severity}.jpg"
                corrupted.save(out_dir / out_name, quality=95)

                writer.writerow(
                    [out_name, filename, class_idx, name, severity, max_severity]
                )

    return manifest_path


def _partition_evenly(items: list, n_groups: int) -> list[list]:
    """Split into ``n_groups`` contiguous chunks differing in size by at most 1."""
    base, extra = divmod(len(items), n_groups)
    groups, start = [], 0
    for i in range(n_groups):
        size = base + (1 if i < extra else 0)
        groups.append(items[start:start + size])
        start += size
    return groups


def _next_power_of_two(n: int) -> int:
    size = 1
    while size < n:
        size *= 2
    return size


def _disk_kernel(radius: int, alias_blur: float = 0.1, dtype=np.float32) -> np.ndarray:
    """Normalized disc kernel, anti-aliased by a small Gaussian (ImageNet-C)."""
    if radius <= 8:
        grid = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        grid = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    xs, ys = np.meshgrid(grid, grid)
    kernel = np.array((xs ** 2 + ys ** 2) <= radius ** 2, dtype=dtype)
    kernel = _gaussian_blur(kernel, sigma=alias_blur, ksize=ksize[0])
    return kernel / max(kernel.sum(), 1e-8)


def _gaussian_blur(arr: np.ndarray, sigma: float, ksize: int) -> np.ndarray:
    """Separable Gaussian blur; avoids a SciPy/OpenCV dependency."""
    if sigma <= 0:
        return arr
    radius = max(ksize // 2, 1)
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel_1d = np.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel_1d /= kernel_1d.sum()
    out = np.apply_along_axis(lambda m: np.convolve(m, kernel_1d, mode="same"), 0, arr)
    return np.apply_along_axis(lambda m: np.convolve(m, kernel_1d, mode="same"), 1, out)


def _convolve2d(channel: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2-D convolution via FFT, with edge padding to avoid wraparound."""
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(channel, ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")

    fft_shape = (padded.shape[0], padded.shape[1])
    result = np.fft.irfft2(
        np.fft.rfft2(padded, fft_shape) * np.fft.rfft2(kernel, fft_shape),
        fft_shape,
    )
    return result[kh - 1:kh - 1 + channel.shape[0], kw - 1:kw - 1 + channel.shape[1]]


def _zoom_nearest(arr: np.ndarray, factor: float) -> np.ndarray:
    """Nearest-neighbour zoom; keeps the snow layer NumPy-only."""
    if factor == 1:
        return arr
    h, w = arr.shape
    new_h, new_w = max(int(h * factor), 1), max(int(w * factor), 1)
    rows = np.clip((np.arange(new_h) / factor).astype(int), 0, h - 1)
    cols = np.clip((np.arange(new_w) / factor).astype(int), 0, w - 1)
    return arr[np.ix_(rows, cols)]


def _motion_blur(arr: np.ndarray, radius: int, angle_deg: float) -> np.ndarray:
    """Directional box blur, standing in for ImageMagick's motion blur."""
    angle = np.deg2rad(angle_deg)
    dx, dy = np.cos(angle), np.sin(angle)

    length = max(int(radius), 1)
    kernel = np.zeros((2 * length + 1, 2 * length + 1), dtype=np.float32)
    for t in range(-length, length + 1):
        row = int(round(length + t * dy))
        col = int(round(length + t * dx))
        kernel[np.clip(row, 0, 2 * length), np.clip(col, 0, 2 * length)] = 1.0
    kernel /= max(kernel.sum(), 1e-8)

    return _convolve2d(arr, kernel)


def _plasma_fractal(size: int = 256, wibbledecay: float = 3.0, rng=None) -> np.ndarray:
    """
    Diamond-square plasma fractal in [0, 1] -- the ImageNet-C fog field.

    ``size`` must be a power of two. The ``wibble * uniform(-wibble, wibble)``
    scaling is carried over from the reference implementation; the field is
    renormalized to [0, 1] at the end, so the absolute scale is irrelevant.
    """
    assert size & (size - 1) == 0, "plasma fractal size must be a power of two"
    rng = _as_rng(rng)
    maparray = np.empty((size, size), dtype=np.float64)
    maparray[0, 0] = 0
    stepsize = size
    wibble = 100.0

    def wibbledmean(array):
        return array / 4 + wibble * rng.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        corners = maparray[0:size:stepsize, 0:size:stepsize]
        squareaccum  = corners + np.roll(corners, 1, axis=0)
        squareaccum += np.roll(squareaccum, 1, axis=1)
        maparray[stepsize // 2:size:stepsize,
                 stepsize // 2:size:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize,
                          stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]

        ldrsum  = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum  = ulgrid + np.roll(ulgrid, -1, axis=1)
        maparray[0:mapsize:stepsize,
                 stepsize // 2:mapsize:stepsize] = wibbledmean(ldrsum + lulsum)

        tdrsum  = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum  = ulgrid + np.roll(ulgrid, -1, axis=0)
        maparray[stepsize // 2:mapsize:stepsize,
                 0:mapsize:stepsize] = wibbledmean(tdrsum + tulsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / max(maparray.max(), 1e-8)
