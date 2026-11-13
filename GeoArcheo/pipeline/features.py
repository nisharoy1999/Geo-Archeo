# pipeline/features.py
#
# PURPOSE:
#   Hand-crafted features shared by build_season_patches.py and
#   pipeline/detect_dem.py, so a patch is described identically everywhere.
#
#   spectral_features  per-band statistics of the central 320 m and of the
#                      whole patch (same formula as detect_crossval.py)
#   season_features    the above for 1 date, and for 4 seasons plus how much
#                      the centre changes over the year (as detect_seasons.py)
#   dem_features       is the centre RAISED above its surroundings, at several
#                      scales (settlement mounds / tells), plus slope and
#                      roughness. Uses relative heights only, so a region's
#                      absolute altitude cannot be learned as a shortcut.

import numpy as np

H = 112          # patch centre (224 / 2)
C = 16           # half-width of the central 320 m


def spectral_features(x):
    """x: (bands, H, W) float, Prithvi-standardized."""
    c = x[:, H - C:H + C, H - C:H + C]
    feats = []
    for region in (c, x):
        r = region.reshape(x.shape[0], -1)
        feats += [r.mean(1), r.std(1), np.percentile(r, 10, axis=1), np.percentile(r, 90, axis=1)]
    feats.append(c.reshape(x.shape[0], -1).mean(1) - x.reshape(x.shape[0], -1).mean(1))
    return np.concatenate(feats)


def standardize(raw, mean, std):
    """raw: (bands, seasons, H, W) int16 -> float32 standardized."""
    m = np.asarray(mean, dtype=np.float32)[:, None, None, None]
    s = np.asarray(std, dtype=np.float32)[:, None, None, None]
    return (raw.astype(np.float32) - m) / s


def season_features(x4):
    """x4: (bands, 4, H, W) standardized. Returns (one_date, four_seasons)."""
    per = [spectral_features(x4[:, t]) for t in range(x4.shape[1])]
    centre = x4[:, :, H - C:H + C, H - C:H + C].mean(axis=(2, 3))     # (bands, 4)
    whole = x4.mean(axis=(2, 3))
    change = [centre.std(axis=1, ddof=1), (centre - whole).std(axis=1, ddof=1)]
    return per[0], np.concatenate(per + change)


DEM_SCALES = (4, 8, 16, 32)     # inner half-widths in pixels: 80, 160, 320, 640 m windows
N_DEM = 2 * len(DEM_SCALES) + 7


def dem_features(d):
    """d: (H, W) elevation in metres (NaN where missing). Returns N_DEM values
    plus nothing absolute: all heights are relative to the surroundings."""
    if d is None or not np.isfinite(d).any() or np.mean(~np.isfinite(d)) > 0.05:
        return np.full(N_DEM, np.nan, dtype=np.float32)
    d = np.where(np.isfinite(d), d, np.nanmedian(d)).astype(np.float32)
    f = []
    for h in DEM_SCALES:
        outer = d[H - 3 * h:H + 3 * h, H - 3 * h:H + 3 * h]
        inner = d[H - h:H + h, H - h:H + h]
        ring = np.concatenate([outer[:h * 2].ravel(), outer[-h * 2:].ravel(),
                               outer[h * 2:-h * 2, :h * 2].ravel(), outer[h * 2:-h * 2, -h * 2:].ravel()])
        f += [inner.mean() - ring.mean(),              # raised above the ring around it?
              inner.max() - np.median(ring)]           # peak above typical surroundings
    gy, gx = np.gradient(d, 10.0)
    slope = np.hypot(gx, gy)
    centre = d[H - C:H + C, H - C:H + C]
    f += [slope[H - C:H + C, H - C:H + C].mean(),      # steep-sided mound?
          slope.mean(),
          slope[H - C:H + C, H - C:H + C].mean() - slope.mean(),
          centre.std(),
          d.std(),
          centre.mean() - d.mean(),                     # centre vs the whole 2.2 km patch
          np.mean(d < centre.mean())]                   # share of the patch lower than the centre
    return np.array(f, dtype=np.float32)
