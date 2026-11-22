"""Alpha158-US OHLCV factors in the original Qlib column order, without VWAP0."""

import numpy as np
import pandas as pd


WINDOWS = (5, 10, 20, 30, 60)
FAMILIES = (
    "ROC",
    "MA",
    "STD",
    "BETA",
    "RSQR",
    "RESI",
    "MAX",
    "MIN",
    "QTLU",
    "QTLD",
    "RANK",
    "RSV",
    "IMAX",
    "IMIN",
    "IMXD",
    "CORR",
    "CORD",
    "CNTP",
    "CNTN",
    "CNTD",
    "SUMP",
    "SUMN",
    "SUMD",
    "VMA",
    "VSTD",
    "WVMA",
    "VSUMP",
    "VSUMN",
    "VSUMD",
)
CANDLE = (
    "KMID",
    "KLEN",
    "KMID2",
    "KUP",
    "KUP2",
    "KLOW",
    "KLOW2",
    "KSFT",
    "KSFT2",
    "OPEN0",
    "HIGH0",
    "LOW0",
)
ORDER = [*CANDLE, *[f + w for f in FAMILIES for w in map(str, WINDOWS)]]


def compute_alpha158_ohlcv(frame):  # ruff: ignore[too-many-locals] -- one output column per explicit Qlib formula
    """Compute the 157 factor columns of the published US DRIFT input."""
    if not frame.index.is_monotonic_increasing or not frame.index.is_unique:
        raise ValueError("Supply one instrument sorted by unique trading day")
    o, h, low, c, v = (frame[k].astype("float32") for k in ("open", "high", "low", "close", "volume"))
    eps = 1e-12
    spread = h - low
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - low
    shift = 2 * c - h - low
    out = dict(
        zip(
            CANDLE,
            [
                (c - o) / o,
                spread / o,
                (c - o) / (spread + eps),
                upper / o,
                upper / (spread + eps),
                lower / o,
                lower / (spread + eps),
                shift / o,
                shift / (spread + eps),
                o / c,
                h / c,
                low / c,
            ],
            strict=True,
        )
    )
    ratio = c / c.shift()
    variation = abs(ratio - 1) * v
    # Qlib gives the independent variable its absolute trading-day offset even when some quotes are missing.
    for w in WINDOWS:

        def roll(s, window=w):
            return s.rolling(window, min_periods=1)

        def corr(left, right):
            r = roll(left).corr(right)
            return r.mask(np.isclose(roll(left).std(), 0, atol=2e-5) | np.isclose(roll(right).std(), 0, atol=2e-5))

        rc = roll(c)
        rh = roll(h)
        rl = roll(low)
        rv = roll(v)
        # A small rolling window around zero avoids subtracting two large absolute trading-date values.
        cy = np.lib.stride_tricks.sliding_window_view(
            np.pad(c.to_numpy(dtype="float64"), (w - 1, 0), constant_values=np.nan), w
        )
        xx = np.arange(w, dtype=float)[None, :]
        mask = np.isfinite(cy)
        n = mask.sum(axis=1)
        mx = (mask * xx).sum(axis=1) / np.maximum(n, 1)
        my = np.nansum(cy, axis=1) / np.maximum(n, 1)
        dx = xx - mx[:, None]
        dy = cy - my[:, None]
        xy = np.nansum(dx * dy, axis=1)
        vx = (mask * dx**2).sum(axis=1)
        vy = np.nansum(dy**2, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            slope = xy / vx
            rsq = xy**2 / (vx * vy)
        hi = rh.max()
        lo = rl.min()
        imax = rh.apply(lambda a: a.argmax() + 1, raw=True)
        imin = rl.apply(lambda a: a.argmin() + 1, raw=True)
        up = roll((c > c.shift()).astype(float)).mean()
        down = roll((c < c.shift()).astype(float)).mean()

        def sums(s):
            change = s - s.shift()
            den = roll(abs(change)).sum() + eps
            a = roll(np.maximum(change, 0)).sum()
            b = roll(np.maximum(-change, 0)).sum()
            return a / den, b / den, (a - b) / den

        sp, sn, sd = sums(c)
        vp, vn, vd = sums(v)
        vals = (
            c.shift(w) / c,
            rc.mean() / c,
            rc.std() / c,
            slope / c,
            pd.Series(rsq, index=c.index).mask(np.isclose(rc.std(), 0, atol=2e-5)),
            (c - my - slope * (w - 1 - mx)) / c,
            hi / c,
            lo / c,
            rc.quantile(0.8) / c,
            rc.quantile(0.2) / c,
            rc.rank(pct=True),
            (c - lo) / (hi - lo + eps),
            imax / w,
            imin / w,
            (imax - imin) / w,
            corr(c, np.log(v + 1)),
            corr(ratio, np.log(v / v.shift() + 1)),
            up,
            down,
            up - down,
            sp,
            sn,
            sd,
            rv.mean() / (v + eps),
            rv.std() / (v + eps),
            roll(variation).std() / (roll(variation).mean() + eps),
            vp,
            vn,
            vd,
        )
        out.update({name + str(w): val for name, val in zip(FAMILIES, vals, strict=True)})
    return pd.DataFrame(out, index=frame.index)[ORDER].astype("float32")
