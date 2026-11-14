#!/usr/bin/env python3
"""polymarket-faa-attribution-02 entry (Python 3.13, standard library only).

Forecasts the Yes price of every market of a Polymarket event family,
minute by minute, 4 hours ahead: for each market and each of the 240
minutes after the checkpoint, five quantiles (0.05/0.25/0.5/0.75/0.95)
of that minute's price, scored by mean pinball loss.

Model (selected by local validation, zero tuned parameters):
empirical quantiles of the market's own one-minute moves, scaled by
sqrt(h), no drift. For one market with lookback prices p_0..p_{n-1}
(chronological, up to 4,320) and checkpoint price x = p_{n-1}:
    d_i = p_{i+1} - p_i over ALL lookback minutes (no window),
    Q(tau) = the linearly interpolated empirical quantile of the d_i,
    q_tau(h) = clamp(x + Q(tau) * sqrt(h), 0, 1), rounded to 6 decimals.
Adaptive-sigma, drift, jump-aware and truncated-normal variants were
all swept and discarded: each won one regime and lost another, while
the empirical quantiles had the smallest worst case, the best pooled
score on the FAA markets' own history, and the best practice score.
The honest expectation is a coin flip against the house baseline; the
mechanism fits the final regime (long lookbacks dominated by quiet
minutes, with past jumps kept in the tails), and it wins big in
jump/collapse regimes while losing small in quiet ones.

Degenerate inputs (never occur in a scored case) fall back to the
house baseline's rows: fewer than 2 moves. A flat market needs no
special case: every Q(tau) is 0, so every quantile is x.

Runs offline, deterministic (no randomness, no clock, no network),
and far inside the 8-second cap: one sort per market per case.
"""

import json
import math
import os
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

TAUS = (0.05, 0.25, 0.5, 0.75, 0.95)
# Standard normal quantiles, used only by the degenerate-input fallback.
# The model itself has no tuned parameters: do not re-tune anything here
# against the practice score without a fresh, untouched confirmation set.
Z = (-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722)
BASELINE_SIGMA = 0.0025


def clamp01(v: float) -> float:
    return min(1.0, max(0.0, v))


def baseline_rows(x: float, horizon: int) -> list:
    """House baseline rows: the fallback for degenerate inputs only."""
    return [
        [round(clamp01(x + Z[j] * BASELINE_SIGMA * math.sqrt(h)), 6) for j in range(5)]
        for h in range(1, horizon + 1)
    ]


def empirical_quantiles(sorted_diffs: list, m: int) -> list:
    """Linearly interpolated empirical quantiles at TAUS (exact formula)."""
    out = []
    for tau in TAUS:
        pos = tau * (m - 1)
        lo = math.floor(pos)
        hi = min(lo + 1, m - 1)
        frac = pos - lo
        out.append(sorted_diffs[lo] * (1.0 - frac) + sorted_diffs[hi] * frac)
    return out


def forecast_market(prices: list, horizon: int) -> list:
    """240 rows x 5 quantiles for one market's lookback prices."""
    if not prices:
        raise ValueError("market has no prices")
    x = float(prices[-1])
    if x != x or x in (float("inf"), float("-inf")):
        raise ValueError("checkpoint price is not a finite number")
    moves = len(prices) - 1
    if moves < 2:
        return baseline_rows(x, horizon)
    diffs = sorted(float(prices[i + 1]) - float(prices[i]) for i in range(moves))
    quantiles = empirical_quantiles(diffs, moves)
    return [
        [round(clamp01(x + quantiles[j] * math.sqrt(h)), 6) for j in range(5)]
        for h in range(1, horizon + 1)
    ]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inputs = sorted(p for p in INPUT_DIR.iterdir() if p.name.endswith(".in"))
    for path in inputs:
        case_id = path.name[: -len(".in")]
        case = json.loads(path.read_text(encoding="utf-8"))
        horizon = int(case.get("horizon_minutes", 240))
        forecasts = []
        for market in case["markets"]:
            prices = market.get("prices")
            if not isinstance(prices, list) or not prices:
                raise ValueError(f"{path.name} market {market.get('id')} holds no prices")
            forecasts.append({"id": market["id"], "q": forecast_market(prices, horizon)})
        out = {"forecasts": forecasts}
        (OUTPUT_DIR / f"{case_id}.out").write_text(
            json.dumps(out) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
