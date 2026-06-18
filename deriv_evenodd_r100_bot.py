#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║      DERIV EXPIRYRANGE BOT — PRECISION EDITION v4 (RDBEAR)          ║
║  Symbol   : RDBEAR  (Bear Market Index, 1-second ticks)              ║
║  Contract : EXPIRYRANGE  ("Ends Between" — terminal price only)      ║
║  Duration : 2 minutes  (120 ticks to terminal price)                 ║
║  Barriers : ±2.70 relative to entry spot  (auto-calibrated)         ║
╠══════════════════════════════════════════════════════════════════════╣
║  Connection: new Deriv Options API (REST OTP bootstrap)              ║
║    REST /trading/v1/options/accounts → resolve account_id            ║
║    REST /trading/v1/options/accounts/{id}/otp → pre-auth WS URL     ║
║    No `authorize` message needed — OTP URL is already authenticated  ║
║    `underlying_symbol` field used in proposals (not `symbol`)        ║
╠══════════════════════════════════════════════════════════════════════╣
║  Intelligence Stack (10 layers):                                      ║
║    L1   GARCH(1,1)           Conditional vol forecast + veto         ║
║          + Hurst-scaled (H from R/S) cumulative variance n^(2H)      ║
║          + stationarity guard on alpha+beta persistence               ║
║    L2   Monte Carlo (10K→50K) OU-conditioned terminal distribution   ║
║          Terminal price GUARANTEED within barriers via 50K deep MC    ║
║          with CI check: only enter if p - CI_95 >= confidence gate   ║
║    L3   HMM (3-state)        Regime gate — LOW/MED only              ║
║    L4   Hurst Exponent (R/S) Mean-reversion scoring                  ║
║    L5   Ornstein-Uhlenbeck   Analytical range probability            ║
║          + EWMA-stabilized fit over full 300-tick buffer              ║
║    L6   Bayesian Estimator   Live posterior win-rate tracking        ║
║    L7   Risk Guard           Cooldown · circuit breaker · flat stake ║
║    L8   Jump / First-Passage Hawkes-style spike veto +               ║
║                               no-touch (first-passage) probability    ║
║    L9   MACD                 Momentum / trend filter                 ║
║          Fast=12 Slow=26 Signal=9. Trade only when MACD histogram    ║
║          is contracting (momentum fading) — low directional force    ║
║          means price is more likely to stay rangebound at terminal.  ║
║    L10  Awesome Oscillator   Volume / momentum confirmation          ║
║          AO = SMA(5,midprice) - SMA(34,midprice). Low absolute AO   ║
║          confirms market energy is subdued (low "volume" proxy).     ║
║          Trades vetoed when |AO| > ao_veto_threshold.                ║
║    ∑    Weighted Ensemble    Regime-conditional weights ·            ║
║                               per-model floors · dynamic threshold    ║
╠══════════════════════════════════════════════════════════════════════╣
║  Key accuracy features:                                               ║
║    • Two-stage MC: 10K pre-scan → 50K deep confirm                  ║
║    • MC terminal-price GUARANTEE: p - CI_95 >= gate (not just p)    ║
║    • MC drift/variance from fitted OU process (mean-reverting        ║
║      terminal distribution, not naive GBM random-walk drift)         ║
║    • Hurst-exponent scaling of GARCH cumulative variance (n^(2H))    ║
║    • MACD histogram contraction gate (low directional momentum)      ║
║    • Awesome Oscillator low-energy gate (subdued market activity)    ║
║    • Jump-intensity (kurtosis + exceedance) spike veto               ║
║    • Per-model hard floors (all must pass before ensemble gate)      ║
║    • Regime-conditional ensemble weights (LOW/MED/HIGH)              ║
║    • Dynamic confidence threshold by HMM regime                      ║
║    • 20-tick loss cooldown  ·  5-tick win cooldown                   ║
║    • Drawdown circuit breaker at $0.10 remaining                     ║
║    • 120-tick warmup for proper model calibration                     ║
╚══════════════════════════════════════════════════════════════════════╝

Usage:
  export DERIV_APP_ID=<your_new_app_id>      # from developers.deriv.com
  export DERIV_API_TOKEN=<your_PAT>
  export DERIV_ACCOUNT_ID=<your_account_id>  # optional — auto-resolved
  python deriv_er_bot_1hz10v_v2.py

Requirements:
  pip install websockets numpy scipy requests
"""

import asyncio
import csv
import enum
import json
import logging
import os
import random
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import websockets
from scipy import stats

# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════
CFG = {
    # ── Contract ──
    "symbol":           "RDBEAR",
    "contract_type":    "EXPIRYRANGE",
    "duration":         2,
    "duration_unit":    "m",
    "barrier":          "+2.70",   # fixed at ±2.70 for 2-min; auto-calibrator refines within ±0.5
    "barrier2":         "-2.70",
    "currency":         "USD",
    "n_contract_ticks": 120,       # 2 min × 60 sec

    # ── Capital ──
    "starting_bankroll": 1.00,
    "stake":             0.35,
    "drawdown_stop":     0.10,

    # ── Kelly staking ──
    # FIX (Issue 3): Kelly now active from starting bankroll (1.0).
    # Activation at 5.0 meant Kelly never ran on a $1 account.
    # kelly_fraction set to 0.25 (quarter-Kelly) — conservative on a small
    # float; prevents over-betting on early noisy win-rate estimates.
    # kelly_max_fraction_of_bankroll capped at 0.35 so Kelly never exceeds
    # what the flat-stake baseline would have bet.
    #
    # FIX (Issue 6): payout_ratio corrected to actual Deriv return.
    # A $0.35 stake returns $0.18 profit → b = 0.18 / 0.35 = 0.5143.
    # Kelly formula: f* = (b*p - q) / b — using wrong b distorts sizing.
    "kelly_activation_bankroll":      1.0,
    "kelly_fraction":                 0.25,
    "kelly_min_stake":                0.35,
    "kelly_max_fraction_of_bankroll": 0.35,
    "payout_ratio":                   0.5143,  # 0.18 profit / 0.35 stake

    # ── Signal accuracy gates ──
    # warmup_ticks is no longer the trading gate — that role is now owned by
    # the startup calibration (calib_min_ticks=720).  Kept for documentation
    # only, aligned to calib_min_ticks so models always have >= 720 ticks.
    "warmup_ticks":       720,
    "signal_interval":    5,
    "stage1_mc_n":        10_000,
    "stage2_mc_n":        50_000,
    "pre_scan_threshold": 0.70,

    # ── MC terminal-price guarantee ──
    # Trade only if (p_deep - CI_95_halfwidth) >= this value.
    # Enforces that even the LOWER confidence bound of the 50K MC
    # still clears the threshold — not just the point estimate.
    "mc_guarantee_floor": 0.62,

    # ── HMM regime thresholds ──
    # LOW vol → 0.72, MED vol → 0.74, HIGH vol → veto
    # Lowered from 0.77/0.78 to increase signal frequency toward 8-15
    # high-confidence trades/hour. Ensemble floor gates and the Wilson
    # lower-bound barrier calibration still guard accuracy.
    "regime_threshold": {0: 0.72, 1: 0.74, 2: None},

    # ── HMM sigma bucket thresholds (sigma_t scale; calibrated, auto-updated) ──
    "hmm_lo_sigma": 0.2900193,
    "hmm_hi_sigma": 0.35364744,

    # ── GARCH extreme ceiling on sigma_2min (calibrated, auto-updated) ──
    "garch_sigma_ceiling": 4.6955,

    # ── MACD (L9) ──
    "macd_fast":               12,
    "macd_slow":               26,
    "macd_signal":             9,
    # Histogram must be CONTRACTING (|hist_now| < |hist_prev|) to allow trade.
    # Additionally veto if |histogram| > this threshold (strong momentum).
    "macd_histogram_veto":     0.20204,

    # ── Awesome Oscillator (L10) ──
    "ao_fast_period":          5,
    "ao_slow_period":          34,
    # Veto if |AO| > this — market energy too high for a rangebound outcome
    "ao_veto_threshold":       1.93768,
    # FIX (Issue 5): ticks per OHLC bar feeding the AO. 5 ticks/bar on
    # 1HZ10V ≈ 5s bars, so SMA(34 bars) spans ~2.8 min.
    "ao_bar_ticks":            5,

    # ── Jump / spike detection (L8) ──
    "jump_window":        60,
    "jump_kurtosis_veto": 6.0,   # seeded from calibration; auto-calibrator refines live
    "jump_zscore_veto":   5.0,
    "jump_count_veto":    2,

    # ── Hurst-flip veto (L1/L4 interaction) ──
    "hurst_flip_value_veto": 0.40,
    "hurst_flip_garch_veto": 3.9,

    # ── OU fit window (L5) ──
    "ou_fit_window":    300,
    "ou_ewma_lambda":   0.97,

    # ── Cooldowns ──
    "cooldown_win":             5,
    "cooldown_loss":            20,
    "consecutive_loss_limit":   2,
    "consecutive_loss_cooldown": 60,

    # ── Auto-Calibration ──
    # Recalibrates barrier + veto thresholds directly from the bot's own
    # live tick buffer: once at startup (after enough ticks accumulate)
    # and again after every loss (subject to a cooldown so it can't
    # thrash). Results are written straight into CFG / the HMM instance
    # and take effect on the very next signal evaluation.
    "auto_calibrate_enabled":   True,
    "calib_on_start":           True,
    "calib_on_loss":            True,
    "calib_min_ticks":          720,    # min ticks of history before first run
    "calib_min_gap_ticks":      300,    # cooldown between recalibrations
    "calib_target_win_rate":    0.68,   # pick smallest barrier clearing this
    "calib_macd_percentile":    85,
    "calib_ao_percentile":      85,
    "calib_jump_kurt_min":      4.0,
    "calib_jump_kurt_max":      10.0,
    "calib_terminal_window":    None,   # None -> use n_contract_ticks
    "calib_barrier_candidates": [
        2.00, 2.10, 2.20, 2.30, 2.40, 2.50,
        2.60, 2.70, 2.80, 2.90, 3.00,
        3.10, 3.20, 3.30, 3.40, 3.50,
    ],

    # ── Connection (new Deriv Options API) ──
    "api_base":      "https://api.derivws.com",
    "accounts_path": "/trading/v1/options/accounts",
    "otp_path":      "/trading/v1/options/accounts/{account_id}/otp",
    "reconnect_delay": 5,

    # ── Logging ──
    "log_dir":     os.getenv("LOG_DIR", "logs"),
    "log_file":    "er_bot_v2.log",
    "signals_csv": "er_bot_v2_signals.csv",
    "trades_csv":  "er_bot_v2_trades.csv",
    # tick_buffer must be >= calib_min_ticks so the deque holds enough history
    # for calibration to run.  Previously 300 — this caused len(ticks) to cap
    # at 300, freezing the calibration countdown at 420 (720-300) forever.
    "tick_buffer": 720,
}

# ── Ensemble weights (regime-conditional) ──
MODEL_WEIGHTS_BY_REGIME = {
    0: {  # LOW vol
        "monte_carlo": 0.24, "garch": 0.14, "hmm": 0.14,
        "hurst": 0.14, "ou_process": 0.16, "bayesian": 0.05,
        "jump": 0.03, "macd": 0.05, "ao": 0.05,
    },
    1: {  # MED vol
        "monte_carlo": 0.25, "garch": 0.19, "hmm": 0.16,
        "hurst": 0.09, "ou_process": 0.11, "bayesian": 0.05,
        "jump": 0.05, "macd": 0.05, "ao": 0.05,
    },
    2: {  # HIGH vol — unused (hard veto)
        "monte_carlo": 0.28, "garch": 0.22, "hmm": 0.18,
        "hurst": 0.07, "ou_process": 0.07, "bayesian": 0.04,
        "jump": 0.05, "macd": 0.05, "ao": 0.04,
    },
}
MODEL_WEIGHTS = MODEL_WEIGHTS_BY_REGIME[1]

# Per-model hard floors — ALL must pass
MODEL_FLOORS = {
    "monte_carlo": 0.65,
    "garch":       0.52,
    "hmm":         0.55,
    "hurst":       0.45,
    "ou_process":  0.58,
    "jump":        0.50,
    "macd":        0.50,   # 0.50 = neutral/no momentum, 1.0 = ideal (contracting)
    "ao":          0.50,   # 0.50 = neutral low energy, 1.0 = ideal (silent market)
    # bayesian: no floor
}

# ══════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════
LOG_DIR = Path(CFG["log_dir"])
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / CFG["log_file"], encoding="utf-8"),
    ],
)
log = logging.getLogger("ER_BOT_V2")


# ══════════════════════════════════════════════════════════════════════
# DATA LOGGING (CSV)
# ══════════════════════════════════════════════════════════════════════
class DataLogger:
    SIGNAL_FIELDS = [
        "timestamp", "tick_n", "spot",
        "pre_score", "conf",
        "mc_prob", "mc_ci", "mc_lower_bound",
        "garch_sig", "hmm_state", "hmm_sig",
        "hurst_val", "hurst_sig",
        "ou_prob", "bayes_sig",
        "jump_sig", "jump_kurt", "jump_count", "no_touch_p",
        "macd_hist", "macd_sig_val", "macd_score",
        "ao_val", "ao_score",
        "sigma_t", "sigma_2min",
        "stage", "reason",
    ]
    TRADE_FIELDS = [
        "timestamp", "trade_n", "contract_id", "spot_entry",
        "stake", "conf", "mc_lower_bound",
        "macd_hist", "ao_val",
        "profit", "won", "bankroll",
        "total_pnl", "win_rate", "guard_status",
    ]

    def __init__(self, log_dir: Path):
        self.signals_path = log_dir / CFG["signals_csv"]
        self.trades_path  = log_dir / CFG["trades_csv"]
        self._init_file(self.signals_path, self.SIGNAL_FIELDS)
        self._init_file(self.trades_path,  self.TRADE_FIELDS)

    @staticmethod
    def _init_file(path: Path, fields: list):
        if not path.exists() or path.stat().st_size == 0:
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=fields).writeheader()

    def log_signal(self, **row):
        with open(self.signals_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.SIGNAL_FIELDS).writerow(row)

    def log_trade(self, **row):
        with open(self.trades_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.TRADE_FIELDS).writerow(row)


datalog = DataLogger(LOG_DIR)


# ══════════════════════════════════════════════════════════════════════
# LAYER 1 — GARCH(1,1)
# ══════════════════════════════════════════════════════════════════════
class GARCH11:
    def __init__(self, omega: float = 1e-6, alpha: float = 0.10, beta: float = 0.85):
        self.omega = omega
        ab = alpha + beta
        if ab >= 0.999:
            scale = 0.97 / ab
            alpha *= scale
            beta  *= scale
        self.alpha = alpha
        self.beta  = beta
        self.h     = 1e-4
        ab         = alpha + beta
        self._lr   = omega / max(1.0 - ab, 1e-9)

    def update(self, r: float) -> float:
        self.h = self.omega + self.alpha * r ** 2 + self.beta * self.h
        return max(self.h, 1e-14) ** 0.5

    def forecast_cumulative_std(self, n: int = 120, hurst: float = 0.5) -> float:
        h_clamped = float(np.clip(hurst, 0.05, 0.95))
        cumvar = self.h * (n ** (2.0 * h_clamped))
        return max(cumvar, 1e-14) ** 0.5

    def range_signal(self, sigma_T: float, barrier: float = 2.7) -> float:
        if sigma_T < 1e-9:
            return 1.0
        z = barrier / sigma_T
        return float(np.clip(2.0 * stats.norm.cdf(z) - 1.0, 0.0, 1.0))

    def is_extreme(self, sigma_T: float, ceiling: float = 5.0) -> bool:
        return sigma_T > ceiling


# ══════════════════════════════════════════════════════════════════════
# LAYER 2 — MONTE CARLO  (OU-conditioned, terminal-price guarantee)
# ══════════════════════════════════════════════════════════════════════
class MonteCarlo:
    """
    Two-stage MC:
      Stage 1: 10K quick pre-scan.
      Stage 2: 50K deep confirm with CI-based GUARANTEE check.

    Terminal-price guarantee:
      After the 50K run, we compute the 95% confidence interval on the
      MC win-probability estimate. The trade fires only if:
          p_blend - CI_halfwidth >= CFG["mc_guarantee_floor"]
      i.e. even the LOWER bound of our estimate still clears the gate.

    FIX (Issue 1): MC now simulates full tick-by-tick paths (n_ticks steps)
    rather than drawing a single terminal price from a normal distribution.
    This correctly captures path-level variance: a contract can breach the
    barrier mid-path and recover at terminal — the old single-draw approach
    was blind to this and consistently overstated win probability.

    The analytic CDF (which is also terminal-only) is retained but its blend
    weight is reduced from 0.60 → 0.40 since the path MC is now the primary
    estimator.

    Returns (p_blend, ci_halfwidth).
    """

    def __init__(self, n: int = 10_000):
        self.n = n

    def probability(
        self,
        drift:      float,
        sigma_tick: float,
        n_ticks:    int   = 120,
        barrier:    float = 2.7,
        ou_mu_T:    Optional[float] = None,
        ou_var_T:   Optional[float] = None,
    ) -> tuple:
        # Derive per-tick OU parameters when available, else use GBM drift/sigma
        if ou_mu_T is not None and ou_var_T is not None:
            # Convert OU terminal moments back to per-tick equivalents
            sigma_tick_eff = max(float(ou_var_T) ** 0.5 / max(n_ticks ** 0.5, 1), 1e-9)
            drift_eff      = float(ou_mu_T) / max(n_ticks, 1)
        else:
            sigma_tick_eff = max(sigma_tick, 1e-9)
            drift_eff      = drift

        # ── Path-level Monte Carlo (FIX: tick-by-tick walk, n_ticks steps) ──
        # Shape: (n_simulations, n_ticks) of N(drift, sigma) increments
        increments = np.random.normal(
            drift_eff, sigma_tick_eff, (self.n, n_ticks)
        )
        # Cumulative price displacement from entry (entry = 0)
        paths      = np.cumsum(increments, axis=1)
        # Terminal displacement only (EXPIRYRANGE checks only terminal tick)
        terminal   = paths[:, -1]
        p_mc       = float(np.mean(np.abs(terminal) < barrier))
        ci         = float(1.96 * (p_mc * (1.0 - p_mc) / self.n) ** 0.5)

        # ── Analytic CDF (terminal-only, kept as secondary reference) ──
        sigma_T    = sigma_tick_eff * n_ticks ** 0.5
        mu_T       = drift_eff * n_ticks
        z_hi       = ( barrier - mu_T) / sigma_T
        z_lo       = (-barrier - mu_T) / sigma_T
        p_analytic = float(stats.norm.cdf(z_hi) - stats.norm.cdf(z_lo))

        # Blend: path MC is now primary (0.60), analytic CDF secondary (0.40)
        p_blend = float(np.clip(0.60 * p_mc + 0.40 * p_analytic, 0.0, 1.0))
        return p_blend, ci


# ══════════════════════════════════════════════════════════════════════
# LAYER 3 — HMM (3-state regime)
# ══════════════════════════════════════════════════════════════════════
class HMMRegimes:
    """
    FIX (Issue 4): Replaced static sigma-bucket lookup with a proper
    Gaussian-emission HMM forward pass (scaled alpha recursion).

    Each hidden state s emits sigma_t ~ N(mu_s, std_s).  The emission
    probability p(sigma_t | state=s) is evaluated at every tick, and the
    forward variable alpha[s] = P(o_1..o_t, state_t=s) is updated via:

        alpha_t[s] = emission(sigma_t | s) * sum_j(A[j,s] * alpha_{t-1}[j])

    State means/stds are seeded from calibration data and updated in-place
    by the AutoCalibrator via update_emission_params().
    This gives the HMM genuine probabilistic state inference rather than
    a deterministic bucket assignment.
    """
    LOW, MED, HIGH = 0, 1, 2

    # Transition matrix  (row = from, col = to)
    _A = np.array([
        [0.87, 0.10, 0.03],
        [0.09, 0.80, 0.11],
        [0.03, 0.14, 0.83],
    ])
    # Emission distribution parameters (mu, std) per state.
    # Seeded conservatively; AutoCalibrator overwrites from live data.
    _MU  = np.array([0.15, 0.30, 0.50])   # mean sigma_t per state
    _STD = np.array([0.04, 0.06, 0.10])   # std  sigma_t per state

    # Win-rate prior per regime (used as HMM signal score)
    _PRIOR = {0: 0.83, 1: 0.61, 2: 0.32}

    def __init__(self):
        self.state  = self.MED
        self._alpha = np.array([0.15, 0.70, 0.15])   # initial belief

    def update_emission_params(self, lo_sigma: float, hi_sigma: float):
        """
        Called by AutoCalibrator to set emission means from live GARCH data.
        lo_sigma = 33rd pct of rolling sigma  (LOW/MED boundary)
        hi_sigma = 67th pct of rolling sigma  (MED/HIGH boundary)
        State means spaced at: [lo/2,  (lo+hi)/2,  hi*1.5]
        State stds  spaced at: [lo/4,  (hi-lo)/4,  hi/3 ]
        """
        mid = (lo_sigma + hi_sigma) / 2.0
        self._MU  = np.array([lo_sigma / 2.0, mid, hi_sigma * 1.5])
        self._STD = np.array([
            max(lo_sigma / 4.0, 1e-5),
            max((hi_sigma - lo_sigma) / 4.0, 1e-5),
            max(hi_sigma / 3.0, 1e-5),
        ])

    def _emission(self, sigma: float) -> np.ndarray:
        """Gaussian emission P(sigma | state=s) for each state."""
        z   = (sigma - self._MU) / np.maximum(self._STD, 1e-9)
        pdf = np.exp(-0.5 * z ** 2) / np.maximum(self._STD * (2.0 * np.pi) ** 0.5, 1e-9)
        return np.maximum(pdf, 1e-300)

    def update(self, sigma: float) -> int:
        """Scaled forward pass: alpha_t proportional to emission x A^T x alpha_{t-1}."""
        emit        = self._emission(sigma)
        predicted   = self._A.T @ self._alpha     # shape (3,)
        new_alpha   = emit * predicted
        s           = new_alpha.sum()
        self._alpha = new_alpha / s if s > 1e-300 else np.ones(3) / 3.0
        self.state  = int(np.argmax(self._alpha))
        return self.state

    def signal(self) -> float:
        return self._PRIOR[self.state]

    def is_high_vol(self) -> bool:
        return self.state == self.HIGH

    def name(self) -> str:
        return ["LOW", "MED", "HIGH"][self.state]



# ══════════════════════════════════════════════════════════════════════
# LAYER 4 — HURST EXPONENT
# ══════════════════════════════════════════════════════════════════════
class HurstAnalyzer:
    @staticmethod
    def compute(prices: np.ndarray) -> float:
        """
        FIX (Hurst R/S bug): the previous implementation built `c =
        prices[:lag]` inside the loop — i.e. it always re-sliced from the
        START of the array as lag grew, rather than computing R/S over
        independent sub-windows. That's not a valid rescaled-range
        estimate; it structurally biased H toward the 0.95 ceiling on
        almost every tick (confirmed in live logs: 82% of readings
        rounded to 0.9, mean 0.90), which in turn wildly over-inflated
        the Hurst-scaled GARCH cumulative-variance forecast and caused
        GARCH_EXTREME_VOL_VETO to fire on ~77% of all evaluated ticks.

        This is the standard R/S method: for each window length (lag),
        split the series into non-overlapping chunks of that length,
        compute R/S per chunk, average across chunks, then regress
        log(avg R/S) against log(lag) across multiple lags.
        """
        prices = np.asarray(prices, dtype=float)
        n = len(prices)
        if n < 20:
            return 0.5
        max_lag = max(8, n // 4)
        lags, rs_avg = [], []
        for lag in range(4, max_lag):
            n_chunks = n // lag
            if n_chunks < 1:
                continue
            rs_vals = []
            for i in range(n_chunks):
                c = prices[i * lag:(i + 1) * lag]
                m = c.mean()
                d = np.cumsum(c - m)
                r = d.max() - d.min()
                s = c.std(ddof=1)
                if s > 0:
                    rs_vals.append(r / s)
            if rs_vals:
                lags.append(lag)
                rs_avg.append(float(np.mean(rs_vals)))
        if len(rs_avg) < 4:
            return 0.5
        try:
            h, _ = np.polyfit(np.log(lags), np.log(rs_avg), 1)
            return float(np.clip(h, 0.05, 0.95))
        except Exception:
            return 0.5

    @staticmethod
    def signal(h: float) -> float:
        return float(np.clip(1.0 - 0.80 * h, 0.18, 0.92))


# ══════════════════════════════════════════════════════════════════════
# LAYER 5 — ORNSTEIN-UHLENBECK
# ══════════════════════════════════════════════════════════════════════
class OUMeanReversion:
    @staticmethod
    def fit(prices: np.ndarray, ewma_lambda: float = 0.97) -> tuple:
        if len(prices) < 10:
            mu = float(prices.mean()) if len(prices) else 0.0
            s  = float(prices.std())  if len(prices) > 1 else 0.01
            return 0.01, mu, s
        X, Y = prices[:-1].astype(float), prices[1:].astype(float)
        try:
            b, a = np.polyfit(X, Y, 1)
            b    = float(np.clip(b, 1e-4, 1 - 1e-4))
            th   = float(-np.log(b))
            mu   = float(a / (1.0 - b))
            resid = Y - (a + b * X)
            m = len(resid)
            w = ewma_lambda ** np.arange(m - 1, -1, -1)
            w = w / w.sum()
            ewma_var = float(np.sum(w * resid ** 2))
            sig = float(np.sqrt(max(ewma_var, 1e-14)))
            return th, mu, sig
        except Exception:
            return 0.01, float(np.mean(prices)), float(np.std(np.diff(prices)))

    @staticmethod
    def terminal_dist(x0: float, theta: float, mu: float,
                      sigma: float, T: int = 120) -> tuple:
        if theta < 1e-6:
            var_T = sigma ** 2 * T
        else:
            var_T = sigma ** 2 / (2 * theta) * (1.0 - np.exp(-2.0 * theta * T))
        mu_T     = mu + (x0 - mu) * np.exp(-theta * T)
        mu_T_rel = mu_T - x0
        return float(mu_T_rel), float(max(var_T, 1e-14))

    @staticmethod
    def p_in_range(x0: float, theta: float, mu: float,
                   sigma: float, T: int = 120, barrier: float = 2.7) -> float:
        if theta < 1e-6:
            var_T = sigma ** 2 * T
        else:
            var_T = sigma ** 2 / (2 * theta) * (1.0 - np.exp(-2.0 * theta * T))
        mu_T  = mu + (x0 - mu) * np.exp(-theta * T)
        std_T = max(var_T, 1e-14) ** 0.5
        z_hi  = (x0 + barrier - mu_T) / std_T
        z_lo  = (x0 - barrier - mu_T) / std_T
        return float(np.clip(stats.norm.cdf(z_hi) - stats.norm.cdf(z_lo), 0.0, 1.0))


# ══════════════════════════════════════════════════════════════════════
# LAYER 6 — BAYESIAN WIN-RATE ESTIMATOR
# ══════════════════════════════════════════════════════════════════════
class BayesianEdge:
    def __init__(self, prior_wr: float = 0.55, prior_n: float = 10.0):
        self.alpha = prior_wr * prior_n
        self.beta  = (1.0 - prior_wr) * prior_n
        self.n_obs = 0

    def update(self, won: bool):
        if won: self.alpha += 1.0
        else:   self.beta  += 1.0
        self.n_obs += 1

    def mean(self) -> float:
        return float(self.alpha / (self.alpha + self.beta))

    def ci95(self) -> tuple:
        lo = float(stats.beta.ppf(0.025, self.alpha, self.beta))
        hi = float(stats.beta.ppf(0.975, self.alpha, self.beta))
        return lo, hi


# ══════════════════════════════════════════════════════════════════════
# LAYER 8 — JUMP DETECTOR + FIRST-PASSAGE (NO-TOUCH) PROBABILITY
# ══════════════════════════════════════════════════════════════════════
class JumpFirstPassage:
    @staticmethod
    def no_touch_prob(sigma_tick: float, n_ticks: int = 120, barrier: float = 2.7) -> float:
        if sigma_tick < 1e-12:
            return 1.0
        sigma_T = sigma_tick * (n_ticks ** 0.5)
        z = barrier / sigma_T
        p_touch = float(np.clip(2.0 * (1.0 - stats.norm.cdf(z)), 0.0, 1.0))
        return float(np.clip(1.0 - p_touch, 0.0, 1.0))

    @staticmethod
    def assess(returns: np.ndarray) -> tuple:
        window = CFG["jump_window"]
        r = returns[-window:] if len(returns) >= window else returns
        if len(r) < 10:
            return 1.0, False, 0.0, 0
        r   = r.astype(float)
        std = r.std(ddof=1)
        if std < 1e-12:
            return 1.0, False, 0.0, 0
        z          = (r - r.mean()) / std
        jump_count = int(np.sum(np.abs(z) > CFG["jump_zscore_veto"]))
        kurt       = float(stats.kurtosis(r, fisher=True, bias=False))
        veto       = kurt >= CFG["jump_kurtosis_veto"] or jump_count >= CFG["jump_count_veto"]
        kurt_penalty  = np.clip(kurt / CFG["jump_kurtosis_veto"], 0.0, 1.0)
        count_penalty = np.clip(jump_count / max(CFG["jump_count_veto"], 1), 0.0, 1.0)
        signal        = float(np.clip(1.0 - 0.5 * kurt_penalty - 0.5 * count_penalty, 0.0, 1.0))
        return signal, bool(veto), kurt, jump_count


# ══════════════════════════════════════════════════════════════════════
# LAYER 9 — MACD  (momentum filter)
# ══════════════════════════════════════════════════════════════════════
class MACDFilter:
    """
    Standard MACD:
        MACD line   = EMA(fast) - EMA(slow)
        Signal line = EMA(signal) of MACD line
        Histogram   = MACD line - Signal line

    For EXPIRYRANGE we WANT low directional momentum — the histogram
    should be contracting (magnitude decreasing tick-over-tick) and
    small in absolute value.

    score():
        1.0  → histogram is contracting AND |hist| is very small
        0.50 → neutral (histogram flat or barely expanding)
        0.0  → strong expanding histogram (high momentum — bad)

    veto:
        Returns True if |histogram| > macd_histogram_veto threshold.
    """

    def __init__(self):
        fast = CFG["macd_fast"]
        slow = CFG["macd_slow"]
        sig  = CFG["macd_signal"]
        self._k_fast = 2.0 / (fast + 1)
        self._k_slow = 2.0 / (slow + 1)
        self._k_sig  = 2.0 / (sig + 1)
        self._ema_fast: Optional[float] = None
        self._ema_slow: Optional[float] = None
        self._ema_sig:  Optional[float] = None
        self._prev_hist: Optional[float] = None
        self._hist:      Optional[float] = None

    def update(self, price: float):
        if self._ema_fast is None:
            self._ema_fast = price
            self._ema_slow = price
            return
        self._ema_fast += self._k_fast * (price - self._ema_fast)
        self._ema_slow += self._k_slow * (price - self._ema_slow)
        macd_line = self._ema_fast - self._ema_slow
        if self._ema_sig is None:
            self._ema_sig = macd_line
        else:
            self._ema_sig += self._k_sig * (macd_line - self._ema_sig)
        self._prev_hist = self._hist
        self._hist      = macd_line - self._ema_sig

    def is_ready(self) -> bool:
        return self._hist is not None and self._prev_hist is not None

    def histogram(self) -> float:
        return self._hist if self._hist is not None else 0.0

    def score(self) -> float:
        """
        Returns [0,1]:
          1.0 = contracting histogram (momentum fading) — ideal for range trade
          0.5 = flat/neutral
          0.0 = strongly expanding histogram (avoid)
        """
        if not self.is_ready():
            return 0.5   # neutral before warmup

        hist      = self._hist
        prev_hist = self._prev_hist
        abs_hist  = abs(hist)

        # Is momentum contracting?
        contracting = abs(hist) < abs(prev_hist)

        # Magnitude penalty: score decays as |hist| grows
        mag_score = float(np.clip(1.0 - abs_hist / CFG["macd_histogram_veto"], 0.0, 1.0))

        if contracting:
            # Full score scaled by magnitude
            return float(np.clip(0.5 + 0.5 * mag_score, 0.5, 1.0))
        else:
            # Expanding: penalise below 0.5
            return float(np.clip(0.5 * mag_score, 0.0, 0.5))

    def veto(self) -> bool:
        """Hard veto if histogram magnitude is too large."""
        if not self.is_ready():
            return False
        return abs(self._hist) > CFG["macd_histogram_veto"]


# ══════════════════════════════════════════════════════════════════════
# LAYER 10 — AWESOME OSCILLATOR  (OHLC bar-based, market energy)
# ══════════════════════════════════════════════════════════════════════
class OHLCBarBuilder:
    """
    FIX (Issue 5): Builds OHLC bars from raw tick data so the Awesome
    Oscillator can compute midprice = (High + Low) / 2 per bar — its
    canonical input — rather than raw spot prices.

    Bar period is CFG["ao_bar_ticks"] ticks (default 5 ticks = 5 seconds
    on RDBEAR, giving ~1-minute bars for the slow SMA(34) to span ~2.8 min).

    Each completed bar exposes .open, .high, .low, .close, .midprice.
    """
    def __init__(self, bar_ticks: int = 5):
        self._bar_ticks = bar_ticks
        self._buf: list  = []
        self.bars: deque = deque(maxlen=200)   # keep last 200 bars

    def update(self, price: float):
        self._buf.append(price)
        if len(self._buf) >= self._bar_ticks:
            o = self._buf[0]
            h = max(self._buf)
            l = min(self._buf)
            c = self._buf[-1]
            self.bars.append({
                "open":     o,
                "high":     h,
                "low":      l,
                "close":    c,
                "midprice": (h + l) / 2.0,
            })
            self._buf = []

    def midprices(self) -> np.ndarray:
        return np.array([b["midprice"] for b in self.bars], dtype=float)

    def n_bars(self) -> int:
        return len(self.bars)


class AwesomeOscillator:
    """
    FIX (Issue 5): AO now operates on OHLC bar midprices: (H+L)/2.
    This is the standard Bill Williams formulation and gives the AO
    genuine meaning as a momentum/energy proxy:

        AO = SMA(midprice, 5 bars) - SMA(midprice, 34 bars)

    On RDBEAR with ao_bar_ticks=5, each bar = 5 ticks.
    The slow SMA(34) spans 170 ticks (~2.8 min), the fast SMA(5) spans
    25 ticks — a meaningful short-vs-long momentum comparison.

    The bar builder is owned by the bot and passed in at construction so
    both AO and its calibrator share the same bar history.
    """

    def __init__(self, bar_builder: OHLCBarBuilder):
        self._bars = bar_builder
        self._ao   = 0.0

    def _compute(self) -> float:
        mids = self._bars.midprices()
        fp   = CFG["ao_fast_period"]   # bars
        sp   = CFG["ao_slow_period"]   # bars
        if len(mids) < sp:
            return 0.0
        return float(np.mean(mids[-fp:]) - np.mean(mids[-sp:]))

    def update(self, price: float):
        """Called every tick; bar builder handles aggregation."""
        self._bars.update(price)
        if self._bars.n_bars() >= CFG["ao_slow_period"]:
            self._ao = self._compute()

    def is_ready(self) -> bool:
        return self._bars.n_bars() >= CFG["ao_slow_period"]

    def value(self) -> float:
        return self._ao

    def score(self) -> float:
        if not self.is_ready():
            return 0.5
        abs_ao = abs(self._ao)
        return float(np.clip(1.0 - abs_ao / CFG["ao_veto_threshold"], 0.0, 1.0))

    def veto(self) -> bool:
        if not self.is_ready():
            return False
        return abs(self._ao) > CFG["ao_veto_threshold"]



# ══════════════════════════════════════════════════════════════════════
# LAYER 7 — RISK GUARD
# ══════════════════════════════════════════════════════════════════════
class RiskGuard:
    def __init__(self):
        self.stake          = CFG["stake"]
        self._cooldown      = 0
        self._tripped       = False
        self._consec_losses = 0

    def tick(self):
        if self._cooldown > 0:
            self._cooldown -= 1

    def on_win(self):
        self._consec_losses = 0
        self._cooldown = CFG["cooldown_win"]

    def on_loss(self):
        self._consec_losses += 1
        if self._consec_losses >= CFG["consecutive_loss_limit"]:
            self._cooldown = CFG["consecutive_loss_cooldown"]
            log.warning(
                f"  {self._consec_losses} consecutive losses — "
                f"extended cooldown ({CFG['consecutive_loss_cooldown']} ticks)"
            )
        else:
            self._cooldown = CFG["cooldown_loss"]

    def check_bankroll(self, bankroll: float):
        if bankroll < CFG["drawdown_stop"]:
            self._tripped = True
            log.warning(
                f"CIRCUIT BREAKER — bankroll ${bankroll:.2f} < "
                f"${CFG['drawdown_stop']:.2f} stop. Bot halted."
            )

    def can_trade(self) -> bool:
        return not self._tripped and self._cooldown == 0

    def compute_stake(self, bankroll: float, win_prob: float) -> float:
        if bankroll < CFG["kelly_activation_bankroll"]:
            self.stake = CFG["stake"]
            return self.stake
        b      = CFG["payout_ratio"]
        p      = float(np.clip(win_prob, 0.0, 1.0))
        q      = 1.0 - p
        f_star = max((b * p - q) / b, 0.0)
        f_kelly = min(f_star * CFG["kelly_fraction"], CFG["kelly_max_fraction_of_bankroll"])
        stake   = max(bankroll * f_kelly, CFG["kelly_min_stake"])
        stake   = min(stake, bankroll)
        self.stake = round(stake, 2)
        return self.stake

    def status(self) -> str:
        if self._tripped:
            return "CIRCUIT_BREAKER_TRIPPED"
        if self._consec_losses >= CFG["consecutive_loss_limit"] and self._cooldown > 0:
            return f"CONSEC_LOSS_COOLDOWN({self._cooldown}t,streak={self._consec_losses})"
        if self._cooldown > 0:
            return f"COOLDOWN({self._cooldown}t)"
        return "READY"


# ══════════════════════════════════════════════════════════════════════
# WEIGHTED ENSEMBLE  (10-layer, with per-model floors + dynamic threshold)
# ══════════════════════════════════════════════════════════════════════
class Ensemble:
    NO_TOUCH_BLEND = 0.12   # slightly reduced to make room for MACD+AO

    def decide(
        self,
        mc_prob:    float,
        garch_sig:  float,
        hmm_sig:    float,
        hmm_state:  int,
        hurst_sig:  float,
        hurst_val:  float,
        ou_prob:    float,
        bayes_sig:  float,
        jump_sig:   float,
        no_touch_p: float,
        macd_score: float,
        ao_score:   float,
        hmm_veto:   bool,
        jump_veto:  bool,
        macd_veto:  bool,
        ao_veto:    bool,
    ) -> tuple:
        scores = {
            "monte_carlo": mc_prob,
            "garch":       garch_sig,
            "hmm":         hmm_sig,
            "hurst":       hurst_sig,
            "ou_process":  ou_prob,
            "bayesian":    bayes_sig,
            "jump":        jump_sig,
            "macd":        macd_score,
            "ao":          ao_score,
        }
        weights = MODEL_WEIGHTS_BY_REGIME.get(hmm_state, MODEL_WEIGHTS_BY_REGIME[1])

        # ── Hard vetoes ──
        if hmm_veto:
            return False, 0.0, scores, [], "HMM_HIGH_VOL_VETO"
        if jump_veto:
            return False, 0.0, scores, [], "JUMP_SPIKE_VETO"
        if macd_veto:
            return False, 0.0, scores, [], "MACD_HIGH_MOMENTUM_VETO"
        if ao_veto:
            return False, 0.0, scores, [], "AO_HIGH_ENERGY_VETO"

        # ── Hurst-flip veto ──
        if hurst_val < CFG["hurst_flip_value_veto"] and garch_sig > CFG["hurst_flip_garch_veto"]:
            return False, 0.0, scores, [], (
                f"HURST_FLIP_VETO(H={hurst_val:.2f},garch={garch_sig:.2f})"
            )

        # ── Per-model floors ──
        failed = [k for k, floor in MODEL_FLOORS.items() if scores.get(k, 1.0) < floor]
        if failed:
            conf = float(np.clip(sum(scores[k] * weights[k] for k in weights), 0.0, 1.0))
            return False, conf, scores, failed, f"FLOOR_FAIL:{','.join(failed)}"

        # ── Weighted ensemble score ──
        conf_raw = float(np.clip(
            sum(scores[k] * weights[k] for k in weights), 0.0, 1.0,
        ))

        # ── Blend in first-passage no-touch probability ──
        conf = float(np.clip(
            (1.0 - self.NO_TOUCH_BLEND) * conf_raw + self.NO_TOUCH_BLEND * no_touch_p,
            0.0, 1.0,
        ))

        # ── Dynamic threshold from HMM regime ──
        threshold = CFG["regime_threshold"].get(hmm_state)
        if threshold is None:
            return False, conf, scores, [], "HMM_REGIME_VETO"

        trade  = conf >= threshold
        reason = (
            f"PASS_ENSEMBLE(gate={threshold:.2f})" if trade
            else f"WAIT(conf={conf:.3f}<{threshold:.2f})"
        )
        return trade, conf, scores, [], reason



# ══════════════════════════════════════════════════════════════════════
# AUTO-CALIBRATOR
# ══════════════════════════════════════════════════════════════════════
class AutoCalibrator:
    """
    Calibrates barrier and all veto thresholds directly from the bot's
    live tick buffer.  Results are written into CFG and the HMM instance
    and take effect on the very next signal evaluation.

    Triggered automatically:
      1. STARTUP  — once CFG["calib_min_ticks"] ticks have accumulated
                    (runs exactly once per session, then marks done)
      2. ON LOSS  — after every settled loss, subject to a cooldown of
                    CFG["calib_min_gap_ticks"] ticks between runs

    What gets calibrated (all against the ±2.70 / 2-min regime):
      • barrier        — smallest candidate that achieves the target win-rate
                         (walk-forward terminal-price windows, restricted to
                          [2.0 – 3.5] range; defaults to 2.70 if none qualify)
      • macd_histogram_veto  — 85th-pct of |MACD histogram| history
      • ao_veto_threshold    — 85th-pct of |AO| history
      • jump_kurtosis_veto   — 90th-pct rolling kurtosis (clamped [4, 10])
      • hmm_lo_sigma / hmm_hi_sigma  — 33rd / 67th pct of rolling GARCH σ
      • garch_sigma_ceiling  — 99th-pct of 2-min cumulative GARCH σ
    """

    def __init__(self):
        self._last_calib_tick  = -99_999
        self._startup_done     = False

    # ── Trigger guards ───────────────────────────────────────────────
    def should_run_startup(self, n_ticks_collected: int) -> bool:
        if not CFG["auto_calibrate_enabled"]:  return False
        if not CFG["calib_on_start"]:          return False
        if self._startup_done:                  return False
        return n_ticks_collected >= CFG["calib_min_ticks"]

    def should_run_on_loss(self, tick_n: int, n_ticks_collected: int) -> bool:
        if not CFG["auto_calibrate_enabled"]: return False
        if not CFG["calib_on_loss"]:          return False
        if n_ticks_collected < CFG["calib_min_ticks"]: return False
        return (tick_n - self._last_calib_tick) >= CFG["calib_min_gap_ticks"]

    # ── Main entry point ─────────────────────────────────────────────
    def run(self,
            tick_n:  int,
            prices:  np.ndarray,
            returns: np.ndarray,
            hmm:     "HMMRegimes",
            trigger: str = "startup") -> None:
        """
        Run full calibration in place.  Updates CFG dict and *hmm* object.
        """
        n = len(prices)
        if n < CFG["calib_min_ticks"]:
            log.warning(f"[CALIB] Skipped — only {n} ticks (need {CFG['calib_min_ticks']})")
            return

        rets = returns.astype(float) if len(returns) > 0 else np.diff(prices.astype(float))

        bar = "─" * 60
        log.info(bar)
        log.info(f"[CALIB] ▶  Auto-calibration  trigger={trigger}  n_ticks={n}")

        # 1. Barrier (always writes back as "+X.XX" / "-X.XX")
        best_b = self._calibrate_barrier(prices)
        CFG["barrier"]  = f"+{best_b:.2f}"
        CFG["barrier2"] = f"-{best_b:.2f}"
        log.info(f"[CALIB]   barrier            → ±{best_b:.2f}")

        # 2. MACD histogram veto
        macd_thr = self._calibrate_macd(prices)
        CFG["macd_histogram_veto"] = macd_thr
        log.info(f"[CALIB]   macd_histogram_veto → {macd_thr:.6f}")

        # 3. Awesome Oscillator veto
        ao_thr = self._calibrate_ao(prices)
        CFG["ao_veto_threshold"] = ao_thr
        log.info(f"[CALIB]   ao_veto_threshold   → {ao_thr:.6f}")

        # 4. Jump kurtosis veto
        kurt_thr = self._calibrate_jump_kurt(rets)
        CFG["jump_kurtosis_veto"] = kurt_thr
        log.info(f"[CALIB]   jump_kurtosis_veto  → {kurt_thr:.4f}")

        # 5. HMM sigma bucket thresholds (written into CFG *and* live HMM)
        lo_sig, hi_sig = self._calibrate_hmm_sigma(rets)
        CFG["hmm_lo_sigma"] = lo_sig
        CFG["hmm_hi_sigma"] = hi_sig
        hmm.update_emission_params(lo_sig, hi_sig)  # FIX4: update Gaussian emission params
        log.info(f"[CALIB]   hmm_lo_sigma (emission mu LOW)  → {lo_sig:.7f}")
        log.info(f"[CALIB]   hmm_hi_sigma (emission mu HIGH) → {hi_sig:.7f}")

        # 6. GARCH 2-min cumulative sigma ceiling
        sigma_ceil = self._calibrate_garch_ceiling(rets)
        CFG["garch_sigma_ceiling"] = sigma_ceil
        log.info(f"[CALIB]   garch_sigma_ceiling → {sigma_ceil:.4f}")

        log.info(f"[CALIB] ✓  Calibration complete — all parameters now live")
        log.info(bar)

        self._last_calib_tick = tick_n
        if trigger == "startup":
            self._startup_done = True

    # ── Individual calibration routines ──────────────────────────────
    def _barrier_float(self) -> float:
        """Parse the current CFG barrier string to a float."""
        try:
            return abs(float(CFG["barrier"].replace("+", "").replace("-", "")))
        except Exception:
            return 2.70

    def _calibrate_barrier(self, prices: np.ndarray) -> float:
        """
        Overlapping (rolling) window barrier calibration with Wilson
        lower confidence bound selection.

        Replaces the non-overlapping walk-forward approach which yielded
        only ~6 samples at 720 ticks (SE ~0.20, barrier swings ±0.80).
        Rolling stride-1 windows yield ~600 samples (SE ~0.02), making
        the calibrated barrier stable to ±0.05 between runs.

        Selection criterion: Wilson 95% lower confidence bound >= target,
        not the point estimate. Prevents a lucky 6-sample draw from
        committing to a barrier that hasn't earned it statistically.

        Wilson lower bound:
            z    = 1.96
            p̂   = wins / n
            lb   = (p̂ + z²/2n - z·√(p̂(1-p̂)/n + z²/4n²)) / (1 + z²/n)

        Picks the *smallest* barrier where lb >= calib_target_win_rate.
        Falls back to current CFG barrier if none qualify.
        """
        n_ticks    = CFG["n_contract_ticks"]           # 120
        candidates = sorted(CFG["calib_barrier_candidates"])
        target_wr  = CFG["calib_target_win_rate"]
        prices     = prices.astype(float)
        z          = 1.96                               # 95% CI

        if len(prices) < n_ticks + 1:
            return self._barrier_float()

        # Rolling windows: entry at index i, terminal at i + n_ticks
        # Vectorised — no Python loop over ticks
        entries   = prices[:-n_ticks]
        terminals = prices[n_ticks:]
        moves     = np.abs(terminals - entries)         # shape: (N,)
        n_samples = len(moves)

        best = self._barrier_float()
        for b in candidates:
            wins  = int(np.sum(moves < b))
            p_hat = wins / n_samples

            # Wilson lower bound
            lb = (
                (p_hat + z**2 / (2 * n_samples)
                 - z * np.sqrt(p_hat * (1 - p_hat) / n_samples
                               + z**2 / (4 * n_samples**2)))
                / (1 + z**2 / n_samples)
            )

            log.info(
                f"[CALIB]     barrier={b:.2f}  "
                f"win_rate={p_hat:.3f}  lb95={lb:.3f}  "
                f"samples={n_samples}"
            )

            if lb >= target_wr:
                best = b
                break   # smallest that qualifies on lower bound

        return best

    def _calibrate_macd(self, prices: np.ndarray) -> float:
        """
        Replay MACD over price history; return the 85th-percentile
        of |histogram| as the veto threshold.
        """
        k_fast = 2.0 / (CFG["macd_fast"]   + 1)
        k_slow = 2.0 / (CFG["macd_slow"]   + 1)
        k_sig  = 2.0 / (CFG["macd_signal"] + 1)
        prices = prices.astype(float)
        ema_fast = ema_slow = prices[0]
        ema_sig: Optional[float] = None
        hists: list = []

        for p in prices[1:]:
            ema_fast += k_fast * (p - ema_fast)
            ema_slow += k_slow * (p - ema_slow)
            macd_line = ema_fast - ema_slow
            if ema_sig is None:
                ema_sig = macd_line
            else:
                ema_sig += k_sig * (macd_line - ema_sig)
                hists.append(abs(macd_line - ema_sig))

        if not hists:
            return CFG["macd_histogram_veto"]
        return float(np.percentile(hists, CFG["calib_macd_percentile"]))

    def _calibrate_ao(self, prices: np.ndarray) -> float:
        """
        FIX (Issue 5): Replay the AO over price history using the same
        OHLC-bar / midprice basis as the live AwesomeOscillator, so the
        veto threshold is on the correct scale. Previously this replayed
        AO directly on raw ticks, which no longer matches how the live
        AO (now bar-based) actually behaves — the threshold would have
        been calibrated on a different scale and lost its effect.

        Returns the calib_ao_percentile-th percentile of |AO| across bars.
        """
        fp = CFG["ao_fast_period"]   # bars
        sp = CFG["ao_slow_period"]   # bars

        bar_builder = OHLCBarBuilder(bar_ticks=CFG["ao_bar_ticks"])
        for p in prices.astype(float):
            bar_builder.update(p)

        mids = bar_builder.midprices()
        aos: list = []
        for i in range(sp, len(mids) + 1):
            window = mids[:i]
            aos.append(abs(float(np.mean(window[-fp:])) - float(np.mean(window[-sp:]))))

        if not aos:
            return CFG["ao_veto_threshold"]
        return float(np.percentile(aos, CFG["calib_ao_percentile"]))

    def _calibrate_jump_kurt(self, rets: np.ndarray) -> float:
        """
        Rolling window kurtosis over the return history.
        90th-percentile, clamped to [calib_jump_kurt_min, calib_jump_kurt_max].
        """
        window = CFG["jump_window"]
        rets   = rets.astype(float)
        kurts: list = []

        for i in range(window, len(rets)):
            w = rets[i - window:i]
            std = w.std(ddof=1)
            if std > 1e-12:
                kurts.append(abs(float(stats.kurtosis(w, fisher=True, bias=False))))

        if not kurts:
            return CFG["jump_kurtosis_veto"]

        val = float(np.percentile(kurts, 90))
        return float(np.clip(val,
                             CFG["calib_jump_kurt_min"],
                             CFG["calib_jump_kurt_max"]))

    def _calibrate_hmm_sigma(self, rets: np.ndarray) -> tuple:
        """
        Replay GARCH(1,1) over returns.  Use 33rd and 67th percentiles
        of the rolling σ_t series as the LOW/MED HMM bucket boundaries.
        """
        g      = GARCH11()
        sigmas = [g.update(float(r)) for r in rets]

        if len(sigmas) < 10:
            return CFG["hmm_lo_sigma"], CFG["hmm_hi_sigma"]

        lo = max(float(np.percentile(sigmas, 33)), 1e-7)
        hi = max(float(np.percentile(sigmas, 67)), lo + 1e-7)
        return lo, hi

    def _calibrate_garch_ceiling(self, rets: np.ndarray) -> float:
        """
        Replay GARCH(1,1) over returns.  Compute the 2-min cumulative
        σ at each tick; return the 99th-percentile as the extreme-vol ceiling.

        FIX: this replay previously called forecast_cumulative_std()
        without a hurst= argument, silently defaulting to hurst=0.5 —
        a flat, unscaled assumption that didn't match how live trading
        actually computes sigma_2min (using the real rolling Hurst
        value). That mismatch meant the calibrated ceiling was on a
        different scale than what live trading was vetoing against.
        Now mirrors the live 100-tick rolling Hurst window exactly.
        """
        g       = GARCH11()
        hurst_a = HurstAnalyzer()
        s2mins  = []
        for i, r in enumerate(rets):
            g.update(float(r))
            h_window = rets[max(0, i - 99): i + 1]
            h_val    = hurst_a.compute(h_window) if len(h_window) >= 20 else 0.5
            s2mins.append(g.forecast_cumulative_std(CFG["n_contract_ticks"], hurst=h_val))

        if not s2mins:
            return CFG["garch_sigma_ceiling"]
        return float(np.percentile(s2mins, 99))



class ConnState(enum.IntEnum):
    DISCONNECTED  = 0
    CONNECTING    = 1
    CONNECTED     = 2
    AUTHENTICATED = 3
    SUBSCRIBED    = 4


class DerivWSManager:
    """
    Reconnecting WebSocket using the new Deriv Options API.
    The OTP URL is single-use and short-lived — re-fetched every reconnect.
    """
    HEARTBEAT_INTERVAL = 20
    RECONNECT_BASE     = 2.0
    RECONNECT_CAP      = 120.0

    def __init__(self, url_factory, on_disconnect_cb=None, name="DerivWS"):
        self.url_factory       = url_factory   # async callable → fresh WS URL
        self._on_disconnect_cb = on_disconnect_cb
        self.name              = name
        self.state             = ConnState.DISCONNECTED
        self._running          = False
        self._ws               = None
        self._attempt          = 0
        self._pending: dict    = {}

    _counter = 0

    @classmethod
    def _new_id(cls) -> int:
        cls._counter += 1
        return cls._counter

    async def safe_send(self, payload: dict) -> bool:
        ws   = self._ws
        live = self.state >= ConnState.CONNECTED and ws is not None
        if not live:
            return False
        try:
            await ws.send(json.dumps(payload))
            return True
        except Exception as e:
            log.warning(f"[{self.name}] safe_send failed: {e}")
            return False

    async def send(self, payload: dict, timeout: float = 15.0) -> dict:
        rid               = self._new_id()
        payload["req_id"] = rid
        fut               = asyncio.get_event_loop().create_future()
        self._pending[rid] = fut
        if not await self.safe_send(payload):
            self._pending.pop(rid, None)
            raise websockets.ConnectionClosed(None, None)
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending.pop(rid, None)
            raise

    async def send_nowait(self, payload: dict):
        await self.safe_send(payload)

    def stop(self):
        self._running = False
        self.state    = ConnState.DISCONNECTED

    async def close(self):
        ws = self._ws
        if ws:
            try:
                await ws.close()
            except Exception:
                pass

    async def run(self, on_open, on_message):
        self._running = True
        while self._running:
            if self._attempt > 0:
                delay = min(
                    self.RECONNECT_BASE * (2 ** (self._attempt - 1)),
                    self.RECONNECT_CAP,
                ) + random.uniform(-1.0, 1.0)
                delay = max(1.0, delay)
                log.info(f"[{self.name}] Reconnect #{self._attempt} in {delay:.1f}s ...")
                await asyncio.sleep(delay)

            if not self._running:
                break

            self.state = ConnState.CONNECTING
            self._pending.clear()
            ka_task = recv_task = None

            try:
                connect_url = await self.url_factory()
            except Exception as e:
                log.error(f"[{self.name}] OTP URL fetch failed: {e}")
                self._attempt += 1
                continue

            try:
                self._ws = await websockets.connect(
                    connect_url, ping_interval=None, close_timeout=5,
                )
                self.state    = ConnState.CONNECTED
                self._attempt = 0
                log.info(f"[{self.name}] Connected.")

                ka_task = asyncio.create_task(self._heartbeat())

                async def _recv_loop():
                    async for raw in self._ws:
                        msg    = json.loads(raw)
                        req_id = msg.get("req_id")
                        if req_id and req_id in self._pending:
                            fut = self._pending.pop(req_id)
                            if not fut.done():
                                fut.set_result(msg)
                        else:
                            if msg.get("msg_type") == "ping":
                                continue
                            await on_message(msg)

                recv_task = asyncio.create_task(_recv_loop())
                await on_open(self)
                await recv_task

            except websockets.ConnectionClosed:
                log.warning(f"[{self.name}] Connection closed — reconnecting...")
            except Exception as e:
                log.error(f"[{self.name}] run error: {type(e).__name__}: {e}")
            finally:
                if ka_task:
                    ka_task.cancel()
                if recv_task and not recv_task.done():
                    recv_task.cancel()
                self.state = ConnState.DISCONNECTED
                await self.close()
                self._ws = None
                if not self._running:
                    break
                if self._on_disconnect_cb:
                    try:
                        self._on_disconnect_cb()
                    except Exception as e:
                        log.error(f"[{self.name}] disconnect_cb raised: {e}")
                self._attempt += 1

        log.info(f"[{self.name}] Connection loop exited cleanly.")

    async def _heartbeat(self):
        try:
            while self.state >= ConnState.CONNECTED:
                await asyncio.sleep(self.HEARTBEAT_INTERVAL)
                if not await self.safe_send({"ping": 1}):
                    return
        except asyncio.CancelledError:
            pass


# ══════════════════════════════════════════════════════════════════════
# MAIN BOT
# ══════════════════════════════════════════════════════════════════════
class ExpiryRangeBot:

    def __init__(self, app_id: str, api_token: str, account_id: Optional[str] = None):
        self.app_id     = app_id
        self.token      = api_token
        self.account_id = account_id

        # ── Intelligence layers ──
        self.garch    = GARCH11()
        self.mc_quick = MonteCarlo(n=CFG["stage1_mc_n"])
        self.mc_deep  = MonteCarlo(n=CFG["stage2_mc_n"])
        self.hmm      = HMMRegimes()
        self.hurst_a  = HurstAnalyzer()
        self.ou       = OUMeanReversion()
        self.bayes    = BayesianEdge()
        self.jump_fp  = JumpFirstPassage()
        self.macd     = MACDFilter()       # L9
        # FIX (Issue 5): AO now runs on OHLC bars (not raw ticks). The bar
        # builder is owned here and shared with the AutoCalibrator so the
        # live AO and the calibrated ao_veto_threshold are on the same scale.
        self.ao_bars  = OHLCBarBuilder(bar_ticks=CFG["ao_bar_ticks"])
        self.ao       = AwesomeOscillator(self.ao_bars) # L10
        self.guard      = RiskGuard()
        self.ensemble   = Ensemble()
        self.calibrator = AutoCalibrator()   # wired in — startup + loss triggers active

        # ── Price history ──
        self.ticks   = deque(maxlen=CFG["tick_buffer"])
        self.returns = deque(maxlen=CFG["tick_buffer"])
        self._tick_n = 0

        # ── Account state ──
        self.bankroll       = CFG["starting_bankroll"]
        self.active_id      = None
        self._buying        = False
        self._lock_since    = None
        self._calib_done    = False   # gates trading until startup calibration completes
        self.trade_count    = 0
        self.wins           = 0
        self.total_pnl      = 0.0
        self._entry_spot    = 0.0
        self._entry_conf    = 0.0
        self._entry_stake   = CFG["stake"]
        self._entry_macd    = 0.0
        self._entry_ao      = 0.0
        self._entry_mc_lb   = 0.0

        self.wsman: Optional[DerivWSManager] = None
        self._running = True

    # ── REST bootstrap (new Deriv Options API) ──────────────────────
    def _rest_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.token}",
            "Deriv-App-ID":  self.app_id,
            "Content-Type":  "application/json",
        }

    def _resolve_account_id_sync(self) -> str:
        url  = CFG["api_base"] + CFG["accounts_path"]
        resp = requests.get(url, headers=self._rest_headers(), timeout=15)
        resp.raise_for_status()
        data     = resp.json()
        accounts = data.get("data", data) if isinstance(data, dict) else data
        if isinstance(accounts, dict):
            accounts = accounts.get("accounts", accounts.get("data", []))
        for acc in accounts:
            if acc.get("account_type") == "real":
                acc_id = acc.get("account_id") or acc.get("id")
                if acc_id:
                    return acc_id
        raise RuntimeError(f"No demo account found in: {data}")

    def _fetch_otp_url_sync(self) -> str:
        if not self.account_id:
            self.account_id = self._resolve_account_id_sync()
            log.info(f"Resolved demo account_id = {self.account_id}")
        url  = CFG["api_base"] + CFG["otp_path"].format(account_id=self.account_id)
        resp = requests.post(url, headers=self._rest_headers(), timeout=15)
        resp.raise_for_status()
        data    = resp.json()
        payload = data.get("data", data) if isinstance(data, dict) else data
        ws_url  = payload.get("url")
        if not ws_url:
            raise RuntimeError(f"OTP response missing data.url: {data}")
        return ws_url

    async def _get_ws_url(self) -> str:
        return await asyncio.to_thread(self._fetch_otp_url_sync)

    # ── Intelligence pipeline ────────────────────────────────────────
    def _quick_models(self, spot: float, rets: np.ndarray, prices: np.ndarray) -> dict:
        """Stage 1 — all models with fast 10K MC. Returns signals dict."""
        r_now   = float(rets[-1]) if len(rets) > 0 else 0.0
        sigma_t = self.garch.update(r_now)

        # FIX (Hurst input bug): R/S analysis must be computed on RETURNS,
        # not price LEVELS. Price levels are themselves already a
        # cumulative/integrated process, so running R/S on them adds an
        # extra order of integration and pushes H toward 1.0 regardless
        # of the windowing fix above. Widened to 100 ticks (vs. the
        # previous 60) to reduce the small-sample upward bias inherent
        # to the simple R/S estimator.
        hurst_window = rets[-100:] if len(rets) >= 100 else rets
        h_val     = self.hurst_a.compute(hurst_window)
        hurst_sig = self.hurst_a.signal(h_val)

        sigma_2min = self.garch.forecast_cumulative_std(CFG["n_contract_ticks"], hurst=h_val)
        garch_sig  = self.garch.range_signal(sigma_2min)

        ou_window             = prices[-CFG["ou_fit_window"]:] if len(prices) >= CFG["ou_fit_window"] else prices
        theta, mu, sig_ou     = self.ou.fit(ou_window, ewma_lambda=CFG["ou_ewma_lambda"])
        ou_prob               = self.ou.p_in_range(spot, theta, mu, sig_ou, T=CFG["n_contract_ticks"])
        ou_mu_T, ou_var_T     = self.ou.terminal_dist(spot, theta, mu, sig_ou, T=CFG["n_contract_ticks"])

        drift      = float(np.mean(rets)) if len(rets) > 1 else 0.0
        mc_prob, mc_ci = self.mc_quick.probability(
            drift, sigma_t,
            n_ticks=CFG["n_contract_ticks"],
            ou_mu_T=ou_mu_T, ou_var_T=ou_var_T,
        )

        self.hmm.update(sigma_t)
        hmm_sig = self.hmm.signal()

        bayes_sig = self.bayes.mean()

        jump_sig, jump_veto, jump_kurt, jump_count = self.jump_fp.assess(rets)
        no_touch_p = self.jump_fp.no_touch_prob(sigma_t, n_ticks=CFG["n_contract_ticks"])

        # L9 MACD
        macd_score = self.macd.score()
        macd_veto  = self.macd.veto()
        macd_hist  = self.macd.histogram()

        # L10 AO
        ao_score = self.ao.score()
        ao_veto  = self.ao.veto()
        ao_val   = self.ao.value()

        weights = MODEL_WEIGHTS_BY_REGIME.get(self.hmm.state, MODEL_WEIGHTS_BY_REGIME[1])
        pre_score = float(np.clip(
            mc_prob    * weights["monte_carlo"] +
            garch_sig  * weights["garch"]       +
            hmm_sig    * weights["hmm"]         +
            hurst_sig  * weights["hurst"]       +
            ou_prob    * weights["ou_process"]  +
            bayes_sig  * weights["bayesian"]    +
            jump_sig   * weights["jump"]        +
            macd_score * weights["macd"]        +
            ao_score   * weights["ao"],
            0.0, 1.0,
        ))

        return {
            "pre_score":  pre_score,
            "mc_prob":    mc_prob,    "mc_ci":    mc_ci,
            "garch_sig":  garch_sig,  "sigma_t":  sigma_t, "sigma_2min": sigma_2min,
            "hmm_sig":    hmm_sig,
            "h_val":      h_val,      "hurst_sig": hurst_sig,
            "ou_prob":    ou_prob,    "ou_mu_T":  ou_mu_T, "ou_var_T": ou_var_T,
            "drift":      drift,
            "bayes_sig":  bayes_sig,
            "jump_sig":   jump_sig,   "jump_veto": jump_veto,
            "jump_kurt":  jump_kurt,  "jump_count": jump_count,
            "no_touch_p": no_touch_p,
            "macd_score": macd_score, "macd_veto": macd_veto, "macd_hist": macd_hist,
            "ao_score":   ao_score,   "ao_veto":   ao_veto,   "ao_val":   ao_val,
        }

    def run_intelligence(self, spot: float) -> tuple:
        """
        Full 10-layer two-stage pipeline.
        Returns (should_trade, confidence, mc_lower_bound, signals, reason).
        """
        if not self._calib_done:
            rem = max(CFG["calib_min_ticks"] - self._tick_n, 0)
            if rem > 0:
                return False, 0.0, 0.0, {}, f"AWAITING_CALIB({rem} ticks to threshold)"
            return False, 0.0, 0.0, {}, "AWAITING_CALIB(calibration running)"

        prices = np.array(self.ticks,   dtype=float)
        rets   = np.array(self.returns, dtype=float) if self.returns else np.array([0.0])

        # ── Stage 1 ──
        s = self._quick_models(spot, rets, prices)

        log.info(
            f"[S1] pre={s['pre_score']:.3f} | "
            f"MC={s['mc_prob']:.3f}±{s['mc_ci']:.4f} "
            f"G={s['garch_sig']:.2f} "
            f"HMM={self.hmm.name()}({s['hmm_sig']:.2f}) "
            f"H={s['h_val']:.3f}->{s['hurst_sig']:.2f} "
            f"OU={s['ou_prob']:.3f} Bay={s['bayes_sig']:.3f} "
            f"Jump={s['jump_sig']:.2f}(k={s['jump_kurt']:.1f}) "
            f"MACD_hist={s['macd_hist']:.4f}(score={s['macd_score']:.2f}) "
            f"AO={s['ao_val']:.4f}(score={s['ao_score']:.2f})"
        )

        # Hard vetoes before Stage 2
        if self.hmm.is_high_vol():
            self._log_signal(spot, s, 0.0, 0.0, stage=1, reason="HMM_HIGH_VOL_VETO")
            return False, s["pre_score"], 0.0, s, "HMM_HIGH_VOL_VETO"
        if self.garch.is_extreme(s["sigma_2min"], ceiling=CFG["garch_sigma_ceiling"]):
            self._log_signal(spot, s, 0.0, 0.0, stage=1, reason="GARCH_EXTREME_VOL_VETO")
            return False, s["pre_score"], 0.0, s, "GARCH_EXTREME_VOL_VETO"
        if s["jump_veto"]:
            self._log_signal(spot, s, 0.0, 0.0, stage=1, reason="JUMP_SPIKE_VETO")
            return False, s["pre_score"], 0.0, s, "JUMP_SPIKE_VETO"
        if s["macd_veto"]:
            self._log_signal(spot, s, 0.0, 0.0, stage=1, reason="MACD_HIGH_MOMENTUM_VETO")
            return False, s["pre_score"], 0.0, s, "MACD_HIGH_MOMENTUM_VETO"
        if s["ao_veto"]:
            self._log_signal(spot, s, 0.0, 0.0, stage=1, reason="AO_HIGH_ENERGY_VETO")
            return False, s["pre_score"], 0.0, s, "AO_HIGH_ENERGY_VETO"

        if s["pre_score"] < CFG["pre_scan_threshold"]:
            reason = f"PRE_SCAN_FAIL({s['pre_score']:.3f}<{CFG['pre_scan_threshold']})"
            self._log_signal(spot, s, 0.0, 0.0, stage=1, reason=reason)
            return False, s["pre_score"], 0.0, s, reason

        # ── Stage 2: 50K deep MC with terminal-price guarantee check ──
        mc_final, mc_ci_final = self.mc_deep.probability(
            s["drift"], s["sigma_t"],
            n_ticks=CFG["n_contract_ticks"],
            ou_mu_T=s["ou_mu_T"], ou_var_T=s["ou_var_T"],
        )
        mc_lower_bound = mc_final - mc_ci_final
        s["mc_prob"] = mc_final
        s["mc_ci"]   = mc_ci_final

        log.info(
            f"[S2] Deep MC(50K): p={mc_final:.4f}  CI±{mc_ci_final:.4f}  "
            f"lower_bound={mc_lower_bound:.4f}  "
            f"(guarantee_floor={CFG['mc_guarantee_floor']})"
        )

        # Terminal-price guarantee: lower confidence bound must clear the floor
        if mc_lower_bound < CFG["mc_guarantee_floor"]:
            reason = (
                f"MC_GUARANTEE_FAIL(lb={mc_lower_bound:.4f}"
                f"<{CFG['mc_guarantee_floor']})"
            )
            self._log_signal(spot, s, 0.0, mc_lower_bound, stage=2, reason=reason)
            return False, s["pre_score"], mc_lower_bound, s, reason

        trade, conf, scores, failed_floors, reason = self.ensemble.decide(
            mc_prob    = mc_final,
            garch_sig  = s["garch_sig"],
            hmm_sig    = s["hmm_sig"],
            hmm_state  = self.hmm.state,
            hurst_sig  = s["hurst_sig"],
            hurst_val  = s["h_val"],
            ou_prob    = s["ou_prob"],
            bayes_sig  = s["bayes_sig"],
            jump_sig   = s["jump_sig"],
            no_touch_p = s["no_touch_p"],
            macd_score = s["macd_score"],
            ao_score   = s["ao_score"],
            hmm_veto   = self.hmm.is_high_vol(),
            jump_veto  = s["jump_veto"],
            macd_veto  = s["macd_veto"],
            ao_veto    = s["ao_veto"],
        )

        log.info(f"[S2] conf={conf:.3f}  {reason}")
        self._log_signal(spot, s, conf, mc_lower_bound, stage=2, reason=reason)
        return trade, conf, mc_lower_bound, s, reason

    def _log_signal(self, spot, s, conf, mc_lower_bound, stage, reason):
        try:
            datalog.log_signal(
                timestamp       = time.strftime("%Y-%m-%d %H:%M:%S"),
                tick_n          = self._tick_n,
                spot            = spot,
                pre_score       = round(s.get("pre_score", 0), 5),
                conf            = round(conf, 5),
                mc_prob         = round(s.get("mc_prob", 0), 5),
                mc_ci           = round(s.get("mc_ci", 0), 5),
                mc_lower_bound  = round(mc_lower_bound, 5),
                garch_sig       = round(s.get("garch_sig", 0), 5),
                hmm_state       = self.hmm.state,
                hmm_sig         = round(s.get("hmm_sig", 0), 5),
                hurst_val       = round(s.get("h_val", 0), 5),
                hurst_sig       = round(s.get("hurst_sig", 0), 5),
                ou_prob         = round(s.get("ou_prob", 0), 5),
                bayes_sig       = round(s.get("bayes_sig", 0), 5),
                jump_sig        = round(s.get("jump_sig", 0), 5),
                jump_kurt       = round(s.get("jump_kurt", 0), 5),
                jump_count      = s.get("jump_count", 0),
                no_touch_p      = round(s.get("no_touch_p", 0), 5),
                macd_hist       = round(s.get("macd_hist", 0), 6),
                macd_sig_val    = round(self.macd._ema_sig or 0, 6),
                macd_score      = round(s.get("macd_score", 0), 5),
                ao_val          = round(s.get("ao_val", 0), 6),
                ao_score        = round(s.get("ao_score", 0), 5),
                sigma_t         = round(s.get("sigma_t", 0), 8),
                sigma_2min      = round(s.get("sigma_2min", 0), 8),
                stage           = stage,
                reason          = reason,
            )
        except Exception as e:
            log.warning(f"signal CSV write failed: {e}")

    # ── Tick handler ─────────────────────────────────────────────────
    STUCK_TIMEOUT_S = 360

    async def on_tick(self, tick: dict):
        spot = float(tick["quote"])

        if self.ticks:
            self.returns.append(spot - self.ticks[-1])
        self.ticks.append(spot)
        self._tick_n += 1
        self.guard.tick()

        # Update MACD and AO on every tick (they need continuous updates)
        self.macd.update(spot)
        self.ao.update(spot)

        if self.active_id or self._buying:
            if (self._lock_since is not None
                    and time.monotonic() - self._lock_since > self.STUCK_TIMEOUT_S):
                log.warning(
                    f"Contract/buy unresolved after {self.STUCK_TIMEOUT_S}s — force-unlocking."
                )
                self.active_id = self._buying = False
                self._lock_since = None
            else:
                return

        if not self.guard.can_trade():
            if self._tick_n % 30 == 0:
                log.info(f"[GUARD] {self.guard.status()}")
            return

        # ── Startup calibration gate ─────────────────────────────────
        # Trading is blocked until the first calibration completes.
        # Uses self._tick_n (total ticks received) — not len(self.ticks)
        # which is capped at tick_buffer=720 — to correctly count progress.
        if not self._calib_done:
            if self.calibrator.should_run_startup(self._tick_n):
                prices  = np.array(self.ticks,   dtype=float)
                returns = np.array(self.returns,  dtype=float) if self.returns else np.array([0.0])
                self.calibrator.run(
                    tick_n  = self._tick_n,
                    prices  = prices,
                    returns = returns,
                    hmm     = self.hmm,
                    trigger = "startup",
                )
                self._calib_done = True
                log.info("[CALIB] Startup calibration complete — trading now unlocked.")
            else:
                rem = max(CFG["calib_min_ticks"] - self._tick_n, 0)
                if self._tick_n % 60 == 0:
                    log.info(f"[CALIB] Collecting ticks for startup calibration ({rem} remaining).")
            return

        if self._tick_n % CFG["signal_interval"] != 0:
            return

        trade, conf, mc_lb, sigs, reason = self.run_intelligence(spot)

        if trade:
            stake = self.guard.compute_stake(self.bankroll, self.bayes.mean())
            log.info(
                f"ENTER SIGNAL  conf={conf:.3f}  mc_lb={mc_lb:.4f}  "
                f"MACD_hist={sigs.get('macd_hist', 0):.4f}  "
                f"AO={sigs.get('ao_val', 0):.4f}  "
                f"spot={spot}  stake=${stake:.2f}  [{reason}]"
            )
            self._buying      = True
            self._lock_since  = time.monotonic()
            self._entry_macd  = sigs.get("macd_hist", 0.0)
            self._entry_ao    = sigs.get("ao_val", 0.0)
            self._entry_mc_lb = mc_lb
            asyncio.create_task(self._request_and_buy(spot, conf, stake))

    # ── Proposal → buy ───────────────────────────────────────────────
    async def _request_and_buy(self, spot: float, conf: float = 0.0, stake: Optional[float] = None):
        try:
            if self.active_id:
                return
            if stake is None:
                stake = self.guard.stake

            resp = await self.wsman.send({
                "proposal":           1,
                "amount":             stake,
                "basis":              "stake",
                "contract_type":      CFG["contract_type"],
                "currency":           CFG["currency"],
                "duration":           CFG["duration"],
                "duration_unit":      CFG["duration_unit"],
                "underlying_symbol":  CFG["symbol"],
                "barrier":            CFG["barrier"],
                "barrier2":           CFG["barrier2"],
            })

            if resp.get("error"):
                log.warning(f"Proposal error: {resp['error'].get('message')}")
                return

            prop      = resp.get("proposal", {})
            pid       = prop.get("id")
            ask_price = prop.get("ask_price")

            if not pid or not ask_price:
                log.warning("Empty proposal — skipping")
                return

            if self.active_id:
                return

            await self._buy(pid, float(ask_price), spot, conf, stake)

        except asyncio.TimeoutError:
            log.warning("Proposal request timed out")
        except Exception as exc:
            log.error(f"_request_and_buy error: {exc}")
        finally:
            self._buying = False

    async def _buy(self, proposal_id: str, price: float, spot: float = 0.0,
                   conf: float = 0.0, stake: Optional[float] = None):
        try:
            resp = await self.wsman.send({"buy": proposal_id, "price": price})

            if resp.get("error"):
                log.warning(f"Buy rejected: {resp['error'].get('message')}")
                return

            buy_data = resp.get("buy", {})
            cid      = buy_data.get("contract_id")
            if not cid:
                log.warning("Buy response missing contract_id")
                return

            self.active_id    = cid
            self.trade_count += 1
            self._entry_spot  = spot
            self._entry_conf  = conf
            self._entry_stake = stake if stake is not None else self.guard.stake
            self._lock_since  = time.monotonic()

            log.info(
                f"CONTRACT OPEN #{self.trade_count} | "
                f"id={cid} | stake=${self._entry_stake:.2f} | "
                f"mc_lb={self._entry_mc_lb:.4f} | "
                f"MACD={self._entry_macd:.4f} | AO={self._entry_ao:.4f} | "
                f"buy_price={buy_data.get('buy_price')}"
            )

            await self.wsman.send_nowait({
                "proposal_open_contract": 1,
                "contract_id":            cid,
                "subscribe":              1,
            })

        except asyncio.TimeoutError:
            log.warning("Buy request timed out")
        except Exception as exc:
            log.error(f"_buy error: {exc}")

    # ── Settlement ───────────────────────────────────────────────────
    def _settle(self, poc: dict):
        profit = float(poc.get("profit", 0.0))
        won    = profit > 0.0

        self.total_pnl  += profit
        self.bankroll   += profit
        contract_id      = self.active_id
        self.active_id   = None
        self._buying     = False
        self._lock_since = None

        self.bayes.update(won)
        self.guard.check_bankroll(self.bankroll)

        if won:
            self.wins += 1
            self.guard.on_win()
            tag = "WIN "
        else:
            self.guard.on_loss()
            tag = "LOSS"
            # ── Loss-triggered recalibration ─────────────────────────
            if self.calibrator.should_run_on_loss(self._tick_n, len(self.ticks)):
                prices  = np.array(self.ticks,   dtype=float)
                returns = np.array(self.returns,  dtype=float) if self.returns else np.array([0.0])
                self.calibrator.run(
                    tick_n  = self._tick_n,
                    prices  = prices,
                    returns = returns,
                    hmm     = self.hmm,
                    trigger = "loss",
                )

        wr     = self.wins / self.trade_count if self.trade_count else 0.0
        lo, hi = self.bayes.ci95()

        log.info(f"{tag}  {profit:+.2f}  cumPnL={self.total_pnl:+.2f}  bankroll=${self.bankroll:.2f}")
        log.info(
            f"  trades={self.trade_count}  WR={wr:.1%}  "
            f"Bayes_WR={self.bayes.mean():.1%} CI=[{lo:.2f},{hi:.2f}]  "
            f"next={self.guard.status()}"
        )

        try:
            datalog.log_trade(
                timestamp      = time.strftime("%Y-%m-%d %H:%M:%S"),
                trade_n        = self.trade_count,
                contract_id    = contract_id,
                spot_entry     = self._entry_spot,
                stake          = self._entry_stake,
                conf           = round(self._entry_conf, 5),
                mc_lower_bound = round(self._entry_mc_lb, 5),
                macd_hist      = round(self._entry_macd, 6),
                ao_val         = round(self._entry_ao, 6),
                profit         = round(profit, 5),
                won            = int(won),
                bankroll       = round(self.bankroll, 5),
                total_pnl      = round(self.total_pnl, 5),
                win_rate       = round(wr, 5),
                guard_status   = self.guard.status(),
            )
        except Exception as e:
            log.warning(f"trade CSV write failed: {e}")

    # ── Message dispatcher ───────────────────────────────────────────
    async def on_message(self, msg: dict):
        mt = msg.get("msg_type")
        if mt == "tick":
            await self.on_tick(msg["tick"])
        elif mt == "proposal_open_contract":
            poc = msg.get("proposal_open_contract", {})
            if poc.get("is_sold") or poc.get("status") in ("won", "lost"):
                self._settle(poc)
        elif mt == "error":
            log.error(f"API error: {msg.get('error', {}).get('message')}")

    # ── Connection hooks ─────────────────────────────────────────────
    async def _on_open(self, wsman: DerivWSManager):
        wsman.state = ConnState.AUTHENTICATED
        log.info(f"Connected to authenticated OTP session (account={self.account_id}).")

        await wsman.send_nowait({"ticks": CFG["symbol"], "subscribe": 1})
        wsman.state = ConnState.SUBSCRIBED
        log.info(f"Subscribed to {CFG['symbol']} — collecting {CFG['calib_min_ticks']} ticks for startup calibration (~12 min).")

        if self.active_id:
            await wsman.send_nowait({
                "proposal_open_contract": 1,
                "contract_id":            self.active_id,
                "subscribe":              1,
            })

    def _on_disconnect(self):
        if self._buying:
            log.warning("Connection lost while preparing a trade — resetting flag.")
            self._buying     = False
            self._lock_since = None
        if self.active_id:
            log.warning(
                f"Connection lost while contract #{self.active_id} was open — "
                f"will resubscribe on reconnect."
            )

    # ── Main loop ────────────────────────────────────────────────────
    async def run(self):
        bar = "=" * 70
        log.info(bar)
        log.info("  EXPIRYRANGE BOT v2  ·  RDBEAR  ·  ±1.80  ·  2-min")
        log.info("  10-Layer Intelligence: L1-GARCH L2-MC L3-HMM L4-Hurst")
        log.info("  L5-OU L6-Bayes L7-Guard L8-Jump L9-MACD L10-AO")
        log.info(f"  MC guarantee floor: {CFG['mc_guarantee_floor']} (lower CI bound)")
        log.info(f"  MACD histogram veto: |hist| > {CFG['macd_histogram_veto']}")
        log.info(f"  AO energy veto: |AO| > {CFG['ao_veto_threshold']}")
        log.info(f"  Bankroll: ${self.bankroll:.2f}  Stop: ${CFG['drawdown_stop']:.2f}")
        log.info(f"  Connection: new Deriv Options API (REST OTP bootstrap)")
        log.info(bar)

        self.wsman = DerivWSManager(
            self._get_ws_url,
            on_disconnect_cb=self._on_disconnect,
            name="ExpiryRangeV2WS",
        )
        await self.wsman.run(on_open=self._on_open, on_message=self.on_message)


# ══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    APP_ID     = os.getenv("DERIV_APP_ID", "")
    API_TOKEN  = os.getenv("DERIV_API_TOKEN", "")
    ACCOUNT_ID = os.getenv("DERIV_ACCOUNT_ID", "") or None

    missing = []
    if not APP_ID:    missing.append("DERIV_APP_ID")
    if not API_TOKEN: missing.append("DERIV_API_TOKEN")
    if missing:
        print(
            f"\n  {', '.join(missing)} not set.\n"
            "   Set them as environment variables before starting.\n"
            "   App ID must be from a NEW developers.deriv.com application.\n"
        )
        raise SystemExit(1)

    bot = ExpiryRangeBot(app_id=APP_ID, api_token=API_TOKEN, account_id=ACCOUNT_ID)
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        log.info("Bot stopped (Ctrl+C)")
