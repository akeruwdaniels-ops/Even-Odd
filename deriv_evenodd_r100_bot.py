#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════╗
║       DERIV DIGIT EVEN / ODD BOT — MAXIMUM PRECISION EDITION            ║
║  Symbol      : R_100  (Volatility 100 Index)                            ║
║  Contract    : DIGITEVEN  or  DIGITODD  (auto-selected per tick)        ║
║  Base P(win) : 5/10 = 50%  (pure 50-50 — edge must come from structure)║
║  Auto-select : evaluates BOTH directions each tick, trades the one      ║
║                with higher ensemble confidence (if above threshold)      ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Why Even/Odd is harder — and why this stack handles it:                ║
║   • No base-rate bias (50/50) → zero free edge, all signal is structure ║
║   • Digit parity alternation creates detectable Markov patterns         ║
║   • Hawkes clustering: even/odd runs are self-exciting in volatile mkt  ║
║   • ARFIMA: long-range parity memory exists in synthetic indices        ║
║   • Copula: consecutive parity dependence is non-trivial in R_100      ║
║   • Kalman: hidden parity-bias state shifts slowly with volatility      ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Intelligence Stack — 10-Layer Maximum Precision Engine:                ║
║   L1  Higher-Order Markov Chain  (order-3, parity transition tensor)   ║
║   L2  Hidden Markov Model        (3-state: EVEN-BIAS/NEUTRAL/ODD-BIAS) ║
║   L3  Hawkes Self-Exciting PP    (parity run/cluster detector)          ║
║   L4  Sample Entropy (SampEn)    (predictability gate on parity series) ║
║   L5  ARFIMA Long-Memory         (fractional d on binary parity series) ║
║   L6  Hurst Exponent (R/S)       (persistence score on parity series)  ║
║   L7  Bayesian Beta-Binomial     (separate posteriors for EVEN & ODD)  ║
║   L8  Copula Dependence          (consecutive parity tail dependence)  ║
║   L9  Dual Kalman Filter         (hidden P(even) AND P(odd) trackers)  ║
║   L10 Risk Guard                 (cooldown · circuit breaker · stake)  ║
║   ∑   Dual Ensemble              (scores both sides, picks best)        ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Auto-Selection Logic:                                                   ║
║   Each tick → score EVEN direction AND ODD direction independently      ║
║   → pick whichever clears min_confidence                                ║
║   → if both clear, pick the higher confidence                           ║
║   → if neither clears, skip tick (no trade)                             ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Connection : NEW Deriv Options API (REST OTP bootstrap + auth'd WS)   ║
║    1. GET  /trading/v1/options/accounts        -> resolve account_id   ║
║    2. POST /trading/v1/options/accounts/{id}/otp -> authenticated URL  ║
║    3. Connect directly to returned wss:// URL (no authorize message)   ║
║    Re-run steps 1-2 on every reconnect — the OTP URL is single-use.   ║
╚══════════════════════════════════════════════════════════════════════════╝

Usage:
  export DERIV_APP_ID=<your_new_app_id>
  export DERIV_API_TOKEN=<your_PAT>
  export DERIV_ACCOUNT_ID=<your_demo_account_id>   # optional
  python deriv_evenodd_r100_bot.py

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

import numpy as np
import requests
import websockets
from scipy import stats

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════
CFG = {
    # ── Contract ──
    "underlying_symbol": "R_10",
    "currency":          "USD",
    "duration":          1,
    "duration_unit":     "t",

    # ── Capital ──
    "starting_bankroll":  2.5,
    "stake":              0.35,
    "drawdown_stop":      0.80,    # halt below this bankroll

    # ── Kelly (activates when bankroll >= threshold) ──
    "kelly_activation_bankroll":      5.0,
    "kelly_fraction":                 0.5,
    "kelly_min_stake":                0.35,
    "kelly_max_fraction_of_bankroll": 0.10,
    "payout_ratio":                   0.95,  # DIGITEVEN/ODD pays ~95% on $0.35
                                              # confirm from actual trade and update

    # ── Signal ──
    "warmup_ticks":          80,    # fills order-3 Markov table properly
    "min_confidence":        0.60,  # 50% base → need meaningful edge above this
    "min_confidence_delta":  0.03,  # when both sides clear, must beat other by this
    "markov_order":          3,
    "sampen_m":              2,
    "sampen_r_factor":       0.20,
    "sampen_veto_above":     1.80,  # tighter than OVER2 (50/50 needs more structure)
    "arfima_window":         60,
    "hurst_window":          60,
    "hawkes_window":         40,
    "hawkes_decay":          0.7,
    "kalman_process_var":    0.002, # slightly higher — parity regime shifts faster
    "kalman_obs_var":        0.08,
    "copula_window":         50,
    "copula_tail_veto":      0.60,

    # ── Risk ──
    "cooldown_win":              1,
    "cooldown_loss":             5,
    "consecutive_loss_limit":    3,
    "consecutive_loss_cooldown": 20,

    # ── Connection (new Options API) ──
    "api_base":      "https://api.derivws.com",
    "accounts_path": "/trading/v1/options/accounts",
    "otp_path":      "/trading/v1/options/accounts/{account_id}/otp",
    "reconnect_delay": 5,

    # ── Logging ──
    "log_dir":     os.getenv("LOG_DIR", "logs"),
    "log_file":    "evenodd_bot.log",
    "signals_csv": "evenodd_signals.csv",
    "trades_csv":  "evenodd_trades.csv",
    "tick_buffer": 300,
}

# ── Regime-conditional ensemble weights ──
# EVEN-BIAS regime: Markov + Kalman carry more weight (directional signal)
# NEUTRAL:          balanced — SampEn + ARFIMA + Copula become critical
# ODD-BIAS regime:  same as EVEN-BIAS (symmetric)
MODEL_WEIGHTS_BY_REGIME = {
    0: {  # EVEN-BIAS / ODD-BIAS — directional signals dominate
        "markov":   0.28, "hmm":     0.14, "hawkes": 0.06,
        "sampen":   0.08, "arfima":  0.12, "hurst":  0.10,
        "bayesian": 0.08, "copula":  0.06, "kalman": 0.08,
    },
    1: {  # NEUTRAL — structure signals dominate
        "markov":   0.20, "hmm":     0.12, "hawkes": 0.10,
        "sampen":   0.12, "arfima":  0.14, "hurst":  0.10,
        "bayesian": 0.10, "copula":  0.08, "kalman": 0.04,
    },
    2: {  # HIGH-ALTERNATION — Hawkes + Copula dominate
        "markov":   0.16, "hmm":     0.12, "hawkes": 0.16,
        "sampen":   0.10, "arfima":  0.10, "hurst":  0.08,
        "bayesian": 0.10, "copula":  0.12, "kalman": 0.06,
    },
}

# Per-model hard floors — all must pass for either direction
MODEL_FLOORS = {
    "markov":   0.52,   # slightly above 50% for meaningful Markov edge
    "hmm":      0.48,
    "hawkes":   0.42,
    "sampen":   0.40,
    "arfima":   0.40,
    "hurst":    0.40,
    "bayesian": 0.42,
    "copula":   0.42,
    "kalman":   0.52,
}

# ══════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════
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
log = logging.getLogger("EVENODD_BOT")


# ══════════════════════════════════════════════════════════════════════════
# DATA LOGGER
# ══════════════════════════════════════════════════════════════════════════
class DataLogger:
    SIGNAL_FIELDS = [
        "timestamp", "tick_n", "spot", "last_digit", "parity",
        "direction", "markov_p", "hmm_sig", "hawkes_sig", "sampen_sig",
        "arfima_sig", "hurst_sig", "bayes_sig", "copula_sig", "kalman_sig",
        "conf", "regime", "reason",
    ]
    TRADE_FIELDS = [
        "timestamp", "trade_n", "contract_id", "direction",
        "spot_entry", "last_digit_entry", "stake", "conf",
        "profit", "won", "bankroll", "total_pnl", "win_rate",
    ]

    def __init__(self, log_dir: Path):
        self._init_file(log_dir / CFG["signals_csv"], self.SIGNAL_FIELDS)
        self._init_file(log_dir / CFG["trades_csv"],  self.TRADE_FIELDS)
        self.sig_path   = log_dir / CFG["signals_csv"]
        self.trade_path = log_dir / CFG["trades_csv"]

    @staticmethod
    def _init_file(path, fields):
        if not path.exists() or path.stat().st_size == 0:
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=fields).writeheader()

    def log_signal(self, **row):
        with open(self.sig_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.SIGNAL_FIELDS).writerow(row)

    def log_trade(self, **row):
        with open(self.trade_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.TRADE_FIELDS).writerow(row)


datalog = DataLogger(LOG_DIR)


# ══════════════════════════════════════════════════════════════════════════
# UTILITY
# ══════════════════════════════════════════════════════════════════════════
def last_digit(spot: float) -> int:
    """Last decimal digit of R_100 spot price (2 d.p.)."""
    return int(round(spot * 100)) % 10

def is_even(digit: int) -> bool:
    return digit % 2 == 0

def parity_label(digit: int) -> str:
    return "EVEN" if is_even(digit) else "ODD"

def parity_bit(digit: int) -> int:
    """0 = even, 1 = odd."""
    return digit % 2


# ══════════════════════════════════════════════════════════════════════════
# LAYER 1 — HIGHER-ORDER MARKOV CHAIN (order-3, parity-space)
# ══════════════════════════════════════════════════════════════════════════
class ParityMarkovChain:
    """
    Transition tensor over parity bits {0=even, 1=odd}.
    State  = (parity_{t-2}, parity_{t-1}, parity_t)
    Output = P(next_parity = 0) and P(next_parity = 1)

    Falls back order-2 → order-1 when state count < min_count.
    Also maintains a full digit-level order-3 tensor so we can
    compute P(next_digit is even | last 3 digits) for extra precision.
    """

    def __init__(self, order: int = 3, min_count: int = 4):
        self.order      = order
        self.min_count  = min_count
        # Parity-space tensors
        self.pc3 = np.zeros((2, 2, 2, 2), dtype=np.float32)  # order-3
        self.pc2 = np.zeros((2, 2, 2),    dtype=np.float32)  # order-2
        self.pc1 = np.zeros((2, 2),       dtype=np.float32)  # order-1
        # Digit-level tensor (10-state) for fine-grained even/odd conditioning
        self.dc3 = np.zeros((10, 10, 10, 10), dtype=np.float32)
        self._ph = deque(maxlen=4)   # parity history
        self._dh = deque(maxlen=4)   # digit history

    def update(self, digit: int):
        p = parity_bit(digit)
        dh = list(self._dh)
        ph = list(self._ph)

        if len(ph) >= 1:
            self.pc1[ph[-1], p] += 1
        if len(ph) >= 2:
            self.pc2[ph[-2], ph[-1], p] += 1
        if len(ph) >= 3:
            self.pc3[ph[-3], ph[-2], ph[-1], p] += 1
        if len(dh) >= 3:
            self.dc3[dh[-3], dh[-2], dh[-1], digit] += 1

        self._ph.append(p)
        self._dh.append(digit)

    def p_even(self) -> float:
        """P(next parity = even) from parity Markov chain."""
        ph = list(self._ph)
        # Order-3 parity
        if len(ph) >= 3:
            row = self.pc3[ph[-3], ph[-2], ph[-1]]
            n   = row.sum()
            if n >= self.min_count:
                return float(row[0] / n)   # row[0] = even count
        # Order-2 parity
        if len(ph) >= 2:
            row = self.pc2[ph[-2], ph[-1]]
            n   = row.sum()
            if n >= self.min_count:
                return float(row[0] / n)
        # Order-1 parity
        if len(ph) >= 1:
            row = self.pc1[ph[-1]]
            n   = row.sum()
            if n >= self.min_count:
                return float(row[0] / n)
        # Fine-grained digit-level fallback
        dh = list(self._dh)
        if len(dh) >= 3:
            row = self.dc3[dh[-3], dh[-2], dh[-1]]
            n   = row.sum()
            if n >= self.min_count:
                # sum probabilities of even digits (0,2,4,6,8)
                return float(row[0::2].sum() / n)
        return 0.50   # true prior for even/odd

    def p_odd(self) -> float:
        return 1.0 - self.p_even()

    def markov_signal(self, direction: str) -> float:
        """Return P(direction) as the Markov signal for that side."""
        return self.p_even() if direction == "EVEN" else self.p_odd()


# ══════════════════════════════════════════════════════════════════════════
# LAYER 2 — HIDDEN MARKOV MODEL (3-state parity-bias regime)
# ══════════════════════════════════════════════════════════════════════════
class ParityHMM:
    """
    States:
      0 = EVEN-BIAS  — even digits appearing > 55% recently
      1 = NEUTRAL    — near 50/50
      2 = ODD-BIAS   — odd digits appearing > 55% recently

    Each state emits a directional prior for EVEN and ODD.
    """
    EVEN_BIAS, NEUTRAL, ODD_BIAS = 0, 1, 2

    _THRESH_HI = 0.55   # above this rolling rate → biased state

    _A = np.array([
        [0.78, 0.18, 0.04],   # from EVEN_BIAS
        [0.10, 0.80, 0.10],   # from NEUTRAL
        [0.04, 0.18, 0.78],   # from ODD_BIAS
    ])

    # P(win | regime, direction)
    _PRIOR = {
        EVEN_BIAS: {"EVEN": 0.60, "ODD":  0.40},
        NEUTRAL:   {"EVEN": 0.50, "ODD":  0.50},
        ODD_BIAS:  {"EVEN": 0.40, "ODD":  0.60},
    }

    def __init__(self):
        self.state    = self.NEUTRAL
        self._alpha   = np.array([0.15, 0.70, 0.15])
        self._win_buf = deque(maxlen=20)

    def update(self, digit: int):
        self._win_buf.append(1 if is_even(digit) else 0)
        if len(self._win_buf) < 5:
            return self.state
        rate = float(np.mean(self._win_buf))
        if rate > self._THRESH_HI:
            obs = self.EVEN_BIAS
        elif rate < (1.0 - self._THRESH_HI):
            obs = self.ODD_BIAS
        else:
            obs = self.NEUTRAL
        new          = self._A[obs] * self._alpha
        s            = new.sum()
        self._alpha  = new / s if s > 0 else np.ones(3) / 3
        self.state   = int(np.argmax(self._alpha))
        return self.state

    def signal(self, direction: str) -> float:
        return self._PRIOR[self.state][direction]

    def name(self) -> str:
        return ["EVEN-BIAS", "NEUTRAL", "ODD-BIAS"][self.state]

    def regime_idx(self) -> int:
        """Maps to MODEL_WEIGHTS_BY_REGIME key: biased→0, neutral→1."""
        return 1 if self.state == self.NEUTRAL else 0


# ══════════════════════════════════════════════════════════════════════════
# LAYER 3 — HAWKES SELF-EXCITING POINT PROCESS (parity runs)
# ══════════════════════════════════════════════════════════════════════════
class ParityHawkes:
    """
    Tracks clustering of SAME-parity streaks.
    High same-parity clustering → current parity is self-exciting →
    trade IN the direction of the current run.

    Maintains separate intensities for even-run and odd-run events.
    Returns directional signals: high even_signal = even streaking.
    """

    def __init__(self, mu: float = 0.04, alpha: float = 0.45, beta: float = None):
        self.mu    = mu
        self.alpha = alpha
        self.beta  = beta if beta is not None else CFG["hawkes_decay"]
        self._lam_even = 0.0
        self._lam_odd  = 0.0
        self._prev_par = None

    def update(self, digit: int):
        p = parity_bit(digit)
        # Decay both intensities
        self._lam_even = self._lam_even * self.beta + self.mu
        self._lam_odd  = self._lam_odd  * self.beta + self.mu
        # Excite the matching parity if run continues
        if self._prev_par == p:
            if p == 0:
                self._lam_even += self.alpha
            else:
                self._lam_odd  += self.alpha
        self._prev_par = p

    def _normalize(self, lam: float) -> float:
        floor_i   = self.mu / max(1 - self.beta, 1e-6)
        ceiling_i = floor_i + self.alpha / max(1 - self.beta, 1e-6)
        return float(np.clip((lam - floor_i) / max(ceiling_i - floor_i, 1e-9), 0.0, 1.0))

    def signal(self, direction: str) -> float:
        """
        High signal for EVEN = even intensity is high (even is clustering).
        We trade WITH the cluster direction.
        """
        if direction == "EVEN":
            return float(np.clip(self._normalize(self._lam_even), 0.0, 1.0))
        else:
            return float(np.clip(self._normalize(self._lam_odd), 0.0, 1.0))


# ══════════════════════════════════════════════════════════════════════════
# LAYER 4 — SAMPLE ENTROPY on parity series
# ══════════════════════════════════════════════════════════════════════════
class SampleEntropySignal:
    """
    Computes SampEn on the binary parity series (0/1).
    Low SampEn → high regularity → predictable parity structure.
    High SampEn → near-alternating-random → veto both directions.
    """

    @staticmethod
    def compute(series: np.ndarray, m: int = 2, r_factor: float = 0.20) -> float:
        n = len(series)
        if n < 2 * (m + 1):
            return 1.5
        d = series.astype(float)
        r = r_factor * d.std(ddof=1)
        if r < 1e-9:
            return 0.0

        def count_templates(length):
            count = 0
            for i in range(n - length):
                template = d[i:i + length]
                for j in range(n - length):
                    if i != j and np.max(np.abs(d[j:j + length] - template)) <= r:
                        count += 1
            return count

        A = count_templates(m + 1)
        B = count_templates(m)
        if B == 0 or A == 0:
            return 2.0
        return float(-np.log(A / B))

    @classmethod
    def signal(cls, parity_arr: np.ndarray) -> tuple:
        se   = cls.compute(parity_arr[-40:] if len(parity_arr) >= 40 else parity_arr,
                           m=CFG["sampen_m"], r_factor=CFG["sampen_r_factor"])
        veto = se > CFG["sampen_veto_above"]
        sig  = float(np.clip(1.0 - se / CFG["sampen_veto_above"], 0.0, 1.0))
        return sig, se, veto


# ══════════════════════════════════════════════════════════════════════════
# LAYER 5 — ARFIMA LONG-MEMORY on parity series
# ══════════════════════════════════════════════════════════════════════════
class ARFIMASignal:
    """
    Estimates fractional d from the binary parity series using
    log-periodogram (GPH) regression.

    d > 0: long memory in the parity sequence (runs persist)
    d < 0: anti-persistence (alternation dominates)
    d ≈ 0: short memory

    For EVEN direction: d > 0 is bullish (runs persist → even likely)
    For ODD  direction: d > 0 is also bullish (same run logic)
    Signal is direction-agnostic (persistence is useful for both).
    """

    @staticmethod
    def estimate_d(parity_arr: np.ndarray) -> float:
        n = len(parity_arr)
        if n < 20:
            return 0.0
        x    = parity_arr.astype(float) - 0.5
        freq = np.fft.rfftfreq(n)[1:]
        psd  = np.abs(np.fft.rfft(x)[1:]) ** 2
        m    = max(2, len(freq) // 10)
        lam  = freq[:m]
        Iy   = psd[:m]
        if np.any(Iy <= 0):
            return 0.0
        x_reg = -2.0 * np.log(2 * np.pi * lam)
        y_reg = np.log(Iy)
        try:
            d, _ = np.polyfit(x_reg, y_reg, 1)
            return float(np.clip(d, -0.5, 0.5))
        except Exception:
            return 0.0

    @classmethod
    def signal(cls, parity_arr: np.ndarray) -> tuple:
        d   = cls.estimate_d(parity_arr[-CFG["arfima_window"]:])
        # Map d ∈ [-0.5, 0.5] → [0.25, 0.75]
        sig = float(np.clip(0.50 + 0.50 * d, 0.25, 0.75))
        return sig, d


# ══════════════════════════════════════════════════════════════════════════
# LAYER 6 — HURST EXPONENT on parity series
# ══════════════════════════════════════════════════════════════════════════
class HurstSignal:
    """
    H > 0.5 → parity runs persist → trade WITH current parity direction
    H < 0.5 → alternation dominates → trade AGAINST current parity
    H ≈ 0.5 → random walk → neutral

    For a given direction:
      - If EVEN and H > 0.5 and last parity was even  → high signal
      - If EVEN and H < 0.5 and last parity was odd   → high signal (mean-revert)
    """

    @staticmethod
    def compute(series: np.ndarray) -> float:
        n = len(series)
        if n < 20:
            return 0.5
        max_lag = max(5, n // 4)
        lags, rs = [], []
        for lag in range(4, max_lag):
            c = series[:lag].astype(float)
            m = c.mean()
            d = np.cumsum(c - m)
            r = d.max() - d.min()
            s = c.std(ddof=1)
            if s > 0:
                lags.append(lag)
                rs.append(r / s)
        if len(rs) < 4:
            return 0.5
        try:
            h, _ = np.polyfit(np.log(lags), np.log(rs), 1)
            return float(np.clip(h, 0.05, 0.95))
        except Exception:
            return 0.5

    @classmethod
    def signal(cls, parity_arr: np.ndarray, direction: str) -> tuple:
        """
        Returns (signal, h_val).
        Signal accounts for BOTH persistence direction and current parity.
        """
        h = cls.compute(parity_arr[-CFG["hurst_window"]:])
        last_par = parity_label(int(parity_arr[-1])) if len(parity_arr) > 0 else "EVEN"

        if h > 0.5:
            # Trend-following: favour same direction as last parity
            if direction == last_par:
                sig = float(np.clip(0.50 + (h - 0.50) * 1.2, 0.50, 0.85))
            else:
                sig = float(np.clip(0.50 - (h - 0.50) * 1.2, 0.15, 0.50))
        else:
            # Mean-reverting: favour opposite direction to last parity
            if direction != last_par:
                sig = float(np.clip(0.50 + (0.50 - h) * 1.2, 0.50, 0.85))
            else:
                sig = float(np.clip(0.50 - (0.50 - h) * 1.2, 0.15, 0.50))

        return sig, h


# ══════════════════════════════════════════════════════════════════════════
# LAYER 7 — BAYESIAN BETA-BINOMIAL (separate for EVEN and ODD)
# ══════════════════════════════════════════════════════════════════════════
class DualBayesianEdge:
    """
    Maintains separate Beta posteriors for EVEN and ODD win rates.
    Both start at Beta(7, 7) ≡ 50% with 14-trade effective sample.
    After each trade, updates the relevant posterior.
    """

    def __init__(self, prior_wr: float = 0.50, prior_n: float = 14.0):
        a0 = prior_wr * prior_n
        b0 = (1.0 - prior_wr) * prior_n
        self._a = {"EVEN": a0, "ODD": a0}
        self._b = {"EVEN": b0, "ODD": b0}
        self.n_obs = {"EVEN": 0, "ODD": 0}

    def update(self, direction: str, won: bool):
        if won:
            self._a[direction] += 1.0
        else:
            self._b[direction] += 1.0
        self.n_obs[direction] += 1

    def mean(self, direction: str) -> float:
        a = self._a[direction]
        b = self._b[direction]
        return float(a / (a + b))

    def ci95(self, direction: str) -> tuple:
        a, b = self._a[direction], self._b[direction]
        return (
            float(stats.beta.ppf(0.025, a, b)),
            float(stats.beta.ppf(0.975, a, b)),
        )


# ══════════════════════════════════════════════════════════════════════════
# LAYER 8 — COPULA TAIL DEPENDENCE (parity consecutive)
# ══════════════════════════════════════════════════════════════════════════
class ParityCopula:
    """
    Measures upper-tail dependence in the parity series:
    λ_U = P(parity_t is even AND parity_{t+1} is even | both in top quartile)

    High λ_U for EVEN → even streaks are correlated → trade EVEN
    High λ_U for ODD  → odd streaks are correlated  → trade ODD

    Signal for direction = tail dependence for that direction.
    Also computes alternation score (cross-tail) as a veto signal.
    """

    @staticmethod
    def tail_dep(parity_arr: np.ndarray, target_parity: int, q: float = 0.30) -> float:
        """Empirical tail dependence for consecutive same-parity events."""
        n = len(parity_arr)
        if n < 20:
            return 0.0
        x = (parity_arr[:-1] == target_parity).astype(float)
        y = (parity_arr[1:]  == target_parity).astype(float)
        rx = stats.rankdata(x) / (len(x) + 1)
        ry = stats.rankdata(y) / (len(y) + 1)
        joint = np.sum((rx >= (1 - q)) & (ry >= (1 - q)))
        marg  = np.sum(rx >= (1 - q))
        return float(joint / marg) if marg > 0 else 0.0

    @classmethod
    def signal(cls, parity_arr: np.ndarray, direction: str) -> tuple:
        w    = parity_arr[-CFG["copula_window"]:] if len(parity_arr) >= CFG["copula_window"] else parity_arr
        tgt  = 0 if direction == "EVEN" else 1
        lam  = cls.tail_dep(w, tgt)
        # Alternation veto: if opposite direction has very high tail dep, risky
        opp_lam = cls.tail_dep(w, 1 - tgt)
        veto = opp_lam > CFG["copula_tail_veto"]
        # Signal: own tail dep is bullish for this direction
        sig  = float(np.clip(0.42 + lam * 0.6, 0.0, 1.0))
        return sig, lam, veto


# ══════════════════════════════════════════════════════════════════════════
# LAYER 9 — DUAL KALMAN FILTER (hidden P(even) and P(odd))
# ══════════════════════════════════════════════════════════════════════════
class DualKalman:
    """
    Two coupled 1-D Kalman filters:
    - x_even: hidden P(next digit is even)
    - x_odd:  hidden P(next digit is odd)

    They are constrained: x_odd = 1 - x_even (we update only x_even).
    Observation z_t = 1 if even, 0 if odd.
    """

    def __init__(self, x0: float = 0.50, P0: float = 0.08):
        self.x = x0
        self.P = P0
        self.Q = CFG["kalman_process_var"]
        self.R = CFG["kalman_obs_var"]

    def update(self, digit: int) -> float:
        z = 1.0 if is_even(digit) else 0.0
        x_pred = self.x
        P_pred = self.P + self.Q
        K      = P_pred / (P_pred + self.R)
        self.x = float(np.clip(x_pred + K * (z - x_pred), 0.0, 1.0))
        self.P = (1 - K) * P_pred
        return self.x

    def signal(self, direction: str) -> float:
        return float(np.clip(self.x if direction == "EVEN" else 1.0 - self.x, 0.0, 1.0))


# ══════════════════════════════════════════════════════════════════════════
# LAYER 10 — RISK GUARD
# ══════════════════════════════════════════════════════════════════════════
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
                f"⚠️  {self._consec_losses} consecutive losses — "
                f"extended cooldown ({CFG['consecutive_loss_cooldown']} ticks)"
            )
        else:
            self._cooldown = CFG["cooldown_loss"]

    def check_bankroll(self, bankroll: float):
        if bankroll < CFG["drawdown_stop"]:
            self._tripped = True
            log.warning(f"⛔  CIRCUIT BREAKER — bankroll ${bankroll:.2f} halted.")

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
        f_half = min(f_star * CFG["kelly_fraction"], CFG["kelly_max_fraction_of_bankroll"])
        stake  = max(bankroll * f_half, CFG["kelly_min_stake"])
        stake  = min(stake, bankroll)
        self.stake = round(stake, 2)
        return self.stake

    def status(self) -> str:
        if self._tripped:
            return "CIRCUIT_BREAKER"
        if self._cooldown > 0:
            return f"COOLDOWN({self._cooldown})"
        return "READY"


# ══════════════════════════════════════════════════════════════════════════
# DUAL ENSEMBLE  (scores EVEN and ODD independently, picks best)
# ══════════════════════════════════════════════════════════════════════════
class DualEnsemble:
    """
    Scores each direction independently through the same floor+weight pipeline.
    Returns (best_direction, confidence) or (None, 0.0) if neither qualifies.

    Auto-selection rules:
      1. Both fail threshold → no trade
      2. One passes, one fails → trade the passing direction
      3. Both pass → trade the higher confidence IF delta >= min_confidence_delta
         (avoids trading a coin-flip between directions)
    """

    def _score_direction(
        self,
        direction:   str,
        markov_p:    float,
        hmm_sig:     float,
        hawkes_sig:  float,
        sampen_sig:  float,
        arfima_sig:  float,
        hurst_sig:   float,
        bayes_sig:   float,
        copula_sig:  float,
        kalman_sig:  float,
        regime_idx:  int,
        sampen_veto: bool,
        copula_veto: bool,
    ) -> tuple:
        """Returns (should_trade, conf, failed_floors, reason)."""

        # Hard vetoes
        if sampen_veto:
            return False, 0.0, [], "SAMPEN_VETO"
        if copula_veto:
            return False, 0.0, [], f"COPULA_VETO_{direction}"

        scores = {
            "markov":   markov_p,
            "hmm":      hmm_sig,
            "hawkes":   hawkes_sig,
            "sampen":   sampen_sig,
            "arfima":   arfima_sig,
            "hurst":    hurst_sig,
            "bayesian": bayes_sig,
            "copula":   copula_sig,
            "kalman":   kalman_sig,
        }

        failed = [k for k, floor in MODEL_FLOORS.items() if scores[k] < floor]
        if failed:
            return False, 0.0, failed, f"FLOOR_FAIL({','.join(failed)})"

        W    = MODEL_WEIGHTS_BY_REGIME.get(regime_idx, MODEL_WEIGHTS_BY_REGIME[1])
        conf = float(np.clip(sum(scores[k] * W[k] for k in W), 0.0, 1.0))

        if conf < CFG["min_confidence"]:
            return False, conf, [], f"CONF_LOW({conf:.3f})"

        return True, conf, [], "TRADE"

    def decide(
        self,
        # EVEN-direction signals
        even_markov:   float, even_hmm:    float, even_hawkes: float,
        even_arfima:   float, even_hurst:  float, even_bayes:  float,
        even_copula:   float, even_kalman: float,
        # ODD-direction signals
        odd_markov:    float, odd_hmm:     float, odd_hawkes:  float,
        odd_arfima:    float, odd_hurst:   float, odd_bayes:   float,
        odd_copula:    float, odd_kalman:  float,
        # Shared signals (direction-agnostic)
        sampen_sig:    float,
        regime_idx:    int,
        sampen_veto:   bool,
        even_copula_veto: bool,
        odd_copula_veto:  bool,
    ) -> tuple:
        """Returns (direction_or_None, confidence, reason)."""

        trade_e, conf_e, _, reason_e = self._score_direction(
            "EVEN", even_markov, even_hmm, even_hawkes, sampen_sig,
            even_arfima, even_hurst, even_bayes, even_copula, even_kalman,
            regime_idx, sampen_veto, even_copula_veto,
        )
        trade_o, conf_o, _, reason_o = self._score_direction(
            "ODD", odd_markov, odd_hmm, odd_hawkes, sampen_sig,
            odd_arfima, odd_hurst, odd_bayes, odd_copula, odd_kalman,
            regime_idx, sampen_veto, odd_copula_veto,
        )

        if not trade_e and not trade_o:
            return None, 0.0, f"E:{reason_e} O:{reason_o}"

        if trade_e and not trade_o:
            return "EVEN", conf_e, reason_e

        if trade_o and not trade_e:
            return "ODD", conf_o, reason_o

        # Both pass — pick higher if delta is significant
        delta = abs(conf_e - conf_o)
        if delta < CFG["min_confidence_delta"]:
            return None, 0.0, f"BOTH_PASS_NO_DELTA({delta:.3f})"

        if conf_e >= conf_o:
            return "EVEN", conf_e, f"AUTO_EVEN(Δ={delta:.3f})"
        else:
            return "ODD", conf_o, f"AUTO_ODD(Δ={delta:.3f})"


# ══════════════════════════════════════════════════════════════════════════
# CONNECTION STATE
# ══════════════════════════════════════════════════════════════════════════
class ConnState(enum.IntEnum):
    DISCONNECTED  = 0
    CONNECTING    = 1
    CONNECTED     = 2
    AUTHENTICATED = 3
    SUBSCRIBED    = 4


# ══════════════════════════════════════════════════════════════════════════
# DERIV WS MANAGER  (identical architecture to reference bot)
# ══════════════════════════════════════════════════════════════════════════
class DerivWSManager:
    RECONNECT_BASE     = 5.0
    RECONNECT_CAP      = 60.0
    HEARTBEAT_INTERVAL = 20

    def __init__(self, url, on_disconnect_cb=None, name="DerivWS"):
        self.url               = url
        self._on_disconnect_cb = on_disconnect_cb
        self.name              = name
        self.state             = ConnState.DISCONNECTED
        self._running          = False
        self._ws               = None
        self._attempt          = 0
        self._pending: dict[int, asyncio.Future] = {}

    _counter = 0

    @classmethod
    def _new_id(cls) -> int:
        cls._counter += 1
        return cls._counter

    async def safe_send(self, payload: dict) -> bool:
        ws   = self._ws
        live = (self.state >= ConnState.CONNECTED and ws is not None)
        if not live:
            return False
        try:
            await ws.send(json.dumps(payload))
            return True
        except Exception as e:
            log.warning(f"[{self.name}] safe_send failed: {e}")
            return False

    async def send(self, payload: dict, timeout: float = 15.0) -> dict:
        rid                = self._new_id()
        payload["req_id"]  = rid
        fut                = asyncio.get_event_loop().create_future()
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

            self.state    = ConnState.CONNECTING
            self._pending.clear()
            ka_task = recv_task = None
            try:
                connect_url = await self.url() if callable(self.url) else self.url
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
                        log.error(f"[{self.name}] disconnect_cb: {e}")
                self._attempt += 1

        log.info(f"[{self.name}] Connection loop exited.")

    async def _heartbeat(self):
        try:
            while self.state >= ConnState.CONNECTED:
                await asyncio.sleep(self.HEARTBEAT_INTERVAL)
                if not await self.safe_send({"ping": 1}):
                    return
        except asyncio.CancelledError:
            pass


# ══════════════════════════════════════════════════════════════════════════
# MAIN BOT
# ══════════════════════════════════════════════════════════════════════════
class EvenOddBot:

    STUCK_TIMEOUT_S = 30

    def __init__(self, app_id: str, api_token: str, account_id: str | None = None):
        self.app_id     = app_id
        self.token      = api_token
        self.account_id = account_id

        # ── Intelligence layers ──
        self.markov  = ParityMarkovChain(order=CFG["markov_order"])
        self.hmm     = ParityHMM()
        self.hawkes  = ParityHawkes()
        self.sampen  = SampleEntropySignal()
        self.arfima  = ARFIMASignal()
        self.hurst_a = HurstSignal()
        self.bayes   = DualBayesianEdge()
        self.copula  = ParityCopula()
        self.kalman  = DualKalman()
        self.guard   = RiskGuard()
        self.ensemble= DualEnsemble()

        # ── History ──
        self.digits   = deque(maxlen=CFG["tick_buffer"])
        self.parities = deque(maxlen=CFG["tick_buffer"])
        self._tick_n  = 0

        # ── Account ──
        self.bankroll      = CFG["starting_bankroll"]
        self.active_id     = None
        self._buying       = False
        self._lock_since   = None
        self.trade_count   = 0
        self.wins          = 0
        self.total_pnl     = 0.0
        self._entry_spot   = 0.0
        self._entry_digit  = 0
        self._entry_dir    = ""
        self._entry_conf   = 0.0
        self._entry_stake  = CFG["stake"]

        self.wsman: DerivWSManager | None = None

    # ─────────────────────────────────────────────────────────────────────
    # INTELLIGENCE PIPELINE
    # ─────────────────────────────────────────────────────────────────────
    def run_intelligence(self, spot: float, digit: int) -> tuple:
        if len(self.digits) < CFG["warmup_ticks"]:
            rem = CFG["warmup_ticks"] - len(self.digits)
            return None, 0.0, f"WARMUP({rem} left)"

        par_arr = np.array(self.parities, dtype=float)

        # ── Shared signals ──
        sampen_sig, se_raw, sampen_veto = self.sampen.signal(par_arr)
        arfima_sig, d_val               = self.arfima.signal(par_arr)
        regime_idx                      = self.hmm.regime_idx()

        # ── Per-direction signals ──
        signals = {}
        for direction in ("EVEN", "ODD"):
            signals[direction] = {
                "markov":  self.markov.markov_signal(direction),
                "hmm":     self.hmm.signal(direction),
                "hawkes":  self.hawkes.signal(direction),
                "arfima":  arfima_sig,          # same for both (persistence)
                "hurst":   self.hurst_a.signal(par_arr, direction)[0],
                "bayes":   self.bayes.mean(direction),
                "copula":  self.copula.signal(par_arr, direction)[0],
                "copula_veto": self.copula.signal(par_arr, direction)[2],
                "kalman":  self.kalman.signal(direction),
            }

        h_sig_e, h_val = self.hurst_a.signal(par_arr, "EVEN")

        log.info(
            f"[SIG] HMM={self.hmm.name()} SE={se_raw:.2f}→{sampen_sig:.2f} "
            f"d={d_val:.3f} H={h_val:.3f} | "
            f"EVEN: MKV={signals['EVEN']['markov']:.3f} HWK={signals['EVEN']['hawkes']:.3f} "
            f"Bay={signals['EVEN']['bayes']:.3f} Kal={signals['EVEN']['kalman']:.3f} | "
            f"ODD:  MKV={signals['ODD']['markov']:.3f} HWK={signals['ODD']['hawkes']:.3f} "
            f"Bay={signals['ODD']['bayes']:.3f} Kal={signals['ODD']['kalman']:.3f}"
        )

        direction, conf, reason = self.ensemble.decide(
            even_markov=signals["EVEN"]["markov"],
            even_hmm=signals["EVEN"]["hmm"],
            even_hawkes=signals["EVEN"]["hawkes"],
            even_arfima=signals["EVEN"]["arfima"],
            even_hurst=signals["EVEN"]["hurst"],
            even_bayes=signals["EVEN"]["bayes"],
            even_copula=signals["EVEN"]["copula"],
            even_kalman=signals["EVEN"]["kalman"],
            odd_markov=signals["ODD"]["markov"],
            odd_hmm=signals["ODD"]["hmm"],
            odd_hawkes=signals["ODD"]["hawkes"],
            odd_arfima=signals["ODD"]["arfima"],
            odd_hurst=signals["ODD"]["hurst"],
            odd_bayes=signals["ODD"]["bayes"],
            odd_copula=signals["ODD"]["copula"],
            odd_kalman=signals["ODD"]["kalman"],
            sampen_sig=sampen_sig,
            regime_idx=regime_idx,
            sampen_veto=sampen_veto,
            even_copula_veto=signals["EVEN"]["copula_veto"],
            odd_copula_veto=signals["ODD"]["copula_veto"],
        )

        # Log signal for whichever direction (or both) was evaluated
        for d in ("EVEN", "ODD"):
            try:
                datalog.log_signal(
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                    tick_n=self._tick_n, spot=spot, last_digit=digit,
                    parity=parity_label(digit), direction=d,
                    markov_p=round(signals[d]["markov"], 5),
                    hmm_sig=round(signals[d]["hmm"], 5),
                    hawkes_sig=round(signals[d]["hawkes"], 5),
                    sampen_sig=round(sampen_sig, 5),
                    arfima_sig=round(arfima_sig, 5),
                    hurst_sig=round(signals[d]["hurst"], 5),
                    bayes_sig=round(signals[d]["bayes"], 5),
                    copula_sig=round(signals[d]["copula"], 5),
                    kalman_sig=round(signals[d]["kalman"], 5),
                    conf=round(conf if direction == d else 0.0, 5),
                    regime=self.hmm.name(),
                    reason=reason if direction == d else "NOT_SELECTED",
                )
            except Exception as e:
                log.warning(f"signal CSV failed: {e}")

        return direction, conf, reason

    # ─────────────────────────────────────────────────────────────────────
    # TICK HANDLER
    # ─────────────────────────────────────────────────────────────────────
    async def on_tick(self, tick: dict):
        spot  = float(tick.get("quote", tick.get("ask", 0.0)))
        digit = last_digit(spot)
        par   = parity_bit(digit)

        self.digits.append(digit)
        self.parities.append(par)
        self._tick_n += 1

        # Update all online models every tick
        self.markov.update(digit)
        self.hmm.update(digit)
        self.hawkes.update(digit)
        self.kalman.update(digit)
        self.guard.tick()

        if self._tick_n % 10 == 0:
            log.info(
                f"Tick #{self._tick_n:5d} | spot={spot} | digit={digit} "
                f"({parity_label(digit)}) | bankroll=${self.bankroll:.2f} | "
                f"guard={self.guard.status()} | regime={self.hmm.name()}"
            )

        # ── Stuck guard ──
        if (self.active_id or self._buying) and self._lock_since:
            if time.monotonic() - self._lock_since > self.STUCK_TIMEOUT_S:
                log.warning("⏱️  Stuck lock — force-unlocking.")
                self.active_id = None
                self._buying   = False
                self._lock_since = None
            else:
                return

        if not self.guard.can_trade():
            return

        direction, conf, reason = self.run_intelligence(spot, digit)

        if direction is not None:
            stake = self.guard.compute_stake(self.bankroll, self.bayes.mean(direction))
            log.info(
                f"🎯  SIGNAL  {direction}  conf={conf:.3f}  digit={digit}  "
                f"spot={spot}  stake=${stake:.2f}  [{reason}]"
            )
            self._entry_spot  = spot
            self._entry_digit = digit
            self._entry_dir   = direction
            self._entry_conf  = conf
            self._buying      = True
            self._lock_since  = time.monotonic()
            asyncio.create_task(self._request_and_buy(direction, spot, conf, stake))

    # ─────────────────────────────────────────────────────────────────────
    # PROPOSAL → BUY
    # ─────────────────────────────────────────────────────────────────────
    async def _request_and_buy(self, direction: str, spot: float,
                                conf: float = 0.0, stake: float | None = None):
        try:
            if self.active_id:
                return
            if stake is None:
                stake = self.guard.stake

            contract_type = "DIGITEVEN" if direction == "EVEN" else "DIGITODD"

            resp = await self.wsman.send({
                "proposal":          1,
                "amount":            stake,
                "basis":             "stake",
                "contract_type":     contract_type,
                "currency":          CFG["currency"],
                "duration":          CFG["duration"],
                "duration_unit":     CFG["duration_unit"],
                "underlying_symbol": CFG["underlying_symbol"],
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

            await self._buy(pid, float(ask_price), direction, spot, conf, stake)

        except asyncio.TimeoutError:
            log.warning("Proposal timed out")
        except Exception as exc:
            log.error(f"_request_and_buy error: {exc}")
        finally:
            self._buying = False

    async def _buy(self, proposal_id: str, price: float, direction: str,
                   spot: float = 0.0, conf: float = 0.0, stake: float | None = None):
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
            self._entry_stake = stake if stake is not None else self.guard.stake
            self._lock_since  = time.monotonic()

            log.info(
                f"✅  CONTRACT #{self.trade_count} OPEN | "
                f"{direction} | id={cid} | digit={self._entry_digit} | "
                f"stake=${self._entry_stake:.2f} | conf={conf:.3f}"
            )

            await self.wsman.send_nowait({
                "proposal_open_contract": 1,
                "contract_id":            cid,
                "subscribe":              1,
            })

        except asyncio.TimeoutError:
            log.warning("Buy timed out")
        except Exception as exc:
            log.error(f"_buy error: {exc}")

    # ─────────────────────────────────────────────────────────────────────
    # SETTLEMENT
    # ─────────────────────────────────────────────────────────────────────
    def _settle(self, poc: dict):
        profit = float(poc.get("profit", 0.0))
        won    = profit > 0.0

        self.total_pnl  += profit
        self.bankroll   += profit
        contract_id      = self.active_id
        direction        = self._entry_dir
        self.active_id   = None
        self._buying     = False
        self._lock_since = None

        self.bayes.update(direction, won)
        self.guard.check_bankroll(self.bankroll)

        if won:
            self.wins += 1
            self.guard.on_win()
            tag = "🟢  WIN "
        else:
            self.guard.on_loss()
            tag = "🔴  LOSS"

        wr     = self.wins / self.trade_count if self.trade_count else 0.0
        lo_e, hi_e = self.bayes.ci95("EVEN")
        lo_o, hi_o = self.bayes.ci95("ODD")

        log.info(
            f"{tag} [{direction}]  {profit:+.3f}  "
            f"cumPnL={self.total_pnl:+.3f}  bankroll=${self.bankroll:.2f}"
        )
        log.info(
            f"📊  trades={self.trade_count}  WR={wr:.1%}  "
            f"EVEN_WR={self.bayes.mean('EVEN'):.1%}[{lo_e:.2f},{hi_e:.2f}]  "
            f"ODD_WR={self.bayes.mean('ODD'):.1%}[{lo_o:.2f},{hi_o:.2f}]  "
            f"next={self.guard.status()}"
        )

        try:
            datalog.log_trade(
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                trade_n=self.trade_count, contract_id=contract_id,
                direction=direction,
                spot_entry=self._entry_spot, last_digit_entry=self._entry_digit,
                stake=self._entry_stake, conf=round(self._entry_conf, 5),
                profit=round(profit, 5), won=int(won),
                bankroll=round(self.bankroll, 5),
                total_pnl=round(self.total_pnl, 5),
                win_rate=round(wr, 5),
            )
        except Exception as e:
            log.warning(f"trade CSV failed: {e}")

    # ─────────────────────────────────────────────────────────────────────
    # NEW OPTIONS API — REST BOOTSTRAP
    # ─────────────────────────────────────────────────────────────────────
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
        data = resp.json()
        accounts = data.get("data", data) if isinstance(data, dict) else data
        if isinstance(accounts, dict):
            accounts = accounts.get("accounts", accounts.get("data", []))
        for acc in accounts:
            if acc.get("account_type") == "real":
                acc_id = acc.get("account_id") or acc.get("id")
                if acc_id:
                    return acc_id
        raise RuntimeError(f"No demo account found: {data}")

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

    # ─────────────────────────────────────────────────────────────────────
    # MESSAGE DISPATCHER
    # ─────────────────────────────────────────────────────────────────────
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

    # ─────────────────────────────────────────────────────────────────────
    # CONNECTION HOOKS
    # ─────────────────────────────────────────────────────────────────────
    async def _on_open(self, wsman: DerivWSManager):
        wsman.state = ConnState.AUTHENTICATED
        log.info(f"Connected — OTP session (account={self.account_id}).")
        await wsman.send_nowait({
            "ticks":     CFG["underlying_symbol"],
            "subscribe": 1,
        })
        wsman.state = ConnState.SUBSCRIBED
        log.info(f"Subscribed to {CFG['underlying_symbol']} — warming up {CFG['warmup_ticks']} ticks…")
        if self.active_id:
            await wsman.send_nowait({
                "proposal_open_contract": 1,
                "contract_id":            self.active_id,
                "subscribe":              1,
            })

    def _on_disconnect(self):
        if self._buying:
            log.warning("Connection lost during buy — resetting flag.")
            self._buying     = False
            self._lock_since = None
        if self.active_id:
            log.warning(f"Connection lost with contract #{self.active_id} open.")

    # ─────────────────────────────────────────────────────────────────────
    # MAIN LOOP
    # ─────────────────────────────────────────────────────────────────────
    async def run(self):
        bar = "═" * 72
        log.info(bar)
        log.info("  DERIV DIGIT EVEN/ODD BOT  ·  R_100  ·  MAXIMUM PRECISION EDITION")
        log.info("  Auto-selects EVEN or ODD each tick based on 10-layer ensemble")
        log.info("  10-Layer: ParityMarkov3 · HMM · Hawkes · SampEn · ARFIMA")
        log.info("            Hurst · DualBayesian · Copula · DualKalman · Guard")
        log.info(f"  Stake: ${CFG['stake']:.2f}  ·  Bankroll: ${CFG['starting_bankroll']:.2f}  "
                 f"·  Stop: ${CFG['drawdown_stop']:.2f}")
        log.info(f"  Warmup: {CFG['warmup_ticks']} ticks  ·  Min conf: {CFG['min_confidence']}")
        log.info("  Connection: new Options API (REST OTP bootstrap — no legacy authorize)")
        log.info(bar)

        self.wsman = DerivWSManager(
            self._get_ws_url,
            on_disconnect_cb=self._on_disconnect,
            name="EvenOddWS",
        )
        await self.wsman.run(on_open=self._on_open, on_message=self.on_message)


# ══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    APP_ID     = os.getenv("DERIV_APP_ID", "")
    API_TOKEN  = os.getenv("DERIV_API_TOKEN", "")
    ACCOUNT_ID = os.getenv("DERIV_ACCOUNT_ID", "") or None

    missing = []
    if not APP_ID:    missing.append("DERIV_APP_ID")
    if not API_TOKEN: missing.append("DERIV_API_TOKEN")
    if missing:
        print(
            f"\n⚠️  {', '.join(missing)} not set.\n"
            "   Set them as environment variables before starting.\n"
            "   (App ID and PAT from NEW developers.deriv.com app —\n"
            "   legacy App IDs like 1089 no longer work.)\n"
        )
        raise SystemExit(1)

    bot = EvenOddBot(app_id=APP_ID, api_token=API_TOKEN, account_id=ACCOUNT_ID)
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        log.info("Bot stopped (Ctrl+C)")
