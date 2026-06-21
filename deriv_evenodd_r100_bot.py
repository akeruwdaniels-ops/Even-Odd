"""
Deriv Multi-Symbol Rise/Fall Trading Bot
==========================================
Single-file bot for deployment on Railway. Scans all eligible synthetic-index
symbols (1HZ variants excluded), runs an 11-layer intelligence pipeline per
symbol, fuses the evidence into a single directional probability via a
Bayesian layer, auto-selects trade duration via Monte Carlo simulation,
and trades the single strongest signal at a time with a 1.24x / 3-step
martingale and balance-scaled staking.

A symbol calibrator runs on a 2-hour schedule and on per-symbol 2-loss
triggers (rate-limited), pausing trading while it re-validates symbols and
updates per-symbol reliability multipliers.

NOTE ON SIMPLIFICATIONS: several layers (HMM, GARCH, ARFIMA, copula) are
implemented as lightweight, dependency-free numpy approximations rather than
full statsmodels/hmmlearn/arch implementations, to keep the Railway deploy
to two pip packages (websockets, numpy). Swap in heavier libraries later if
you want more statistical rigor once this is validated end to end.

ENV VARS REQUIRED:
    DERIV_API_TOKEN   - API token for your Deriv account
    DERIV_APP_ID      - your registered app_id (defaults to demo app id 1089)
"""

import asyncio
import websockets
import json
import os
import time
import math
import numpy as np
from collections import deque, defaultdict
from datetime import datetime

# ---------------------------------------------------------------------------
# CONFIG  (tune these via your walk-forward validation before going live)
# ---------------------------------------------------------------------------
DERIV_APP_ID = os.getenv("DERIV_APP_ID", "1089")
DERIV_API_TOKEN = os.getenv("DERIV_API_TOKEN")
WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"

MIN_STAKE = 0.35
STAKE_PCT = 0.02                       # stake = max(MIN_STAKE, balance * STAKE_PCT)

MARTINGALE_FACTOR = 1.24
MARTINGALE_MAX_STEPS = 3               # up to 3 recovery steps after the initial stake

SCHEDULED_CALIBRATION_INTERVAL = 2 * 60 * 60   # seconds (2 hours)
LOSS_TRIGGER_THRESHOLD = 2                     # consecutive losses on the SAME symbol
MAX_LOSS_CALIBRATIONS_PER_24H = 3              # rate limiter, default - tune as needed
CALIBRATION_COOLDOWN = 5 * 60                  # grace period after calibration ends
TOP_K_DEEP_DIVE = 5                            # symbols deep-validated per calibration

CONFIDENCE_THRESHOLD = 0.58            # minimum ensemble score to trade (0-1 scale)
MIN_SCORE_GAP = 0.03                   # required gap over runner-up symbol
CANDIDATE_DURATIONS = [1, 3, 5, 10, 15]  # ticks, Monte Carlo picks the best of these

MIN_TICKS_REQUIRED = 60                # minimum ticks buffered before a symbol is scored


# ---------------------------------------------------------------------------
# SHARED STATE  (single source of truth - every module reads/writes through this)
# ---------------------------------------------------------------------------
class TradeState:
    def __init__(self):
        self.balance = 0.0
        self.trading_locked = False
        self.trade_in_progress = False
        self.consecutive_losses = defaultdict(int)      # symbol -> count
        self.reliability = defaultdict(lambda: 1.0)      # symbol -> multiplier
        self.loss_triggered_calibrations_24h = deque()   # timestamps
        self.last_scheduled_calibration = time.time()
        self.last_calibration_end = 0.0


class SymbolData:
    def __init__(self, symbol, maxlen=2000):
        self.symbol = symbol
        self.ticks = deque(maxlen=maxlen)  # (epoch, price)

    def add_tick(self, epoch, price):
        self.ticks.append((epoch, price))

    def prices(self):
        return np.array([p for _, p in self.ticks], dtype=float)

    def returns(self):
        p = self.prices()
        if len(p) < 2:
            return np.array([])
        return np.diff(p) / p[:-1]


# ---------------------------------------------------------------------------
# DERIV API CLIENT (minimal async wrapper over the public websocket API)
# ---------------------------------------------------------------------------
class DerivClient:
    def __init__(self, ws_url, token):
        self.ws_url = ws_url
        self.token = token
        self.ws = None
        self.req_id = 0
        self.pending = {}
        self.subscriptions = defaultdict(list)  # msg_type -> list[asyncio.Queue]
        self.account = None

    async def connect(self):
        self.ws = await websockets.connect(self.ws_url, ping_interval=20)
        asyncio.create_task(self._reader())
        auth = await self.send({"authorize": self.token})
        if "error" in auth:
            raise RuntimeError(f"Authorization failed: {auth['error']}")
        self.account = auth["authorize"]
        return self.account

    async def _reader(self):
        async for message in self.ws:
            data = json.loads(message)
            req_id = data.get("req_id")
            msg_type = data.get("msg_type")
            if req_id is not None and req_id in self.pending:
                fut = self.pending.pop(req_id)
                if not fut.done():
                    fut.set_result(data)
                    continue
            if msg_type in self.subscriptions:
                for q in self.subscriptions[msg_type]:
                    await q.put(data)

    async def send(self, request, timeout=20):
        self.req_id += 1
        rid = self.req_id
        request = dict(request)
        request["req_id"] = rid
        fut = asyncio.get_event_loop().create_future()
        self.pending[rid] = fut
        await self.ws.send(json.dumps(request))
        return await asyncio.wait_for(fut, timeout=timeout)

    def subscribe_channel(self, msg_type):
        q = asyncio.Queue()
        self.subscriptions[msg_type].append(q)
        return q


async def fetch_tradable_symbols(client):
    """Builds the symbol universe dynamically: synthetic indices only,
    1HZ variants excluded, and only symbols that actually support CALL/PUT
    (rise/fall) contracts. Avoids hardcoding symbol codes that drift over time."""
    resp = await client.send({"active_symbols": "brief", "product_type": "basic"})
    candidates = []
    for s in resp.get("active_symbols", []):
        symbol = s["symbol"]
        if "1HZ" in symbol:
            continue
        if s.get("market") != "synthetic_index":
            continue
        if not s.get("exchange_is_open", 1):
            continue
        candidates.append(symbol)

    verified = []
    for symbol in candidates:
        try:
            cf = await client.send({"contracts_for": symbol, "currency": "USD"})
            types = {c["contract_type"] for c in cf.get("contracts_for", {}).get("available", [])}
            if "CALL" in types and "PUT" in types:
                verified.append(symbol)
        except Exception:
            continue
    return verified


async def buy_contract(client, symbol, direction, duration, duration_unit, stake):
    contract_type = "CALL" if direction > 0 else "PUT"
    req = {
        "buy": 1,
        "price": stake,
        "parameters": {
            "amount": stake,
            "basis": "stake",
            "contract_type": contract_type,
            "currency": "USD",
            "duration": duration,
            "duration_unit": duration_unit,
            "symbol": symbol,
        },
    }
    resp = await client.send(req)
    if "error" in resp:
        raise RuntimeError(resp["error"].get("message", "buy failed"))
    return resp["buy"]["contract_id"]


async def wait_for_contract_result(client, contract_id):
    q = client.subscribe_channel("proposal_open_contract")
    await client.send({"proposal_open_contract": 1, "contract_id": contract_id, "subscribe": 1})
    while True:
        data = await q.get()
        poc = data.get("proposal_open_contract", {})
        if poc.get("contract_id") == contract_id and poc.get("is_sold"):
            profit = float(poc.get("profit", 0))
            return profit > 0, profit


# ---------------------------------------------------------------------------
# INTELLIGENCE LAYERS
# ---------------------------------------------------------------------------
def markov_directional_prob(returns, order=1):
    """Layer 1: P(next tick up | last `order` directional states)."""
    signs = np.sign(returns)
    signs = signs[signs != 0]
    if len(signs) < order + 10:
        return 0.5
    table = {}
    for i in range(len(signs) - order):
        state = tuple(signs[i:i + order])
        table.setdefault(state, []).append(signs[i + order])
    current_state = tuple(signs[-order:])
    outcomes = table.get(current_state)
    if not outcomes:
        return 0.5
    return float(np.mean(np.array(outcomes) > 0))


def hmm_regime(returns):
    """Layer 2: lightweight regime classifier (trending / ranging / volatile).
    Outputs trend_weight: how much to trust momentum layers right now."""
    if len(returns) < 30:
        return {"regime": "unknown", "trend_weight": 0.5}
    recent = returns[-30:]
    vol_recent = np.std(recent)
    vol_overall = np.std(returns) if np.std(returns) > 0 else 1e-9
    if vol_recent > vol_overall * 1.3:
        return {"regime": "volatile", "trend_weight": 0.3}
    mean_recent = np.mean(recent)
    if abs(mean_recent) > vol_recent * 0.3:
        return {"regime": "trending", "trend_weight": 0.75}
    return {"regime": "ranging", "trend_weight": 0.35}


def hawkes_intensity(returns, decay=0.3, lookback=50):
    """Layer 3: self-exciting momentum clustering. Positive = up-momentum bursts."""
    if len(returns) < 5:
        return 0.0
    recent = returns[-lookback:]
    weights = np.exp(-decay * np.arange(len(recent))[::-1])
    signed = np.sign(recent)
    return float(np.sum(weights * signed) / np.sum(weights))


def ou_reversion_signal(prices, lookback=100):
    """Layer 4: Ornstein-Uhlenbeck style mean reversion distance."""
    lb = min(lookback, len(prices))
    if lb < 10:
        return {"z": 0.0, "reversion_dir": 0.0, "strength": 0.0}
    window = prices[-lb:]
    mu = np.mean(window)
    sigma = np.std(window) if np.std(window) > 0 else 1e-9
    z = (prices[-1] - mu) / sigma
    return {"z": z, "reversion_dir": float(-np.sign(z)), "strength": float(min(abs(z) / 2, 1.0))}


def hurst_exponent(prices, max_lag=20):
    """Layer 5: persistence (>0.5 momentum favored) vs anti-persistence (<0.5 reversion favored)."""
    if len(prices) < max_lag * 2:
        return 0.5
    lags = range(2, max_lag)
    tau = [np.std(np.subtract(prices[lag:], prices[:-lag])) for lag in lags]
    tau = [t if t > 0 else 1e-9 for t in tau]
    poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
    h = poly[0] * 2.0
    return float(np.clip(h, 0.0, 1.0))


def arfima_bias(returns, d=0.3, lookback=80):
    """Layer 6: fractional-differencing weighted long-memory directional bias."""
    if len(returns) < 10:
        return 0.0
    recent = returns[-lookback:]
    n = len(recent)
    weights = np.array([math.gamma(k - d) / (math.gamma(-d) * math.gamma(k + 1)) for k in range(n)])
    weights = weights / np.sum(np.abs(weights))
    bias = np.sum(weights[::-1] * recent)
    return float(np.tanh(bias * 50))


def garch_volatility_trust(returns, alpha=0.1, beta=0.85):
    """Layer 7: GARCH(1,1)-style volatility regime filter. Lower trust when vol is spiking."""
    if len(returns) < 20:
        return {"vol": 0.0, "trust": 0.5}
    omega = np.var(returns) * (1 - alpha - beta)
    if omega <= 0:
        omega = 1e-8
    var = np.var(returns[:10])
    for r in returns:
        var = omega + alpha * r ** 2 + beta * var
    current_vol = math.sqrt(max(var, 1e-12))
    baseline_vol = np.std(returns) if np.std(returns) > 0 else 1e-9
    ratio = current_vol / baseline_vol
    trust = 1.0 / (1.0 + max(ratio - 1, 0) * 2)
    return {"vol": current_vol, "trust": float(np.clip(trust, 0.1, 1.0))}


def entropy_trust(returns, bins=10):
    """Layer 8: Shannon entropy of recent returns. Low entropy (structured) = high trust."""
    if len(returns) < 20:
        return 0.5
    hist, _ = np.histogram(returns[-100:], bins=bins)
    probs = hist / np.sum(hist)
    probs = probs[probs > 0]
    ent = -np.sum(probs * np.log(probs))
    max_ent = np.log(bins)
    norm_ent = ent / max_ent if max_ent > 0 else 0.5
    return float(np.clip(1.0 - norm_ent, 0.1, 1.0))


def kalman_trend(prices, q=1e-5, r=0.01):
    """Layer 9: Kalman-filtered instantaneous trend slope, noise-reduced."""
    if len(prices) < 5:
        return 0.0
    x, p, slope = prices[0], 1.0, 0.0
    for price in prices[1:]:
        p = p + q
        k = p / (p + r)
        innovation = price - x
        x = x + k * innovation
        p = (1 - k) * p
        slope = innovation
    denom = np.std(prices) + 1e-9
    return float(np.sign(slope) * min(abs(slope) / denom, 1.0))


def copula_agreement(symbol, all_momentum):
    """Layer 10: cross-symbol confirmation proxy. Real copula fitting (e.g. Gaussian
    copula via scipy) can replace this grouping heuristic once you have more history."""
    peers = {s: v for s, v in all_momentum.items() if s != symbol}
    if not peers:
        return 0.5
    target_sign = np.sign(all_momentum.get(symbol, 0))
    agree = [1 if np.sign(v) == target_sign else 0 for v in peers.values()]
    return float(np.mean(agree)) if agree else 0.5


def bayesian_fusion(features):
    """Layer 11: final fusion layer. Combines every other layer's evidence into one
    P(up) and a confidence score. This is the ONLY layer that owns the directional call."""
    momentum_score = features["hawkes"] * features["trend_weight"] + features["kalman"] * 0.3
    reversion_score = features["ou_dir"] * features["ou_strength"] * (1 - features["trend_weight"])
    hurst_weight = features["hurst"]
    momentum_component = momentum_score * hurst_weight
    reversion_component = reversion_score * (1 - hurst_weight)
    arfima_component = features["arfima_bias"] * 0.2
    markov_component = (features["markov_p"] - 0.5) * 2

    directional_lean = momentum_component + reversion_component
    copula_component = features["copula_agree"] * np.sign(directional_lean + 1e-9)

    raw_signal = (
        0.25 * markov_component
        + 0.25 * momentum_component
        + 0.20 * reversion_component
        + 0.15 * arfima_component
        + 0.15 * copula_component
    )
    trust_multiplier = features["vol_trust"] * features["entropy_trust"]
    raw_signal *= trust_multiplier

    p_up = float(np.clip(0.5 + raw_signal / 2, 0.01, 0.99))
    confidence = abs(p_up - 0.5) * 2 * trust_multiplier
    return p_up, confidence


def compute_features(sd, all_momentum):
    prices = sd.prices()
    returns = sd.returns()
    hmm = hmm_regime(returns)
    ou = ou_reversion_signal(prices)
    garch = garch_volatility_trust(returns)
    return {
        "markov_p": markov_directional_prob(returns),
        "hawkes": hawkes_intensity(returns),
        "trend_weight": hmm["trend_weight"],
        "ou_dir": ou["reversion_dir"],
        "ou_strength": ou["strength"],
        "hurst": hurst_exponent(prices),
        "arfima_bias": arfima_bias(returns),
        "vol_trust": garch["trust"],
        "entropy_trust": entropy_trust(returns),
        "kalman": kalman_trend(prices),
        "copula_agree": copula_agreement(sd.symbol, all_momentum),
    }


def monte_carlo_duration(prices, returns, direction, candidate_durations, n_sims=300):
    """Layer 12: Monte Carlo duration selector. Takes the direction already decided
    by the Bayesian layer and finds which duration maximizes expected win probability.
    Does NOT re-decide direction - only times it."""
    if len(returns) < 20:
        return candidate_durations[0], 0.5
    vol = np.std(returns[-50:]) if len(returns) >= 50 else np.std(returns)
    vol = vol if vol > 0 else 1e-6
    drift = direction * abs(np.mean(returns[-50:])) if len(returns) >= 50 else 0.0
    best = None
    for dur in candidate_durations:
        sim_returns = np.random.normal(drift, vol, size=(n_sims, dur))
        path_totals = np.sum(sim_returns, axis=1)
        wins = np.sum((path_totals > 0) if direction > 0 else (path_totals < 0))
        win_rate = wins / n_sims
        if best is None or win_rate > best[1]:
            best = (dur, win_rate)
    return best


# ---------------------------------------------------------------------------
# ENSEMBLE SELECTOR
# ---------------------------------------------------------------------------
def select_trade(symbol_scores, reliability):
    scored = []
    for symbol, (p_up, confidence) in symbol_scores.items():
        score = confidence * reliability.get(symbol, 1.0)
        direction = 1 if p_up > 0.5 else -1
        scored.append((symbol, direction, p_up, score))
    if not scored:
        return None
    scored.sort(key=lambda x: x[3], reverse=True)
    top = scored[0]
    if top[3] < CONFIDENCE_THRESHOLD:
        return None
    if len(scored) > 1 and (top[3] - scored[1][3]) < MIN_SCORE_GAP:
        return None
    return top  # (symbol, direction, p_up, score)


# ---------------------------------------------------------------------------
# STAKING
# ---------------------------------------------------------------------------
def calculate_stake(balance):
    """stake = max($0.35, 2% of balance) - single formula, no seam/discontinuity."""
    return round(max(MIN_STAKE, balance * STAKE_PCT), 2)


def martingale_stakes(base_stake):
    stakes = [round(base_stake, 2)]
    for _ in range(MARTINGALE_MAX_STEPS):
        stakes.append(round(stakes[-1] * MARTINGALE_FACTOR, 2))
    return stakes


# ---------------------------------------------------------------------------
# TRADE EXECUTION
# ---------------------------------------------------------------------------
def log_trade(symbol, direction, stake, won, profit, step):
    ts = datetime.utcnow().isoformat()
    side = "CALL" if direction > 0 else "PUT"
    print(f"[{ts}] {symbol} {side} step={step} stake={stake:.2f} won={won} profit={profit:.2f}")


async def execute_sequence(client, state, symbol, direction, duration):
    state.trade_in_progress = True
    base_stake = calculate_stake(state.balance)
    stakes = martingale_stakes(base_stake)
    sequence_won = False
    try:
        for step, stake in enumerate(stakes):
            contract_id = await buy_contract(client, symbol, direction, duration, "t", stake)
            won, profit = await wait_for_contract_result(client, contract_id)
            log_trade(symbol, direction, stake, won, profit, step)
            if won:
                sequence_won = True
                break
    except Exception as e:
        print(f"Trade error on {symbol}: {e}")

    state.consecutive_losses[symbol] = 0 if sequence_won else state.consecutive_losses[symbol] + 1

    try:
        bal_resp = await client.send({"balance": 1})
        state.balance = bal_resp["balance"]["balance"]
    except Exception:
        pass

    state.trade_in_progress = False


# ---------------------------------------------------------------------------
# SYMBOL CALIBRATOR (trigger manager + calibration engine)
# ---------------------------------------------------------------------------
def check_calibration_triggers(state):
    now = time.time()
    if now - state.last_calibration_end < CALIBRATION_COOLDOWN:
        return None
    if now - state.last_scheduled_calibration >= SCHEDULED_CALIBRATION_INTERVAL:
        return "scheduled", None
    for symbol, count in list(state.consecutive_losses.items()):
        if count >= LOSS_TRIGGER_THRESHOLD:
            recent = [t for t in state.loss_triggered_calibrations_24h if now - t < 86400]
            state.loss_triggered_calibrations_24h = deque(recent)
            if len(recent) < MAX_LOSS_CALIBRATIONS_PER_24H:
                return "loss_triggered", symbol
    return None


def light_walk_forward(sd, window=200, step=20):
    """Simplified walk-forward proxy used by the calibrator's deep-dive step.
    Replace with a call into your full walk_forward_validator.py logic for more
    rigorous out-of-sample testing once this is wired up end to end."""
    prices = sd.prices()
    returns = sd.returns()
    if len(prices) < window + step + 10:
        return 0.5
    hits, total = 0, 0
    for i in range(window, len(prices) - step, step):
        hist_returns = returns[:i - 1]
        if len(hist_returns) < 30:
            continue
        p_up = 0.5 + np.tanh(hawkes_intensity(hist_returns)) / 2
        actual_dir = 1 if prices[i + step - 1] > prices[i] else -1
        predicted_dir = 1 if p_up > 0.5 else -1
        hits += int(predicted_dir == actual_dir)
        total += 1
    return hits / total if total > 0 else 0.5


async def run_calibration(state, symbol_data, symbols, trigger_reason):
    state.trading_locked = True
    kind, symbol = trigger_reason
    print(f"Calibration starting (trigger={kind}{':' + symbol if symbol else ''}), trading locked.")
    start = time.time()
    if kind == "loss_triggered":
        state.loss_triggered_calibrations_24h.append(start)

    # cheap broad scan across all symbols using already-computed live metrics
    scan_scores = {}
    for s in symbols:
        sd = symbol_data[s]
        if len(sd.ticks) < MIN_TICKS_REQUIRED:
            continue
        returns = sd.returns()
        scan_scores[s] = entropy_trust(returns) * garch_volatility_trust(returns)["trust"]

    top_k = sorted(scan_scores, key=scan_scores.get, reverse=True)[:TOP_K_DEEP_DIVE]
    # if this was a loss-triggered calibration, always include that symbol in the deep dive
    if symbol and symbol not in top_k:
        top_k.append(symbol)

    for s in top_k:
        hit_rate = light_walk_forward(symbol_data[s])
        state.reliability[s] = float(np.clip(hit_rate / 0.5, 0.3, 1.5))
        state.consecutive_losses[s] = 0

    state.last_scheduled_calibration = time.time()
    state.last_calibration_end = time.time()
    elapsed = state.last_calibration_end - start
    print(f"Calibration complete in {elapsed:.1f}s. Deep-dived: {top_k}")
    state.trading_locked = False


# ---------------------------------------------------------------------------
# STREAM CONSUMERS
# ---------------------------------------------------------------------------
async def tick_consumer(queue, symbol_data):
    while True:
        data = await queue.get()
        tick = data.get("tick")
        if not tick:
            continue
        symbol = tick.get("symbol")
        if symbol in symbol_data:
            symbol_data[symbol].add_tick(tick["epoch"], tick["quote"])


async def balance_consumer(queue, state):
    while True:
        data = await queue.get()
        bal = data.get("balance")
        if bal:
            state.balance = bal["balance"]


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
async def main():
    if not DERIV_API_TOKEN:
        raise RuntimeError("Set the DERIV_API_TOKEN environment variable.")

    client = DerivClient(WS_URL, DERIV_API_TOKEN)
    account = await client.connect()
    print(f"Authorized as {account.get('loginid')}")

    balance_resp = await client.send({"balance": 1, "subscribe": 1})
    balance_queue = client.subscribe_channel("balance")

    state = TradeState()
    state.balance = balance_resp["balance"]["balance"]
    print(f"Starting balance: {state.balance}")

    symbols = await fetch_tradable_symbols(client)
    if not symbols:
        raise RuntimeError("No tradable rise/fall symbols found (check API credentials/connectivity).")
    print(f"Tradable universe ({len(symbols)} symbols, 1HZ excluded): {symbols}")

    symbol_data = {s: SymbolData(s) for s in symbols}
    tick_queue = client.subscribe_channel("tick")
    for s in symbols:
        await client.send({"ticks": s, "subscribe": 1})

    asyncio.create_task(tick_consumer(tick_queue, symbol_data))
    asyncio.create_task(balance_consumer(balance_queue, state))

    print("Bot running. Entering main decision loop.")
    while True:
        await asyncio.sleep(2)

        if state.trading_locked or state.trade_in_progress:
            continue

        trigger = check_calibration_triggers(state)
        if trigger:
            await run_calibration(state, symbol_data, symbols, trigger)
            continue

        # pass 1: raw momentum per symbol, needed for the copula confirmation layer
        raw_momentum = {}
        for s in symbols:
            sd = symbol_data[s]
            if len(sd.ticks) < MIN_TICKS_REQUIRED:
                continue
            returns = sd.returns()
            raw_momentum[s] = hawkes_intensity(returns) * 0.6 + kalman_trend(sd.prices()) * 0.4

        if not raw_momentum:
            continue

        # pass 2: full feature fusion per symbol
        symbol_scores = {}
        for s in raw_momentum:
            sd = symbol_data[s]
            feats = compute_features(sd, raw_momentum)
            p_up, confidence = bayesian_fusion(feats)
            symbol_scores[s] = (p_up, confidence)

        pick = select_trade(symbol_scores, state.reliability)
        if not pick:
            continue

        symbol, direction, p_up, score = pick
        sd = symbol_data[symbol]
        duration, exp_win_rate = monte_carlo_duration(
            sd.prices(), sd.returns(), direction, CANDIDATE_DURATIONS
        )
        print(
            f"Selected {symbol} dir={'UP' if direction > 0 else 'DOWN'} "
            f"p_up={p_up:.3f} score={score:.3f} duration={duration}t exp_win={exp_win_rate:.2f}"
        )
        await execute_sequence(client, state, symbol, direction, duration)


if __name__ == "__main__":
    asyncio.run(main())
