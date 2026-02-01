# data.py
# QTC EduFund - Market Data Downloader (minute bars -> Parquet)
# Uses: GET /api/v1/market-data/symbols
#       GET /api/v1/market-data/bars?symbols=...&start=...&end=...&key=...
#
# Output:
#   data/minute/AAPL.parquet
#   data/minute/_manifest.json

import os
import time
import json
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any

import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed


# =========================
# CONFIG
# =========================
BASE_URL = "https://api.qtcq.xyz"

QTC_API_KEY = ""

OUT_DIR = "data/minute"

# symbols per time
MAX_SYMBOLS_PER_REQUEST = 20

WORKERS = 3

# limit rate
SLEEP_PER_REQUEST = 0.20

# how many days interested
N_TRADING_DAYS = 10

MAX_SYMBOLS = None  # e.g. 50 for smoke test

DAY_START = "09:30:00"
DAY_END = "16:00:00"

PROBE_SYMBOL = "AAPL"

PROBE_LOOKBACK_DAYS = 180

SKIP_IF_EXISTS = True

# =========================
# END CONFIG
# =========================


@dataclass
class FetchStats:
    ok_requests: int = 0
    rate_limited: int = 0
    other_errors: int = 0


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "QTC-MarketData-Downloader/1.0",
        "Accept": "application/json",
    })
    return s


def robust_get_json(
    session: requests.Session,
    url: str,
    params: dict,
    stats: Optional[FetchStats] = None,
    timeout: int = 20,
    max_retries: int = 8,
) -> dict:
    """
    Robust GET:
    - 429: exponential backoff + jitter
    - 5xx/timeouts: retry
    - 401/403/422: fail fast with message (usually key/params/date issue)
    """
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            r = session.get(url, params=params, timeout=timeout)

            if r.status_code == 429:
                if stats:
                    stats.rate_limited += 1
                sleep = min(2 ** attempt, 20) * random.uniform(0.8, 1.2)
                print(f"[429] rate limited, sleep={sleep:.1f}s")
                time.sleep(sleep)
                continue

            # Fast-fail for likely "your request is wrong"
            if r.status_code in (401, 403, 422):
                msg = r.text[:400]
                raise RuntimeError(f"HTTP {r.status_code}: {msg}")

            # no data might be 404 depending on implementation; treat as empty.
            if r.status_code == 404:
                return {}

            if 500 <= r.status_code < 600:
                sleep = min(2 ** attempt, 20) * random.uniform(0.8, 1.2)
                time.sleep(sleep)
                continue

            r.raise_for_status()
            if stats:
                stats.ok_requests += 1
            return r.json()

        except Exception as e:
            last_err = e
            if stats:
                stats.other_errors += 1
            sleep = min(2 ** attempt, 20) * random.uniform(0.8, 1.2)
            time.sleep(sleep)

    raise RuntimeError(f"GET failed after retries: {url} | last_err={last_err}")


def get_symbols(session: requests.Session) -> List[str]:
    url = f"{BASE_URL}/api/v1/market-data/symbols"
    js = robust_get_json(session, url, params={})
    syms = js.get("symbols", [])
    if not syms:
        raise RuntimeError("No symbols returned. API might be down or path changed.")
    return syms


def bars_request(
    session: requests.Session,
    symbols: List[str],
    start_dt: str,
    end_dt: str,
    stats: Optional[FetchStats] = None,
) -> dict:
    """
    GET /api/v1/market-data/bars
    Required: symbols, start, end, key
    """
    url = f"{BASE_URL}/api/v1/market-data/bars"
    params = {
        "symbols": ",".join(symbols),
        "start": start_dt,
        "end": end_dt,
        "key": QTC_API_KEY,
    }
    return robust_get_json(session, url, params=params, stats=stats)


# --- schema normalization (fix volume) ---
_PRINTED_DEBUG_ONCE = False


def normalize_bar_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize to columns:
      symbol, timestamp, open, high, low, close, volume

    If volume missing, create it (NA).
    Supports common aliases: vol, v, qty, size, volume_traded, etc.
    """
    global _PRINTED_DEBUG_ONCE

    if df.empty:
        return df

    rename_map = {
        "datetime": "timestamp",
        "time": "timestamp",

        # volume aliases
        "vol": "volume",
        "v": "volume",
        "qty": "volume",
        "size": "volume",
        "volume_traded": "volume",
        "trade_volume": "volume",
    }
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})

    required = ["symbol", "timestamp", "open", "high", "low", "close"]
    missing_req = [c for c in required if c not in df.columns]
    if missing_req:
        if not _PRINTED_DEBUG_ONCE:
            _PRINTED_DEBUG_ONCE = True
            print("[DEBUG] Missing required columns:", missing_req)
            print("[DEBUG] Available columns:", list(df.columns))
            try:
                print("[DEBUG] Sample rows:", df.head(3).to_dict(orient="records"))
            except Exception:
                pass
        # Return empty to avoid writing broken schema
        return pd.DataFrame()

    if "volume" not in df.columns:
        df["volume"] = pd.NA

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values(["symbol", "timestamp"])
    return df


def parse_bars_response(js: dict) -> pd.DataFrame:
    """
    OpenAPI describes two response shapes:

    Single:
      {
        "symbol": "AAPL",
        "bars": [ {timestamp, open, high, low, close, volume?}, ... ]
      }

    Multi:
      {
        "symbols": [...],
        "data": {
          "AAPL": {"bar_count":..., "bars":[...]},
          "SPY": {"bar_count":..., "bars":[...]},
          ...
        }
      }

    Returns a long dataframe with (standardized):
      symbol, timestamp, open, high, low, close, volume
    """
    if not js:
        return pd.DataFrame()

    rows: List[dict] = []

    # Multi-shape
    if isinstance(js, dict) and isinstance(js.get("data"), dict):
        data = js["data"]
        for sym, pack in data.items():
            bars = (pack or {}).get("bars", []) or []
            for b in bars:
                b2 = dict(b)
                b2["symbol"] = sym
                rows.append(b2)

    # Single-shape
    elif isinstance(js, dict) and isinstance(js.get("bars"), list) and "symbol" in js:
        sym = js["symbol"]
        for b in js["bars"]:
            b2 = dict(b)
            b2["symbol"] = sym
            rows.append(b2)

    else:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return normalize_bar_columns(df)


def date_to_window(day: datetime.date) -> Tuple[str, str]:
    """
    Build start/end ISO strings used by this API.
    Using docs-style without 'Z' is safest (their examples omit Z).
    """
    d = day.isoformat()
    start = f"{d}T{DAY_START}"
    end = f"{d}T{DAY_END}"
    return start, end


def find_recent_available_days(session: requests.Session, need_days: int) -> List[datetime.date]:
    """
    Probe backwards to find dates where PROBE_SYMBOL has at least some bars.
    """
    print(f"Probing available days using symbol={PROBE_SYMBOL} ...")
    today = datetime.now(timezone.utc).date()
    found: List[datetime.date] = []
    stats = FetchStats()

    for i in range(PROBE_LOOKBACK_DAYS):
        day = today - timedelta(days=i)
        if day.weekday() >= 5:  # weekend
            continue

        start, end = date_to_window(day)
        try:
            js = bars_request(session, [PROBE_SYMBOL], start, end, stats=stats)
            df = parse_bars_response(js)
            if not df.empty:
                found.append(day)
                print(f"[OK] {day} has bars for {PROBE_SYMBOL} (found {len(found)}/{need_days})")
                if len(found) >= need_days:
                    break
        except Exception as e:
            print(f"[WARN] probe {day} failed: {e}")

        time.sleep(0.10 + random.uniform(0, 0.05))

    print(f"Probe done. ok_requests={stats.ok_requests}, rate_limited={stats.rate_limited}, other_errors={stats.other_errors}")
    return found


def chunk_list(xs: List[str], n: int) -> List[List[str]]:
    return [xs[i:i+n] for i in range(0, len(xs), n)]


def save_symbol_parquet(out_dir: str, symbol: str, df_symbol: pd.DataFrame) -> int:
    """
    Save a single symbol's bars to Parquet.
    If file exists, we append+dedupe (safe if rerun with overlapping days).
    """
    ensure_dir(out_dir)
    path = os.path.join(out_dir, f"{symbol}.parquet")

    df_symbol = df_symbol.copy()
    df_symbol = df_symbol.sort_values("timestamp")

    if os.path.exists(path):
        old = pd.read_parquet(path)
        merged = pd.concat([old, df_symbol], ignore_index=True)
        merged.drop_duplicates(subset=["timestamp"], inplace=True)
        merged = merged.sort_values("timestamp")
        merged.to_parquet(path, index=False)
        return len(merged)
    else:
        df_symbol.to_parquet(path, index=False)
        return len(df_symbol)


def fetch_for_day(
    day: datetime.date,
    symbols: List[str],
    out_dir: str,
) -> Tuple[datetime.date, Dict[str, int]]:
    """
    Fetch data for one day, all symbols, using batched multi-symbol requests (max 20).
    Returns rows saved per symbol (counts after save/merge).
    """
    session = make_session()
    stats = FetchStats()

    start, end = date_to_window(day)
    symbol_batches = chunk_list(symbols, MAX_SYMBOLS_PER_REQUEST)

    print(f"\n=== Fetch day {day} | batches={len(symbol_batches)} | window={start}..{end} ===")

    rows_saved: Dict[str, int] = {}
    cols = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]

    for batch_idx, batch in enumerate(symbol_batches, start=1):
        try:
            js = bars_request(session, batch, start, end, stats=stats)
            df = parse_bars_response(js)

            if not df.empty:
                for sym, g in df.groupby("symbol"):
                    # parse_bars_response 已经保证 volume 存在（缺失时补 NA），不会再 KeyError
                    new_count = save_symbol_parquet(out_dir, sym, g[cols])
                    rows_saved[sym] = new_count

        except Exception as e:
            print(f"[WARN] day={day} batch={batch_idx}/{len(symbol_batches)} failed: {e}")

        time.sleep(SLEEP_PER_REQUEST + random.uniform(0, 0.05))

    print(f"Day {day} done. ok_requests={stats.ok_requests}, rate_limited={stats.rate_limited}, other_errors={stats.other_errors}")
    return day, rows_saved


def main():
    if not QTC_API_KEY or QTC_API_KEY == "REPLACE_WITH_YOUR_KEY":
        raise RuntimeError("Please paste your team API key into QTC_API_KEY at top of file (temporary), then rerun.")

    ensure_dir(OUT_DIR)
    session = make_session()

    # 1) symbols
    symbols = get_symbols(session)
    if MAX_SYMBOLS is not None:
        symbols = symbols[:MAX_SYMBOLS]
    print(f"Symbols universe: {len(symbols)}")

    # 2) find actual available trading days
    days = find_recent_available_days(session, need_days=N_TRADING_DAYS)
    if not days:
        raise RuntimeError("No available days found. Check key or lookback window.")

    print("\nTrading days selected:")
    for d in days:
        print(" ", d)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": BASE_URL,
        "symbols_count": len(symbols),
        "symbols_sample": symbols[:30],
        "days": [d.isoformat() for d in days],
        "out_dir": os.path.abspath(OUT_DIR),
        "settings": {
            "max_symbols_per_request": MAX_SYMBOLS_PER_REQUEST,
            "sleep_per_request": SLEEP_PER_REQUEST,
            "n_trading_days": N_TRADING_DAYS,
            "probe_symbol": PROBE_SYMBOL,
            "probe_lookback_days": PROBE_LOOKBACK_DAYS,
            "day_start": DAY_START,
            "day_end": DAY_END,
            "skip_if_exists": SKIP_IF_EXISTS,
        },
        "rows_saved_after_merge": {},
    }

    # Optional: show existing
    if SKIP_IF_EXISTS and os.path.exists(OUT_DIR):
        existing = [fn[:-8] for fn in os.listdir(OUT_DIR) if fn.endswith(".parquet")]
        if existing:
            print(f"Found existing parquet files: {len(existing)} (will merge/dedupe as needed).")

    # 3) fetch sequentially by day (stable)
    for d in days:
        _, rows_saved = fetch_for_day(d, symbols, OUT_DIR)
        manifest["rows_saved_after_merge"].update(rows_saved)

    with open(os.path.join(OUT_DIR, "_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\nAll done.")
    print(f"Manifest: {os.path.join(OUT_DIR, '_manifest.json')}")
    print(f"Parquets in: {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
