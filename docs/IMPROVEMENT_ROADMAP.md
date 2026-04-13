# AI Trader — Improvement Roadmap

> **Status:** Active. Tier 1 substantially completed 2026-04-08 after
> data-integrity corrections + entry-gate ports + tick-precise backtest
> rewrite + option-premium confirmation gate. Current best:
> **MEDIUM ₹+53,715 / HIGH ₹+62,762** over 18 tick-replay days with max
> DD ≤ ₹5.5k.
>
> **Last backtest:** 2026-04-08 (honest tick-level, ₹+53,715 MEDIUM,
> 71% WR, R:R 1.37, max DD ₹5,411 on ₹50k capital).

The current system has positive expectancy on tick-level replay but there
are several known levers that can lift win rate, reduce drawdowns, or open
up more tradeable setups. Items below are ordered by **expected ROI per
hour of work**, not by impressiveness.

---

## Priority Tier 1 — High value, low effort

These don't need new training data; they're code changes against the
existing pipeline. Do these first.

| # | Item | Effort | Estimated impact (MEDIUM profile) | Status |
|---|---|---|---|---|
| 1 | **Wire `micro_model` into scoring** (70% macro + 30% micro blend) — model is trained but was unused at inference | 1 hour | +0–8% WR (free win, validate via A/B backtest) | ✅ **Done 2026-04-07** |
| 2 | **Time-of-day no-entry window** 13:30–14:15 — institutional decision window where most current SL hits cluster | 30 min | -₹2k drawdown, +3% WR | ⬜ Open |
| 3 | **VIX gate** — reduce lots 50% above India VIX > 16; only `mean_reversion` fires below VIX < 11 | 2 hours | +10% on dollar return | ⬜ Open |
| 4 | **Pre-event blackout calendar** — static `events.json` with FOMC / CPI / RBI / Budget dates; system holds new entries on those days | 2 hours | Prevents 2–3 catastrophic losses per quarter | ⬜ Open |

### Tier 1a — Emergent items completed 2026-04-08

Unplanned but high-impact fixes that came out of investigating a ₹-7,031
live loss on 2026-04-08. These were treated as Tier 1 because they fixed
real correctness bugs, not just performance tuning.

| # | Item | Impact | Status |
|---|---|---|---|
| 1a.1 | **NIFTY-I dual-stream bug** — collector was subscribing to NIFTY 50 spot AND NIFTY-I futures and storing both as `symbol='NIFTY-I'`. Ticks alternated between the two (~50pt gap), corrupting every macro feature (RSI/ATR/Bollinger) and silently ruining both training data and live signals. Fixed by dropping the spot subscription in `collect_ticks.py`. | Eliminated systemic signal corruption | ✅ Done 2026-04-08 |
| 1a.2 | **Expired-options subscription bug** — `get_nearest_expiry()` fell back to `expiries[-1]` from the DB cache, which returned *yesterday's* expiry (already-dead contracts) when called on the next trading day. Live collector was subscribing to dead weeklies every morning. Fixed by fetching the authoritative expiry list from TrueData REST (`getSymbolExpiryList`) with a 1h cache. | No more dead contracts; holiday-shifted weeks handled automatically | ✅ Done 2026-04-08 |
| 1a.3 | **Tick backfill on startup and during session** — backend had candle backfill but no equivalent for ticks. Session gaps from WebSocket drops persisted until EOD. New `_detect_tick_gaps_today()` + `_backfill_ticks_if_stale()` fill leading/internal/trailing gaps via `getticks` REST every scan cycle. | Closes the 5-day REST window gaps in near-real-time | ✅ Done 2026-04-08 |
| 1a.4 | **Entry gates ported from backtest to live** — `backend/app.py:scan_market()` had none of the filter gates the backtest enforced. Ported previous-bar direction confirmation + micro-momentum confirmation + option-premium confirmation, all with strategy-aware logic (continuation vs reversal). | Closes the backtest-vs-live correctness gap | ✅ Done 2026-04-08 |
| 1a.5 | **Option-premium confirmation gate (aka "Fix D")** — new gate reads the last 30 seconds of tick_data for the specific option contract we're about to buy and rejects if premium has been falling >0.8%. Catches "catching a falling knife" entries (exact pattern of 2026-04-08 ₹-7,031 loss). Time-windowed, not tick-count-windowed, so illiquid strikes fail-open. | +₹21,800 MEDIUM vs broken-gate iteration; +₹2,467 vs no-gate baseline | ✅ Done 2026-04-08 |
| 1a.6 | **Tick-precise backtest exit loop** — `check_exit()` was walking option *minute candles*, not option ticks. Every SL/target/trailing decision was bar-resolution. Rewrote to walk per-tick within each minute when tick data is available; candle-mode fallback for older days. Journey display stays minute-summarized but the exit decisions are now tick-precise. | Backtest matches live reality; no more minute-bar fiction | ✅ Done 2026-04-08 |
| 1a.7 | **Three-tier intra-trade regime exit** — old code only tightened SL above 0.30% adverse NIFTY move (too loose and too late). Now has three tiers: 0.10% log-warn, 0.20% SL tighten to entry+2%, 0.30% HARD EXIT at current price if underwater. Interval dropped 30s → 10s. | Would have exited 2026-04-08 ₹-7k trade ~20 min earlier with a ₹~5k smaller loss | ✅ Done 2026-04-08 |
| 1a.8 | **Strategy-aware gate application** — previous-bar and micro-momentum gates both apply to continuation strategies only (`vwap_momentum_breakout`). Reversal strategies (`bearish_momentum`, `mean_reversion`) skip them because a reversal signal *should* fire against recent momentum by design. | Recovered 8 valid reversal trades that were being wrongly filtered | ✅ Done 2026-04-08 |

**Net backtest impact of Tier 1a:** MEDIUM went from ₹+51,248 (previous
best, without any of these fixes) to ₹+53,715 (+5%) with healthier
strategy mix and tighter drawdown. HIGH jumped from ₹+56,778 to ₹+62,762
(+10.5%). Crucially, the system now catches the *exact* live failure
mode from 2026-04-08 that prompted these fixes.

---

## Priority Tier 2 — Higher effort, meaningful payoff

| # | Item | Effort | Estimated impact | Status |
|---|---|---|---|---|
| 5 | **Multi-symbol expansion** — add BANKNIFTY (and possibly FINNIFTY) as a second underlying. Same pipeline, ~2× signal count. | 2 days | 1.5–2× total trade count if strategies generalise | 🟡 **Data collection done** (Plan D, 2026-04-07). Modelling deferred until NIFTY system shows consistent live profit. See "Symbol Budget & Plan D" below. |
| 6 | **Gemma 3 / local LLM news sentiment** → populate the existing-but-empty `news_boost` field in scoring | 1 week | +2–3% WR on news-driven days | ⬜ Open |
| 7 | **Order-flow microstructure features (tier 2)** — top-of-book changes, volume-weighted bid/ask delta, level-2 imbalance (needs deeper TrueData feed) | 1–2 weeks | +5% WR (estimate; very signal-dependent) | ⬜ Open |

---

## Priority Tier 3 — Data-hungry, defer until enough samples

These need more training data than we currently have (~50 outcome-labelled
trades). Don't build them now — they'll just overfit to noise.

| # | Item | Effort | Data needed | Estimated impact |
|---|---|---|---|---|
| 8 | **Day-of-week strategy models** — separate `bearish_momentum_TUE.pkl`, `_WED.pkl`, etc. Captures expiry-day theta patterns and new-contract-day volatility. | 1 day code, **6 weeks data wait** | ~30 trades per weekday (≈10 weeks of trading) | +5–10% WR |
| 9 | **Sequence model (GRU / TCN)** over last 30 candles, replacing the per-bar XGBoost. Learns temporal patterns the current model can't. | 2 weeks code, **3 months data wait** | 600+ outcome-labelled trades | +3–5% AUC, +5% WR |
| 10 | **RL entry agent** — currently RL only optimises exits. An entry-side RL policy could learn signal-skip rules no fixed gate captures. | 2–3 weeks | 200+ outcome-labelled trades | Highly variable; could be transformative or marginal |

---

## What NOT to build (and why)

- **LLM as price predictor** — LLMs (Gemma, Llama, Claude) have no native
  time-series reasoning. XGBoost is *strictly better* at structured numeric
  prediction. Use LLMs for unstructured inputs only (news, journaling).
- **Yet another candle-based feature** — diminishing returns. The macro
  feature set is already 50 columns and the model's CV AUC plateau is the
  *data*, not the model capacity.
- **Looser thresholds to "trade more"** — we tried this on 2026-04-07 and
  it produced 10 trades / 0 wins / ₹-8,619. The strict 0.70 floor is an
  accidental safety filter; keep it.

---

## Realistic 6-month outlook

**Current (2026-04-08, post-Tier-1a, tick-precise):**
- MEDIUM: ₹+53,715 / 49 trades / 71% WR / R:R 1.37 / max DD ₹-5,411
- HIGH:   ₹+62,762 / 52 trades / 77% WR / R:R 1.12 / max DD ₹-4,954
- ~₹3,000/day average on MEDIUM across 18 trading days

Per-day average is ~15% higher than last week's baseline and the system
now has real intra-trade reversal protection. Ship this version to live,
monitor for 5 consecutive profitable days, then move to Tier 1 items
2/3/4 (time-of-day, VIX, event blackout).

**After Tier 1 #2–4 complete:** ₹+3.5–4k/day, WR edging to 75%, max DD ₹3k
**After Tier 2 (#5 BANKNIFTY + #6 news):** ₹+5k/day if BANKNIFTY generalises

Anything beyond ~₹5k/day on this capital stack almost certainly requires
broker integration with real fills + slippage feedback. Simulation alone
cannot get us there.

---

## Symbol Budget & Plan D — Multi-Symbol Data Collection

**Constraint:** Current TrueData subscription = 50 live websocket symbols.
NIFTY alone uses 16 (spot + futures + ATM±3 = 14 options), and the dynamic
ATM-walk re-subscription on trending days can add 14–28 more.

### Considered plans

| Plan | Live slots | Tick fidelity for new symbols | Verdict |
|---|---|---|---|
| A. Futures+spot only for new symbols | 20 | None — REST only | Safe but no live ticks |
| B. ATM±2 options for new symbols | 36 baseline / 70 worst | Partial | Overflows on trending days |
| C. ATM±3 options for new symbols | 48 baseline / 76 worst | Full | Overflows even on calm days |
| **D. ZERO live slots for new symbols, REST EOD only** | **NIFTY's 16, unchanged** | Full via historical REST | ✅ **Chosen** |

### Plan D — what was implemented (2026-04-07)

1. **`scripts/seed_other_symbols.py`** — one-shot 6-month historical seed
   - 6mo of 1-min candles for spot + futures of BANKNIFTY, FINNIFTY
   - Last 5 trading days of option ticks + 1-min candles for ATM±3 strikes
   - For each historical day, ATM is computed from THAT day's futures close
     (not today's), so the option strikes always reflect the actual
     at-the-money window for that session.

2. **`scripts/eod_collect_other_symbols.py`** — daily idempotent EOD runner
   - Run after market close every weekday
   - Detects gaps in last N days for spot + futures candles, futures ticks,
     and per-strike option candles + ticks
   - Only fetches what's missing (idempotent, safe to re-run)
   - `--dry-run` reports gaps without fetching
   - `--also-nifty` adds a safety backfill for NIFTY-I gaps

3. **`scripts/backup_data.py`** — daily snapshot to `~/Dev/Backups/ai-trader/` + Dropbox
   - DB tables via `psql COPY ... TO STDOUT CSV HEADER` (NOT pg_dump -t, which
     is broken on TimescaleDB hypertables — see comment in the script). The
     CSVs are restorable to ANY PostgreSQL via plain `\COPY ... FROM STDIN`.
   - Models stored two ways:
     - `models/current/` → verbatim copy of `models/saved/`
     - `models/by_train_date/YYYY-MM-DD/` → mtime-bucketed `.pkl` files for
       easy "what model was active on date X" rollback
   - `backtest_results/`, `config/` copied verbatim
   - **Integrity verification**: before each new snapshot, runs `gunzip -t`
     on every `.gz` in the previous snapshot to catch silent corruption
   - **Off-machine mirror** via `--extra-dest` (repeatable): rsync's the
     local snapshot to additional destinations like `~/Dropbox/...`. Skipped
     silently if the parent doesn't exist (so an unmounted external drive
     doesn't fail the run).
   - `--rotate N` to delete snapshots older than N days from the primary dest
   - Baseline snapshot size: ~148 MB (tick_data + minute_candles + features_macro
     + models)

4. **`scripts/restore_from_backup.py`** — single-command restore on a fresh
   machine. Validates the snapshot, runs schema init (idempotent), restores
   each table via `gunzip → psql \COPY` in dependency order, then copies
   models and supporting files back into the project. Supports
   `--dry-run`, `--tables-only`, `--models-only`, and `--no-schema-init`.
   `latest` resolves to the most recent date-named subfolder of `--backup-root`.

### Notes from the seed

- **BANKNIFTY data is healthy** — 32 option contracts × 5 days, 17,004 candles
  and 179,944 ticks with 99.99% bid/ask coverage. Comparable to NIFTY in quality.
- **FINNIFTY data is much thinner** — 14 contracts, 409 candles, 770 ticks.
  Many OTM strikes have <10 ticks/day. Liquidity is genuinely poor and this
  should be considered when (eventually) training models on FINNIFTY.
- **NIFTY weekly** is the only weekly expiry; **BANKNIFTY and FINNIFTY are
  monthly only** (NSE discontinued BANKNIFTY weekly options in late 2024).

### When to upgrade to a wider TrueData plan

The "100+ symbol plan" is the right move if/when:
- NIFTY model proves profitable in live for ≥4 consecutive weeks
- You're ready to train BANKNIFTY/FINNIFTY signal models
- You want live tick microstructure features (not just historical) for the
  new symbols

Until then, Plan D's REST-only collection is sufficient — the historical
data accumulates daily and is restored next session by EOD runs.

### Recommended cron / launchd schedule

```bash
# Every weekday at 16:00 IST (after market close)
0 16 * * 1-5  cd /Users/aaryansinha/Dev/Projects/ai-trader && \
              .venv/bin/python scripts/eod_collect_other_symbols.py --include-today

# Every weekday at 16:30 IST — daily backup with 30-day rotation + Dropbox mirror
30 16 * * 1-5 cd /Users/aaryansinha/Dev/Projects/ai-trader && \
              .venv/bin/python scripts/backup_data.py \
                --rotate 30 \
                --extra-dest ~/Dropbox/ai-trader-backups
```

### Restoring on a fresh machine

```bash
# 1. Set up project + DB
git clone <repo> ai-trader && cd ai-trader
python -m venv .venv && .venv/bin/pip install -r requirements.txt
brew install postgresql@17 timescaledb && brew services start postgresql@17
createdb trading

# 2. Set DB_*, TRUEDATA_USER, TRUEDATA_PASSWORD in .env

# 3. Pull a snapshot from Dropbox (any machine logged into the same Dropbox
#    account already has a synced copy at ~/Dropbox/ai-trader-backups/)

# 4. Restore — `latest` picks the most recent date folder
.venv/bin/python scripts/restore_from_backup.py latest \
    --backup-root ~/Dropbox/ai-trader-backups
```

The restore script:
- Initializes the schema (idempotent)
- Restores tables in dependency order (`symbol_master` → bulk data last)
- Copies `models/current/` → `models/saved/`
- Copies `backtest_results/` and `config/` back into the project
- Prompts before overwriting (unless `--dry-run`)

---

## How to use this file

When picking up new work, check Tier 1 first. If a Tier 1 item is blocked,
move to Tier 2. **Do not start Tier 3 items until the data prerequisite is
met** — premature training on small samples produces models that are worse
than no model at all (we've already hit this with `bearish_momentum_model.pkl`).

Update the **Status** column when work is done. Add new ideas at the bottom
of the relevant tier; re-rank when the priorities shift.
