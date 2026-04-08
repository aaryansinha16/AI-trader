"""
Restore an AI Trader backup snapshot onto a fresh machine.

What it does:
  1. Validates the snapshot folder structure
  2. Initializes the DB schema (idempotent — re-running is safe)
  3. Restores every CSV.gz table via psql \\COPY in dependency order
  4. Copies models/current/ → project's models/saved/
  5. Copies backtest_results/ and config/

What you must do FIRST on the new machine:
  1. Install the project (git clone, .venv, pip install -r requirements.txt)
  2. Install + start TimescaleDB:  brew install postgresql@17 timescaledb
  3. Create the empty database:    createdb trading
  4. Set TRUEDATA_USER, TRUEDATA_PASSWORD, DB_* in .env
  5. Then run THIS script.

Usage:
  python scripts/restore_from_backup.py ~/Dropbox/ai-trader-backups/2026-04-08
  python scripts/restore_from_backup.py ~/Dropbox/ai-trader-backups/2026-04-08 --tables-only
  python scripts/restore_from_backup.py ~/Dropbox/ai-trader-backups/2026-04-08 --dry-run
  python scripts/restore_from_backup.py latest --backup-root ~/Dropbox/ai-trader-backups
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import argparse
import shutil
import subprocess
from datetime import datetime, date
from pathlib import Path

from utils.logger import get_logger

logger = get_logger("restore_backup")

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Restore tables in this order. Reference tables first, then bulk data.
# tick_data and minute_candles are last (largest, slowest).
RESTORE_ORDER = [
    "symbol_master",
    "trade_log",
    "daily_performance",
    "second_candles",
    "features_micro",
    "features_macro",
    "minute_candles",
    "tick_data",
]


def find_pg_binary(name: str) -> str:
    candidates = [
        f"/opt/homebrew/opt/postgresql@17/bin/{name}",
        f"/opt/homebrew/opt/postgresql/bin/{name}",
        f"/usr/local/opt/postgresql@17/bin/{name}",
        shutil.which(name) or "",
    ]
    for c in candidates:
        if c and Path(c).exists():
            return c
    return name


def get_db_credentials() -> dict:
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": os.getenv("DB_PORT", "5432"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", ""),
        "name": os.getenv("DB_NAME", "trading"),
    }


def resolve_snapshot(snap_arg: str, backup_root: Path | None) -> Path:
    """`latest` resolves to the most recent YYYY-MM-DD subfolder of backup_root."""
    if snap_arg != "latest":
        return Path(snap_arg).expanduser()
    if not backup_root:
        raise ValueError("--backup-root is required when snap is 'latest'")
    backup_root = backup_root.expanduser()
    candidates: list[tuple[date, Path]] = []
    for child in backup_root.iterdir():
        if not child.is_dir():
            continue
        try:
            d = datetime.strptime(child.name, "%Y-%m-%d").date()
            candidates.append((d, child))
        except ValueError:
            continue
    if not candidates:
        raise FileNotFoundError(f"No date-named snapshots found in {backup_root}")
    candidates.sort(reverse=True)
    return candidates[0][1]


def validate_snapshot(snap: Path) -> tuple[bool, list[str]]:
    issues: list[str] = []
    if not snap.exists():
        issues.append(f"snapshot folder does not exist: {snap}")
        return (False, issues)
    if not (snap / "MANIFEST.txt").exists():
        issues.append("MANIFEST.txt missing — is this a valid snapshot?")
    if not (snap / "db").exists():
        issues.append("db/ subfolder missing")
    return (len(issues) == 0, issues)


def init_schema():
    """Run schema.sql via the project's init_db()."""
    print("\n  Initializing schema (idempotent)...")
    from database.db import init_db
    init_db()
    print("  ✓ schema initialized")


def restore_table(table: str, csv_gz: Path, creds: dict, dry_run: bool) -> tuple[bool, int]:
    """gunzip the CSV and pipe into psql \\COPY ... FROM STDIN."""
    if not csv_gz.exists():
        return (False, 0)

    if dry_run:
        return (True, csv_gz.stat().st_size)

    psql = find_pg_binary("psql")
    env = os.environ.copy()
    env["PGPASSWORD"] = creds["password"]

    sql = f"\\COPY {table} FROM STDIN CSV HEADER"
    cmd = [
        psql,
        "-h", creds["host"],
        "-p", creds["port"],
        "-U", creds["user"],
        "-d", creds["name"],
        "-c", sql,
    ]
    try:
        # gunzip → psql via pipe
        gz = subprocess.Popen(["gunzip", "-c", str(csv_gz)], stdout=subprocess.PIPE)
        ps = subprocess.Popen(cmd, stdin=gz.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        gz.stdout.close()
        out, err = ps.communicate(timeout=3600)
        if ps.returncode != 0:
            logger.error(f"restore {table} failed: {err.decode(errors='ignore')}")
            return (False, 0)
        # Parse "COPY <n>" from stdout
        out_str = out.decode(errors="ignore").strip()
        n = 0
        if out_str.startswith("COPY"):
            try:
                n = int(out_str.split()[-1])
            except ValueError:
                pass
        return (True, n)
    except Exception as e:
        logger.error(f"restore {table} exception: {e}")
        return (False, 0)


def restore_models(snap: Path, dry_run: bool) -> int:
    """Copy models/current/ → project's models/saved/, leaving any backups intact."""
    src = snap / "models" / "current"
    if not src.exists():
        print("  ! no models/current/ in snapshot")
        return 0

    dst = PROJECT_ROOT / "models" / "saved"
    if dry_run:
        print(f"  [dry-run] would copy {src} → {dst}")
        return 0

    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for root, _dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        target_root = dst / rel
        target_root.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(Path(root) / f, target_root / f)
            n += 1
    return n


def restore_files(snap: Path, dry_run: bool) -> dict:
    """Copy backtest_results/ and config/ back into the project."""
    out = {}
    for src_rel in ["backtest_results", "config"]:
        src = snap / src_rel
        if not src.exists():
            out[src_rel] = 0
            continue
        dst = PROJECT_ROOT / src_rel
        if dry_run:
            print(f"  [dry-run] would copy {src} → {dst}")
            out[src_rel] = -1
            continue
        dst.mkdir(parents=True, exist_ok=True)
        n = 0
        for root, _dirs, files in os.walk(src):
            rel = Path(root).relative_to(src)
            target_root = dst / rel
            target_root.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(Path(root) / f, target_root / f)
                n += 1
        out[src_rel] = n
    return out


def main():
    parser = argparse.ArgumentParser(description="Restore an AI Trader backup snapshot")
    parser.add_argument("snap", help="Path to a snapshot folder, or 'latest'")
    parser.add_argument("--backup-root", type=Path, help="Required if snap='latest'")
    parser.add_argument("--tables-only", action="store_true",
                        help="Restore DB tables only — skip models, configs, results")
    parser.add_argument("--models-only", action="store_true",
                        help="Restore models only — skip everything else")
    parser.add_argument("--no-schema-init", action="store_true",
                        help="Skip schema init (use if your DB schema is already set up)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen")
    args = parser.parse_args()

    snap = resolve_snapshot(args.snap, args.backup_root)
    print(f"\n{'#' * 60}")
    print(f"#  AI TRADER RESTORE")
    print(f"#  Snapshot: {snap}")
    print(f"#  Dry-run:  {args.dry_run}")
    print(f"{'#' * 60}\n")

    ok, issues = validate_snapshot(snap)
    if not ok:
        for i in issues:
            print(f"  ! {i}")
        return 1

    print("Manifest preview:")
    manifest = (snap / "MANIFEST.txt").read_text()
    for ln in manifest.split("\n")[:18]:
        print(f"  {ln}")
    print("  ...")

    if not args.dry_run:
        try:
            ans = input("\nProceed with restore? This will overwrite current state. [y/N] ").strip().lower()
        except EOFError:
            ans = "n"
        if ans != "y":
            print("Aborted.")
            return 0

    # ── Schema init ──────────────────────────────────────────────────
    if not args.models_only and not args.no_schema_init:
        init_schema()

    # ── DB tables ────────────────────────────────────────────────────
    if not args.models_only:
        creds = get_db_credentials()
        db_dir = snap / "db"
        print("\nRestoring DB tables (in order):")
        for table in RESTORE_ORDER:
            csv_gz = db_dir / f"{table}.csv.gz"
            if not csv_gz.exists():
                print(f"  - {table}: no file in snapshot, skipping")
                continue
            print(f"  - {table}...", end=" ", flush=True)
            ok, n = restore_table(table, csv_gz, creds, args.dry_run)
            if args.dry_run:
                print(f"would restore ({csv_gz.stat().st_size / 1024:.0f} KB gzipped)")
            else:
                print(f"{'OK' if ok else 'FAIL'}  ({n:,} rows)")

    # ── Models ───────────────────────────────────────────────────────
    if not args.tables_only:
        print("\nRestoring models...")
        n = restore_models(snap, args.dry_run)
        print(f"  {n} model files copied to models/saved/")

    # ── Files ────────────────────────────────────────────────────────
    if not args.tables_only and not args.models_only:
        print("\nRestoring files...")
        results = restore_files(snap, args.dry_run)
        for k, v in results.items():
            print(f"  {k}: {v} files")

    print(f"\n{'=' * 60}")
    print(f"  RESTORE COMPLETE")
    print(f"{'=' * 60}")
    print("\nNext steps:")
    print("  1. Verify with:  python -c \"from database.db import read_sql; "
          "print(read_sql('SELECT COUNT(*) FROM tick_data'))\"")
    print("  2. Start backend: python backend/app.py")
    print("  3. Start frontend: cd dashboard && npm run dev")
    return 0


if __name__ == "__main__":
    sys.exit(main())
