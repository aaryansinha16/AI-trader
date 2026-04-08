"""
Backup AI Trader's important state to one or more destinations.

What gets backed up:
  • DB tables → CSV.gz via psql COPY (NOT pg_dump — broken on hypertables)
  • models/saved/             → models/current/  (verbatim copy)
  • models/saved/*.pkl         → models/by_train_date/YYYY-MM-DD/  (mtime-bucketed)
  • backtest_results/          → backtest_results/
  • config/                    → config/

Not backed up:
  • logs/  (large, regenerable)
  • node_modules, .venv, __pycache__

Output structure:
  <dest>/
    └── 2026-04-08/
        ├── db/                       # gzipped CSVs, restorable via psql \\COPY
        │   ├── tick_data.csv.gz
        │   ├── minute_candles.csv.gz
        │   └── ...
        ├── models/
        │   ├── current/              # exact copy of models/saved/
        │   └── by_train_date/        # bucketed by .pkl mtime for easy rollback
        │       ├── 2026-04-07/
        │       │   ├── bearish_momentum_model.pkl
        │       │   └── ...
        │       └── ...
        ├── backtest_results/
        ├── config/
        └── MANIFEST.txt              # tables, sizes, file counts, git sha

Why CSV?
  • Restorable to ANY PostgreSQL — gunzip + psql \\COPY, no proprietary format
  • Inspectable by anything (pandas, Excel, DuckDB, less)
  • Plain text → ratio-friendly to gzip, diffable across snapshots

Usage:
  python scripts/backup_data.py
  python scripts/backup_data.py --rotate 30
  python scripts/backup_data.py --extra-dest ~/Dropbox/ai-trader-backups
  python scripts/backup_data.py --extra-dest ~/Dropbox/... --extra-dest /Volumes/SSD/...
  python scripts/backup_data.py --no-db
  python scripts/backup_data.py --dest /tmp/test_backup
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
from datetime import datetime, date, timedelta
from pathlib import Path

from utils.logger import get_logger

logger = get_logger("backup_data")

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DEFAULT_DEST = Path.home() / "Dev" / "Backups" / "ai-trader"

DB_TABLES = [
    "tick_data",
    "minute_candles",
    "symbol_master",
    "trade_log",
    "daily_performance",
    "second_candles",
    "features_macro",
    "features_micro",
]

FILE_DIRS = [
    ("backtest_results", "backtest_results"),
    ("config", "config"),
]
# models/saved is handled specially (current/ + by_train_date/)


def get_db_credentials() -> dict:
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": os.getenv("DB_PORT", "5432"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", ""),
        "name": os.getenv("DB_NAME", "trading"),
    }


def find_pg_binary(name: str) -> str:
    """Locate a PG client binary that matches the running server (PG17)."""
    candidates = [
        f"/opt/homebrew/opt/postgresql@17/bin/{name}",
        f"/opt/homebrew/opt/postgresql/bin/{name}",
        f"/usr/local/opt/postgresql@17/bin/{name}",
        shutil.which(name) or "",
    ]
    for c in candidates:
        if c and Path(c).exists():
            return c
    return name  # last resort


def dump_table(table: str, out_dir: Path, creds: dict) -> tuple[bool, int]:
    """
    Dump one table to gzipped CSV via psql COPY ... TO STDOUT.

    Why not pg_dump --data-only -t tablename:
      tick_data and minute_candles are TimescaleDB hypertables. pg_dump -t
      only dumps the parent table, which is empty — real data lives in
      child chunks under _timescaledb_internal. COPY reads via the parent
      view and returns all chunked rows correctly.

    Output is CSV (with header) so it's restorable via:
        gunzip -c table.csv.gz | psql ... -c "\\COPY table FROM STDIN CSV HEADER"
    """
    out = out_dir / f"{table}.csv.gz"
    env = os.environ.copy()
    env["PGPASSWORD"] = creds["password"]

    psql = find_pg_binary("psql")
    sql = f"\\COPY (SELECT * FROM {table}) TO STDOUT CSV HEADER"
    cmd = [
        psql,
        "-h", creds["host"],
        "-p", creds["port"],
        "-U", creds["user"],
        "-d", creds["name"],
        "-c", sql,
    ]
    try:
        with open(out, "wb") as f:
            ps = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
            gz = subprocess.Popen(["gzip", "-9"], stdin=ps.stdout, stdout=f, stderr=subprocess.PIPE)
            ps.stdout.close()
            _, _ = gz.communicate(timeout=1800)
            ps.wait(timeout=1800)
        if ps.returncode != 0:
            stderr = ps.stderr.read().decode("utf-8", errors="ignore") if ps.stderr else ""
            logger.error(f"COPY {table} failed: {stderr.strip()}")
            return (False, 0)
        return (True, out.stat().st_size if out.exists() else 0)
    except FileNotFoundError:
        logger.error("psql not found — install postgresql client tools")
        return (False, 0)
    except subprocess.TimeoutExpired:
        logger.error(f"COPY {table} timed out")
        return (False, 0)


def copy_dir(src_rel: str, dest_rel: str, dest_root: Path) -> tuple[int, int]:
    """Recursively copy a project directory. Returns (file_count, total_bytes)."""
    src = PROJECT_ROOT / src_rel
    dst = dest_root / dest_rel
    if not src.exists():
        return (0, 0)
    dst.mkdir(parents=True, exist_ok=True)
    n_files = 0
    n_bytes = 0
    for root, _dirs, files in os.walk(src):
        rel_root = Path(root).relative_to(src)
        target_root = dst / rel_root
        target_root.mkdir(parents=True, exist_ok=True)
        for f in files:
            if f.endswith((".pyc",)):
                continue
            sf = Path(root) / f
            tf = target_root / f
            shutil.copy2(sf, tf)
            n_files += 1
            n_bytes += sf.stat().st_size
    return (n_files, n_bytes)


def get_git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def write_manifest(
    snap_dir: Path,
    db_results: list,
    file_results: list,
    model_summary: dict,
    integrity_msg: str,
    extra_dests: list,
):
    lines = [
        "AI Trader Backup",
        "================",
        f"created_at: {datetime.now().isoformat()}",
        f"git_sha:    {get_git_sha()}",
        f"hostname:   {os.uname().nodename}",
        "",
        f"Previous snapshot integrity: {integrity_msg}",
        "",
        "Database tables (gzipped CSV via psql COPY):",
    ]
    for table, ok, sz in db_results:
        status = "✓" if ok else "✗"
        lines.append(f"  {status} {table:<25s}  {sz:>14,} bytes")
    lines.append("")
    lines.append("Files:")
    for src, dst, n, sz in file_results:
        lines.append(f"  ✓ {src:<25s}  {n:>5} files  {sz:>14,} bytes")
    lines.append("")
    lines.append("Models:")
    lines.append(f"  current/        {model_summary.get('current_files', 0):>5} files  "
                 f"{model_summary.get('current_bytes', 0):>14,} bytes")
    lines.append(f"  by_train_date/  {sum(model_summary.get('by_date', {}).values()):>5} pkl files "
                 f"bucketed across {len(model_summary.get('by_date', {}))} dates:")
    for d in sorted(model_summary.get("by_date", {}).keys()):
        lines.append(f"    {d}: {model_summary['by_date'][d]} files")
    lines.append("")
    if extra_dests:
        lines.append("Mirrored to:")
        for path, ok, sz in extra_dests:
            mark = "✓" if ok else "✗"
            lines.append(f"  {mark} {path}  ({sz:,} bytes)")
    lines.append("")
    lines.append("Restore (per table):")
    lines.append("  gunzip -c db/<table>.csv.gz | \\")
    lines.append("    psql -d trading -c \"\\COPY <table> FROM STDIN CSV HEADER\"")
    lines.append("")
    lines.append("Or use scripts/restore_from_backup.py to restore everything in order.")
    (snap_dir / "MANIFEST.txt").write_text("\n".join(lines) + "\n")


def rotate_old(dest_root: Path, keep_days: int):
    cutoff = date.today() - timedelta(days=keep_days)
    removed = 0
    for child in sorted(dest_root.iterdir()):
        if not child.is_dir():
            continue
        try:
            d = datetime.strptime(child.name, "%Y-%m-%d").date()
            if d < cutoff:
                shutil.rmtree(child)
                removed += 1
                print(f"  rotated out: {child.name}")
        except ValueError:
            continue
    if removed:
        print(f"  removed {removed} snapshot(s) older than {cutoff}")


def backup_models(dest_root: Path) -> tuple[int, int, dict[str, int]]:
    """
    Copy models/saved/ → dest_root/models/current/  (verbatim)
    Plus bucket every .pkl into dest_root/models/by_train_date/YYYY-MM-DD/
    by file mtime, so you can easily roll back to a model that was active
    on a specific date.

    Returns (n_files_total, total_bytes, {date: file_count}).
    """
    src = PROJECT_ROOT / "models" / "saved"
    if not src.exists():
        return (0, 0, {})

    # 1. Verbatim copy → models/current/
    current_dst = dest_root / "models" / "current"
    current_dst.mkdir(parents=True, exist_ok=True)
    n_files, n_bytes = 0, 0
    for root, _dirs, files in os.walk(src):
        rel_root = Path(root).relative_to(src)
        target_root = current_dst / rel_root
        target_root.mkdir(parents=True, exist_ok=True)
        for f in files:
            if f.endswith(".pyc"):
                continue
            sf = Path(root) / f
            tf = target_root / f
            shutil.copy2(sf, tf)
            n_files += 1
            n_bytes += sf.stat().st_size

    # 2. Bucket .pkl files by training date (mtime) → models/by_train_date/
    # We only bucket the *active* pkls (not the historical models/saved/backups
    # subtree, which is its own internal versioning system).
    by_date_dst = dest_root / "models" / "by_train_date"
    by_date_counts: dict[str, int] = {}

    for root, _dirs, files in os.walk(src):
        # Skip the existing backups/ subtree — its filenames already encode the date
        rel = Path(root).relative_to(src)
        if rel.parts and rel.parts[0] == "backups":
            continue
        for f in files:
            if not f.endswith(".pkl"):
                continue
            sf = Path(root) / f
            mtime = datetime.fromtimestamp(sf.stat().st_mtime).date()
            bucket_name = mtime.strftime("%Y-%m-%d")
            bucket = by_date_dst / bucket_name
            bucket.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sf, bucket / f)
            by_date_counts[bucket_name] = by_date_counts.get(bucket_name, 0) + 1

    return (n_files, n_bytes, by_date_counts)


def verify_previous_snapshot(dest_root: Path) -> tuple[bool, str]:
    """
    Walk the most recent prior snapshot and `gunzip -t` every .gz to catch
    silent corruption. Returns (all_ok, message). Skipped if no prior exists.
    """
    if not dest_root.exists():
        return (True, "no prior snapshots")
    today_name = date.today().strftime("%Y-%m-%d")
    candidates = []
    for child in dest_root.iterdir():
        if not child.is_dir() or child.name == today_name:
            continue
        try:
            d = datetime.strptime(child.name, "%Y-%m-%d").date()
            candidates.append((d, child))
        except ValueError:
            continue
    if not candidates:
        return (True, "no prior snapshots")
    candidates.sort(reverse=True)
    prev_dir = candidates[0][1]

    bad: list[str] = []
    checked = 0
    for root, _dirs, files in os.walk(prev_dir):
        for f in files:
            if not f.endswith(".gz"):
                continue
            p = Path(root) / f
            checked += 1
            try:
                r = subprocess.run(
                    ["gunzip", "-t", str(p)],
                    capture_output=True,
                    timeout=120,
                )
                if r.returncode != 0:
                    bad.append(str(p.relative_to(prev_dir)))
            except Exception as e:
                bad.append(f"{p.relative_to(prev_dir)} ({e})")

    if bad:
        return (False, f"prev snapshot {prev_dir.name}: {len(bad)}/{checked} files corrupt: {bad[:3]}")
    return (True, f"prev snapshot {prev_dir.name}: {checked} gz files OK")


def mirror_to_extra_dest(snap_dir: Path, extra_dest: Path) -> tuple[bool, int]:
    """
    Mirror today's snapshot folder to an extra destination (e.g. Dropbox).

    Skipped silently if extra_dest's parent doesn't exist (so an unmounted
    external drive doesn't fail the run). The parent must exist to enable
    the mirror — we won't auto-create arbitrary paths.
    """
    extra_dest = extra_dest.expanduser()
    parent = extra_dest.parent
    if not parent.exists():
        print(f"  ! skipping {extra_dest} — parent does not exist")
        return (False, 0)

    extra_dest.mkdir(parents=True, exist_ok=True)
    target = extra_dest / snap_dir.name

    # Use rsync if available for incremental + efficient sync; fall back to cp -R
    rsync = shutil.which("rsync")
    if rsync:
        cmd = [rsync, "-a", "--delete", str(snap_dir) + "/", str(target) + "/"]
    else:
        if target.exists():
            shutil.rmtree(target)
        cmd = ["cp", "-R", str(snap_dir), str(target)]

    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=1800)
    except subprocess.CalledProcessError as e:
        logger.error(f"mirror failed: {e.stderr.decode(errors='ignore')}")
        return (False, 0)
    except subprocess.TimeoutExpired:
        logger.error("mirror timed out")
        return (False, 0)

    # Compute size
    total = 0
    for root, _d, files in os.walk(target):
        for f in files:
            total += (Path(root) / f).stat().st_size
    return (True, total)


def main():
    parser = argparse.ArgumentParser(description="Snapshot DB tables + models + backtest results")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST,
                        help=f"Primary backup root (default: {DEFAULT_DEST})")
    parser.add_argument("--extra-dest", type=Path, action="append", default=[],
                        help="Additional destinations to mirror to. Repeatable. "
                             "E.g. --extra-dest ~/Dropbox/ai-trader-backups. "
                             "Skipped if parent does not exist (e.g. unmounted drive).")
    parser.add_argument("--no-db", action="store_true", help="Skip DB dump (files only)")
    parser.add_argument("--rotate", type=int, default=0,
                        help="Delete snapshots older than N days from PRIMARY dest")
    parser.add_argument("--skip-integrity-check", action="store_true",
                        help="Skip the gunzip -t check on the previous snapshot")
    args = parser.parse_args()

    args.dest = args.dest.expanduser()
    args.dest.mkdir(parents=True, exist_ok=True)
    snap_name = date.today().strftime("%Y-%m-%d")
    snap_dir = args.dest / snap_name

    print(f"\n{'#' * 60}")
    print(f"#  AI TRADER BACKUP")
    print(f"#  Primary dest: {snap_dir}")
    if args.extra_dest:
        for d in args.extra_dest:
            print(f"#  Extra dest:   {d.expanduser()}")
    print(f"{'#' * 60}\n")

    # ── Integrity check on previous snapshot ─────────────────────────
    if not args.skip_integrity_check:
        print("Verifying previous snapshot integrity...")
        ok, msg = verify_previous_snapshot(args.dest)
        print(f"  {'✓' if ok else '✗'} {msg}")
        integrity_msg = msg
        if not ok:
            print("\n  ! Previous snapshot has corruption — proceeding anyway, but inspect manually.")
    else:
        integrity_msg = "skipped (--skip-integrity-check)"

    snap_dir.mkdir(parents=True, exist_ok=True)

    # ── DB tables ────────────────────────────────────────────────────
    db_results: list[tuple[str, bool, int]] = []
    if not args.no_db:
        db_dir = snap_dir / "db"
        db_dir.mkdir(parents=True, exist_ok=True)
        creds = get_db_credentials()
        print("\nDB tables:")
        for tbl in DB_TABLES:
            print(f"  dumping {tbl}...", end=" ", flush=True)
            ok, sz = dump_table(tbl, db_dir, creds)
            db_results.append((tbl, ok, sz))
            print(f"{'OK' if ok else 'FAIL'}  ({sz:,} bytes)")
    else:
        print("\nDB tables: skipped (--no-db)")

    # ── File trees (backtest_results, config) ────────────────────────
    print("\nFiles:")
    file_results: list[tuple[str, str, int, int]] = []
    for src_rel, dst_rel in FILE_DIRS:
        n, sz = copy_dir(src_rel, dst_rel, snap_dir)
        file_results.append((src_rel, dst_rel, n, sz))
        print(f"  {src_rel:<25s} → {n:>5} files, {sz:,} bytes")

    # ── Models (current/ + by_train_date/) ───────────────────────────
    print("\nModels:")
    n_files, n_bytes, by_date = backup_models(snap_dir)
    model_summary = {"current_files": n_files, "current_bytes": n_bytes, "by_date": by_date}
    print(f"  current/        {n_files:>5} files, {n_bytes:,} bytes")
    print(f"  by_train_date/  {sum(by_date.values()):>5} pkl files, {len(by_date)} dates:")
    for d in sorted(by_date.keys())[-5:]:  # show last 5
        print(f"    {d}: {by_date[d]} files")
    if len(by_date) > 5:
        print(f"    ... and {len(by_date) - 5} earlier dates")

    # ── Mirror to extra destinations ─────────────────────────────────
    extra_results: list[tuple[str, bool, int]] = []
    if args.extra_dest:
        print("\nMirroring to extra destinations:")
        for ed in args.extra_dest:
            print(f"  → {ed.expanduser()}", end=" ", flush=True)
            ok, sz = mirror_to_extra_dest(snap_dir, ed)
            extra_results.append((str(ed.expanduser()), ok, sz))
            if ok:
                print(f"OK  ({sz / 1024 / 1024:.1f} MB)")

    # ── Manifest (written last so it includes everything) ────────────
    write_manifest(snap_dir, db_results, file_results, model_summary, integrity_msg, extra_results)
    print(f"\nWrote {snap_dir / 'MANIFEST.txt'}")

    # Re-mirror manifest only (so extra dests have the final manifest)
    if args.extra_dest:
        for ed in args.extra_dest:
            target = ed.expanduser() / snap_dir.name
            if target.exists():
                shutil.copy2(snap_dir / "MANIFEST.txt", target / "MANIFEST.txt")

    # ── Rotation ─────────────────────────────────────────────────────
    if args.rotate > 0:
        print(f"\nRotation (keep {args.rotate} days, primary dest only):")
        rotate_old(args.dest, args.rotate)

    # ── Summary ──────────────────────────────────────────────────────
    total_size = 0
    for root, _d, files in os.walk(snap_dir):
        for f in files:
            total_size += (Path(root) / f).stat().st_size

    print(f"\n{'=' * 60}")
    print(f"  BACKUP COMPLETE — {snap_name}")
    print(f"  Total size: {total_size / 1024 / 1024:.1f} MB")
    print(f"  Location:   {snap_dir}")
    if args.extra_dest:
        for path, ok, sz in extra_results:
            mark = "✓" if ok else "✗"
            print(f"  {mark} Mirrored:  {path}  ({sz / 1024 / 1024:.1f} MB)")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
