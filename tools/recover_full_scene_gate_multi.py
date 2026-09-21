"""Resume R02/R03 full-scene-gate trajectories concurrently with frozen feedback recovery."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools import full_scene_gate_study as study
from tools import recover_full_scene_gate as recovery


RUNS = ("R02", "R03")


def prepare(root: Path) -> None:
    for run in RUNS:
        recovery.prepare(root, run)
    status(root)


def status(root: Path, *, write: bool = False) -> list[dict]:
    for run in RUNS:
        recovery.load(root, run)
    rows = study.report(root, RUNS, write=write)
    pending = [row for row in rows if row["complete_episodes"] < 60]
    print(f"complete={len(rows) - len(pending)}/{len(rows)} pending={len(pending)} recovery={recovery.VERSION}")
    return rows


def generate(root: Path, limit: int, workers: int) -> None:
    with (root / ".tasks.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = status(root)
        pending = [row for row in rows if row["complete_episodes"] < limit]
        print(f"pending={len(pending)} full_concurrency={min(workers, len(pending))} recovery={recovery.VERSION}")

        def launch(row: dict) -> int:
            arm, run = row["arm"], row["run"]
            log = recovery.profile_dir(root, run) / f"generate_{arm}_multi.log"
            command = [
                sys.executable,
                str(REPO / "tools/recover_full_scene_gate.py"),
                "generate",
                "--root",
                str(root),
                "--run",
                run,
                "--arm",
                arm,
                "--worker",
                "--limit",
                str(limit),
                "--execute-api",
            ]
            with log.open("a", encoding="utf-8") as stream:
                result = subprocess.run(command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT)
            print(json.dumps({"arm": arm, "run": run, "returncode": result.returncode, "log": str(log)}, ensure_ascii=False), flush=True)
            return result.returncode

        with ThreadPoolExecutor(max_workers=min(workers, len(pending)) or 1) as executor:
            failures = [future.result() for future in as_completed([executor.submit(launch, row) for row in pending])]
        rows = status(root)
        if any(failures) or any(row["complete_episodes"] < limit for row in rows):
            raise SystemExit("Recovery incomplete; successful outputs retained; inspect multi recovery logs and rerun")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "status", "generate", "report"))
    parser.add_argument("--root", type=Path, default=study.ROOT)
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--execute-api", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.limit <= 60 or args.workers < 1:
        parser.error("Require limit 1..60 and positive workers")
    root = args.root.resolve()
    if args.action == "prepare":
        prepare(root)
    elif args.action == "generate" and args.execute_api:
        generate(root, args.limit, args.workers)
    else:
        status(root, write=args.action == "report")
        if args.action == "generate":
            print("Read-only; add --execute-api to call Qwen3.6")


if __name__ == "__main__":
    main()
