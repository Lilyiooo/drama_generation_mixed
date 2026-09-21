"""Resume R02/R03 with deterministic S-span to scene-ID metadata binding."""

import argparse
import asyncio
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from drama_local.runtime import atomic_json
from service.drama_by_creativity import scene_state_gate as original
from service.drama_by_creativity.conservative_scene_gate import ConservativeSceneGate, decision
from tools.continue_scene_gate_decision import compatible_validate
from tools.repair_scene_gate_calibration import evidence_catalog
from tools import full_scene_gate_study as study


VERSION = "metadata_binding_v3"
RUNS = ("R02", "R03")
RULES = """元数据绑定要求：checks.scene_id 必须逐字使用 scene_evidence_catalog 中相应 S 编号的 scene_id。
同一条 check 的所有 scene_span_ids 必须属于同一个场次。不得使用简写、去掉括号说明或把其他场次的 S 编号挂到当前场次。
这只约束证据元数据格式，不改变原始状态判断规则；不要为通过格式校验修改语义结论。"""


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dependencies() -> dict[str, str]:
    return {
        str(path): file_hash(path)
        for path in (Path(__file__), REPO / "recover_full_scene_gate_binding.sh")
    }


def normalize_metadata(raw: str, scenes: list[dict], catalog: dict) -> tuple[dict, str, list[dict]]:
    value = json.loads(raw.strip().removeprefix("```json").removesuffix("```").strip())
    if not isinstance(value, dict):
        raise ValueError("Require a JSON object")
    normalized = copy.deepcopy(value)
    spans = evidence_catalog(scenes)
    changes: list[dict] = []
    checks = normalized.get("checks")
    if not isinstance(checks, list):
        raise ValueError("checks must be a list")
    for index, check in enumerate(checks):
        if not isinstance(check, dict):
            raise ValueError(f"checks[{index}] must be an object")
        selected = check.get("scene_span_ids")
        if not isinstance(selected, list) or not selected:
            raise ValueError(f"checks[{index}].scene_span_ids must be a nonempty list")
        if any(not isinstance(span, str) or span not in spans for span in selected):
            raise ValueError(f"checks[{index}] cites an unknown scene span")
        actual = {spans[span]["scene_id"] for span in selected}
        if len(actual) != 1:
            raise ValueError(f"checks[{index}] mixes evidence from multiple scenes")
        actual_scene = next(iter(actual))
        reported_scene = check.get("scene_id")
        if reported_scene != actual_scene:
            changes.append({
                "check_index": index,
                "reported_scene_id": reported_scene,
                "evidence_bound_scene_id": actual_scene,
                "scene_span_ids": selected,
            })
            check["scene_id"] = actual_scene
    normalized_raw = json.dumps(normalized, ensure_ascii=False)
    result = compatible_validate(normalized_raw, scenes, catalog)
    result["metadata_binding_compatibility"] = changes
    return result, normalized_raw, changes


class BindingGate(ConservativeSceneGate):
    async def check(self, ctx, scenes, slot="check_before"):
        packet = {**self.packet(scenes), "scene_evidence_catalog": evidence_catalog(scenes)}
        user = json.dumps(packet, ensure_ascii=False) + "\n" + RULES
        recovered_slot = f"{slot}_{VERSION}"
        cache_path = self.directory / f"{recovered_slot}.json"
        source_path = self.directory / f"{recovered_slot}_source.json"

        def validator(raw: str) -> dict:
            result, _, _ = normalize_metadata(raw, scenes, self.catalog)
            return result

        if not cache_path.exists():
            patterns = (
                self.directory / "attempts" / f"{slot}_check_feedback_v1",
                self.directory / "attempts" / slot,
            )
            attempts = sorted(
                (path for root in patterns for path in root.glob("*/*.json")),
                key=lambda path: (path.stat().st_mtime_ns, str(path)),
                reverse=True,
            )
            for path in attempts:
                record = original.read(path)
                raw = record.get("raw")
                if not isinstance(raw, str):
                    continue
                try:
                    result, _, changes = normalize_metadata(raw, scenes, self.catalog)
                except (ValueError, TypeError, KeyError, json.JSONDecodeError):
                    continue
                if not changes:
                    continue
                payload = {
                    "model": "Qwen3.6-27B",
                    "base_url": self.base_url,
                    "system": decision.RULES,
                    "user": user,
                    "temperature": 0,
                    "max_tokens": 4096,
                    "binding": self.binding,
                }
                atomic_json(cache_path, {
                    "fingerprint": original.digest(payload),
                    "request": payload,
                    "raw": raw,
                    "result": result,
                })
                atomic_json(source_path, {
                    "version": VERSION,
                    "source_attempt": str(path),
                    "source_attempt_sha256": file_hash(path),
                    "normalization": changes,
                    "policy": "Only scene_id metadata is rebound from cited S spans; semantic fields are unchanged.",
                })
                break

        return await self.request(
            ctx,
            recovered_slot,
            decision.RULES,
            user,
            validator,
        )

    async def apply(self, ctx, scenes):
        selected = await super().apply(ctx, scenes)
        path = self.directory / "outcome.json"
        outcome = original.read(path)
        outcome["recovery_protocol"] = VERSION
        outcome["recovery_implementation"] = dependencies()
        atomic_json(path, outcome)
        return selected


def profile_dir(root: Path, run: str) -> Path:
    return root / "recovery" / VERSION / run


def prepare_one(root: Path, run: str) -> None:
    directory = profile_dir(root, run)
    if (directory / "manifest.json").exists():
        load_one(root, run)
        return
    manifest, bundle = study.load(root)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(f"Recovery directory is nonempty without a manifest: {directory}")
    rows = study.report(root, (run,))
    protected = [root / "manifest.json", root / "frozen_inputs.json"]
    for arm in study.ARMS:
        trajectory = study.verify_directory(root, arm, run, manifest, bundle)
        for folder in ("05_drama", "state_lifecycle", "scene_state_gate"):
            protected.extend(path for path in (trajectory / folder).rglob("*.json") if "failures" not in path.parts)
        native = trajectory / "narrative_memory"
        for path in native.rglob("*.json"):
            atomic_json(directory / "before" / arm / path.relative_to(trajectory), original.read(path))
        protected.extend([trajectory / "arm_binding.json", trajectory / "generation_assets.json"])
    atomic_json(directory / "manifest.json", {
        "version": VERSION,
        "run": run,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "implementation": dependencies(),
        "original_manifest_sha256": file_hash(root / "manifest.json"),
        "retained_files": {str(path.relative_to(root)): file_hash(path) for path in sorted(set(protected))},
        "starting_progress": rows,
        "scope": "Resume with deterministic scene_id binding from cited S spans. No script, memory, state refs, classifications, reasons, or prior outputs are rewritten.",
    })


def load_one(root: Path, run: str) -> dict:
    study.load(root)
    profile = original.read(profile_dir(root, run) / "manifest.json")
    if profile["version"] != VERSION or profile["run"] != run or profile["implementation"] != dependencies():
        raise ValueError("Binding recovery code or run changed")
    for name, expected in profile["retained_files"].items():
        if file_hash(root / name) != expected:
            raise ValueError(f"Retained artifact changed: {name}")
    return profile


def prepare(root: Path) -> None:
    with (root / ".tasks.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for run in RUNS:
            prepare_one(root, run)
    status(root)


def status(root: Path) -> list[dict]:
    for run in RUNS:
        load_one(root, run)
    rows = study.report(root, RUNS)
    pending = [row for row in rows if row["complete_episodes"] < 60]
    print(f"complete={len(rows) - len(pending)}/{len(rows)} pending={len(pending)} recovery={VERSION}")
    return rows


def worker(root: Path, arm: str, run: str, limit: int) -> None:
    load_one(root, run)
    manifest, bundle = study.load(root)
    directory = study.verify_directory(root, arm, run, manifest, bundle)
    with (directory / ".trajectory.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.environ["DRAMA_SCENE_STATE_GATE"] = "off"
        factory = BindingGate.factory if arm == "candidate_gate" else None
        asyncio.run(study.study.generate_directory(
            bundle["inputs"], directory, "hybrid", study.identifier(arm, run), limit,
            scene_gate_factory=factory,
        ))
    load_one(root, run)


def generate(root: Path, limit: int, workers: int) -> None:
    with (root / ".tasks.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = status(root)
        pending = [row for row in rows if row["complete_episodes"] < limit]
        print(f"pending={len(pending)} full_concurrency={min(workers, len(pending))} recovery={VERSION}", flush=True)

        def launch(row: dict) -> int:
            arm, run = row["arm"], row["run"]
            log = profile_dir(root, run) / f"generate_{arm}.log"
            command = [
                sys.executable, str(Path(__file__)), "generate", "--root", str(root),
                "--run", run, "--arm", arm, "--worker", "--limit", str(limit), "--execute-api",
            ]
            with log.open("a", encoding="utf-8") as stream:
                result = subprocess.run(command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT)
            print(json.dumps({"arm": arm, "run": run, "returncode": result.returncode, "log": str(log)}, ensure_ascii=False), flush=True)
            return result.returncode

        with ThreadPoolExecutor(max_workers=min(workers, len(pending)) or 1) as executor:
            failures = [future.result() for future in as_completed([executor.submit(launch, row) for row in pending])]
        rows = status(root)
        if any(failures) or any(row["complete_episodes"] < limit for row in rows):
            raise SystemExit("Binding recovery incomplete; successful outputs retained; inspect logs and rerun")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "status", "generate", "report"))
    parser.add_argument("--root", type=Path, default=study.ROOT)
    parser.add_argument("--run", choices=RUNS)
    parser.add_argument("--arm", choices=study.ARMS)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--execute-api", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.limit <= 60 or args.workers < 1:
        parser.error("Require limit 1..60 and positive workers")
    root = args.root.resolve()
    if args.action == "prepare":
        prepare(root)
    elif args.worker:
        if args.action != "generate" or not args.execute_api or not args.run or not args.arm:
            parser.error("Worker requires generation, run, arm and API permission")
        worker(root, args.arm, args.run, args.limit)
    elif args.action == "generate" and args.execute_api:
        generate(root, args.limit, args.workers)
    else:
        rows = status(root)
        if args.action == "report":
            atomic_json(root / "reports" / f"{VERSION}.json", {"rows": rows, "recovery": VERSION})
        elif args.action == "generate":
            print("Read-only; add --execute-api to call Qwen3.6")


if __name__ == "__main__":
    main()
