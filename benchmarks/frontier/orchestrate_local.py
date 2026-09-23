"""Laptop-side orchestrator for the frontier run inside a Blaxel sandbox.

Dev tooling (needs `pip install blaxel` + BL_API_KEY/BL_WORKSPACE env):
    python benchmarks/frontier/orchestrate_local.py setup      # create sandbox + upload + install deps
    python benchmarks/frontier/orchestrate_local.py smoke      # 5-query validation run (blocking)
    python benchmarks/frontier/orchestrate_local.py full       # launch full run (background process)
    python benchmarks/frontier/orchestrate_local.py poll       # print progress
    python benchmarks/frontier/orchestrate_local.py collect    # download results to benchmarks/results/frontier
    python benchmarks/frontier/orchestrate_local.py teardown   # delete sandbox

Sandbox layout: /workspace holds src/, benchmarks/, pyproject.toml; results
accumulate under /workspace/results/frontier (resumable JSONL per dataset x
system), so a re-run of any phase never repeats finished queries.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SANDBOX_NAME = "jev-frontier-20260923"
REMOTE_DIR = "/workspace"
KEY_NAMES = ("TYPESAFE_API_KEY", "VOYAGE_API_KEY", "COHERE_API_KEY")

REQUIREMENTS = """typesafe-sdk>=0.5.7
pydantic>=2.0
requests>=2.31
bm25s>=0.2
numpy>=1.26
pyarrow>=21.0
pytrec-eval-terrier>=0.5
transformers>=4.44
"""


def _instance():
    from blaxel.core import SyncSandboxInstance
    try:
        return SyncSandboxInstance.get(SANDBOX_NAME)
    except Exception:
        envs = [{"name": k, "value": os.environ[k], "secret": True} for k in KEY_NAMES if os.environ.get(k)]
        cfg = {
            "name": SANDBOX_NAME,
            "memory": 8192,
            "envs": envs,
            "ttl": "6h",
        }
        try:
            return SyncSandboxInstance.create(cfg)
        except Exception:
            cfg.pop("ttl")
            return SyncSandboxInstance.create(cfg)


def _upload(inst) -> int:
    fs = inst.fs
    n = 0
    for rel in ("benchmarks/__init__.py",):
        p = REPO / rel
        fs.write(f"{REMOTE_DIR}/{rel}", p.read_text(encoding="utf-8"))
        n += 1
    for sub in ("src/jev_reranker", "benchmarks/frontier"):
        for p in sorted((REPO / sub).rglob("*.py")):
            rel = p.relative_to(REPO).as_posix()
            if "__pycache__" in rel:
                continue
            fs.write(f"{REMOTE_DIR}/{rel}", p.read_text(encoding="utf-8"))
            n += 1
    fs.write(f"{REMOTE_DIR}/requirements_sandbox.txt", REQUIREMENTS)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()
    fs.write(f"{REMOTE_DIR}/results/frontier/data/git_commit.txt", commit)
    n += 1
    return n


def _exec(inst, command: str, name: str, timeout: int = 1200, wait: bool = True, quiet: bool = False, env: bool = False) -> None:
    def on_out(line: str) -> None:
        if not quiet:
            print(line.rstrip())

    req: dict = {
        "command": command,
        "name": name,
        "working_dir": REMOTE_DIR,
        "wait_for_completion": wait,
        "timeout": timeout,
        "on_stdout": on_out,
        "on_stderr": on_out,
    }
    if env:
        req["env"] = {k: os.environ[k] for k in KEY_NAMES if os.environ.get(k)}
    inst.process.exec(req)


def setup() -> None:
    inst = _instance()
    print(f"sandbox up: {SANDBOX_NAME}")
    n = _upload(inst)
    print(f"uploaded {n} files")
    _exec(
        inst,
        "python3 -m venv /workspace/venv && "
        "/workspace/venv/bin/pip install -q --upgrade pip && "
        "/workspace/venv/bin/pip install -q -r requirements_sandbox.txt",
        "setup",
        timeout=1800,
    )
    print("deps installed")


def smoke() -> None:
    inst = _instance()
    _exec(
        inst,
        "PYTHONPATH=/workspace/src:/workspace /workspace/venv/bin/python3 -m benchmarks.frontier.run_frontier "
        "--smoke --out /workspace/results/frontier",
        "smoke",
        timeout=1800,
        env=True,
    )


def full() -> None:
    inst = _instance()
    _exec(
        inst,
        "PYTHONPATH=/workspace/src:/workspace nohup /workspace/venv/bin/python3 -m benchmarks.frontier.run_frontier "
        "--out /workspace/results/frontier > /workspace/results/full.log 2>&1 & echo started",
        "full-launch",
        timeout=60,
        quiet=True,
        env=True,
    )
    print("full run launched; poll with `poll`")


def poll() -> None:
    inst = _instance()
    fs = inst.fs
    try:
        partial = fs.read(f"{REMOTE_DIR}/results/frontier/metrics_partial.json")
        print(partial if isinstance(partial, str) else str(partial))
    except Exception as exc:
        print(f"(no metrics yet: {exc})")
    try:
        log = fs.read(f"{REMOTE_DIR}/results/full.log")
        text = log if isinstance(log, str) else str(log)
        print("---- log tail ----")
        print("\n".join(text.splitlines()[-15:]))
    except Exception as exc:
        print(f"(no log yet: {exc})")


def collect() -> None:
    inst = _instance()
    fs = inst.fs
    files: list[str] = []
    stack = [f"{REMOTE_DIR}/results/frontier"]
    while stack:
        d = stack.pop()
        for entry in fs.ls(d) or []:
            path = entry if isinstance(entry, str) else getattr(entry, "path", None)
            if not path:
                continue
            if str(getattr(entry, "type", "")) in ("dir", "directory") or (
                not isinstance(entry, str) and hasattr(entry, "path") and getattr(entry, "is_dir", False)
            ):
                stack.append(path)
            else:
                files.append(path)
    local_root = REPO / "benchmarks" / "results" / "frontier"
    for path in files:
        rel = path.split("/results/frontier/", 1)[-1]
        dest = local_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = fs.read(path)
        dest.write_text(data if isinstance(data, str) else str(data), encoding="utf-8")
        print(f"collected {rel}")
    print("collect done")


def teardown() -> None:
    inst = _instance()
    inst.delete()
    print(f"sandbox deleted: {SANDBOX_NAME}")


if __name__ == "__main__":
    if not all(os.environ.get(k) for k in (*KEY_NAMES, "BL_API_KEY", "BL_WORKSPACE")):
        sys.exit("source .env first (set -a; source .env; set +a)")
    cmd = sys.argv[1] if len(sys.argv) > 1 else "poll"
    {"setup": setup, "smoke": smoke, "full": full, "poll": poll, "collect": collect, "teardown": teardown}[cmd]()
    time.sleep(0.1)
