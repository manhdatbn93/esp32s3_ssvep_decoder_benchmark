#!/usr/bin/env python3
"""Durable logs for one pipeline invocation. Dry-runs do not construct this."""
from __future__ import annotations

import json
import os
import re
import secrets
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{secrets.token_hex(4)}"


def git_state(repo: Path) -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None, None
    sha = commit.stdout.strip() if commit.returncode == 0 else None
    is_dirty = bool(dirty.stdout.strip()) if dirty.returncode == 0 else None
    return sha or None, is_dirty


def log_filename(index: int, step) -> str:
    parts = [f"{index:04d}"]
    if getattr(step, "phase", None):
        parts.append(step.phase)
    if getattr(step, "model", None):
        parts.append(step.model)
    if getattr(step, "fold", None):
        parts.append(f"fold{int(step.fold):02d}")
    if getattr(step, "precision", None):
        parts.append(str(step.precision))
    slug = "_".join(parts)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", slug).strip("_").lower()
    return f"{slug}.log"


def write_stdout_bytes(chunk: bytes) -> None:
    buffer = getattr(sys.stdout, "buffer", None)
    if buffer is not None:
        try:
            buffer.write(chunk)
            buffer.flush()
            return
        except Exception:
            pass
    sys.stdout.write(chunk.decode("utf-8", errors="replace"))
    sys.stdout.flush()


def stop_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def child_env() -> dict[str, str]:
    """Let Python children emit lines promptly without changing their calculations."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def run_logged_process(argv: list[str], cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    header = ((" ".join(argv)) + "\n").encode("utf-8", errors="backslashreplace")
    try:
        proc = subprocess.Popen(
            argv,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=child_env(),
        )
    except OSError as exc:
        log_path.write_bytes(header + f"{type(exc).__name__}: {exc}\n".encode("utf-8", errors="backslashreplace"))
        raise
    try:
        with log_path.open("wb") as log:
            log.write(header)
            log.flush()
            assert proc.stdout is not None
            read_available = getattr(proc.stdout, "read1", None)
            if read_available is None:
                raise OSError("pipeline log requires a buffered pipe with read1()")
            while True:
                chunk = read_available(8192)
                if not chunk:
                    break
                log.write(chunk)
                log.flush()
                os.fsync(log.fileno())
                write_stdout_bytes(chunk)
        return int(proc.wait())
    except KeyboardInterrupt:
        stop_process(proc)
        raise
    except OSError:
        stop_process(proc)
        raise


class RunLog:
    def __init__(self, directory: Path, summary: dict) -> None:
        self.directory = directory
        self.summary_path = directory / "summary.json"
        self.events_path = directory / "steps.jsonl"
        self.summary = summary
        self._started: dict[int, datetime] = {}

    @classmethod
    def create(cls, root: Path, repo: Path, action: str, argv: list[str], planned: int) -> "RunLog":
        root.mkdir(parents=True, exist_ok=True)
        for _ in range(5):
            run_id = new_run_id()
            directory = root / run_id
            try:
                directory.mkdir()
            except FileExistsError:
                continue
            break
        else:
            raise RuntimeError(f"could not allocate a pipeline run directory under {root}")
        (directory / "logs").mkdir()
        commit, dirty = git_state(repo)
        summary = {
            "run_id": directory.name,
            "action": action,
            "argv": list(argv),
            "git_commit": commit,
            "git_dirty": dirty,
            "started_at": utc_now(),
            "ended_at": None,
            "status": "RUNNING",
            "planned_steps": planned,
            "completed_steps": 0,
            "failed_steps": 0,
            "remaining_steps": planned,
            "first_failed_step_index": None,
            "first_failed_step_label": None,
            "failure_exit_code": None,
            "failed_log": None,
            "interrupted_step_index": None,
            "interrupted_step_label": None,
        }
        logger = cls(directory, summary)
        logger.write_summary()
        return logger

    def write_summary(self) -> None:
        temporary = self.summary_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.summary, indent=2), encoding="utf-8")
        os.replace(temporary, self.summary_path)

    def append_event(self, record: dict) -> None:
        with self.events_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.directory).as_posix()

    def log_path_for(self, index: int, step) -> Path:
        return self.directory / "logs" / log_filename(index, step)

    def begin_step(self, index: int, step) -> Path:
        self._started[index] = datetime.now(timezone.utc)
        log_path = self.log_path_for(index, step)
        self.append_event(self._event(index, step, "RUNNING", log_path, exit_code=None))
        return log_path

    def finish_step(self, index: int, step, status: str, exit_code: int | None, note: str | None = None) -> Path:
        log_path = self.log_path_for(index, step)
        if note:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(note.rstrip() + "\n")
        record = self._event(index, step, status, log_path, exit_code=exit_code)
        self.append_event(record)
        if status == "COMPLETED":
            self.summary["completed_steps"] += 1
        elif status == "FAILED":
            self.summary["failed_steps"] += 1
            self.summary["status"] = "FAILED"
            self.summary["ended_at"] = utc_now()
            self.summary["first_failed_step_index"] = index
            self.summary["first_failed_step_label"] = step.label
            self.summary["failure_exit_code"] = exit_code
            self.summary["failed_log"] = self._relative(log_path)
        elif status == "INTERRUPTED":
            self.summary["status"] = "INTERRUPTED"
            self.summary["ended_at"] = utc_now()
            self.summary["interrupted_step_index"] = index
            self.summary["interrupted_step_label"] = step.label
            self.summary["failure_exit_code"] = exit_code
            self.summary["failed_log"] = self._relative(log_path)
        planned = int(self.summary["planned_steps"])
        self.summary["remaining_steps"] = planned - int(self.summary["completed_steps"]) - int(self.summary["failed_steps"])
        self.write_summary()
        return log_path

    def complete(self) -> None:
        if self.summary["status"] != "RUNNING":
            return
        if int(self.summary["completed_steps"]) != int(self.summary["planned_steps"]):
            return
        self.summary["status"] = "COMPLETED"
        self.summary["ended_at"] = utc_now()
        self.summary["remaining_steps"] = 0
        self.write_summary()

    def _event(self, index: int, step, status: str, log_path: Path, exit_code: int | None) -> dict:
        started = self._started.get(index, datetime.now(timezone.utc))
        ended = None if status == "RUNNING" else datetime.now(timezone.utc)
        duration = None
        if ended is not None:
            duration = round((ended - started).total_seconds(), 6)
        return {
            "step_index": index,
            "label": step.label,
            "argv": list(step.argv),
            "model": getattr(step, "model", None),
            "fold": getattr(step, "fold", None),
            "precision": getattr(step, "precision", None),
            "phase": getattr(step, "phase", None),
            "started_at": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "ended_at": None if ended is None else ended.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "duration_seconds": duration,
            "exit_code": exit_code,
            "status": status,
            "log": self._relative(log_path),
        }
