"""Durable local task-queue execution for resumable Igneous stages."""

from __future__ import annotations

import hashlib
import importlib
import json
import multiprocessing
from pathlib import Path
import threading
import time


def stable_fingerprint(value) -> str:
	"""Return a stable short identifier for JSON-compatible configuration."""
	encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
	return hashlib.sha256(encoded).hexdigest()[:16]


def read_state(path: Path, default=None):
	if not path.exists():
		return default
	with path.open("r", encoding="utf-8") as handle:
		return json.load(handle)


def write_state(path: Path, value) -> None:
	"""Atomically replace a small JSON state file."""
	path.parent.mkdir(parents=True, exist_ok=True)
	temporary = path.with_name(f".{path.name}.tmp")
	with temporary.open("w", encoding="utf-8") as handle:
		json.dump(value, handle, indent=2, sort_keys=True)
		handle.write("\n")
	temporary.replace(path)


def file_queue_url(path: Path) -> str:
	return f"fq://{path.resolve()}"


def _require_taskqueue():
	try:
		from taskqueue import QueueEmptyError, TaskQueue
	except ImportError as exc:
		raise RuntimeError(
			"persistent task execution requires task-queue; "
			"install with pip install -e '.[mesh]'"
		) from exc
	return QueueEmptyError, TaskQueue


def _renew_lease(queue, task, lease_seconds: int, stopped: threading.Event) -> None:
	interval = max(1.0, lease_seconds / 2)
	while not stopped.wait(interval):
		queue.renew(task, lease_seconds)


def _drain_worker(
	queue_url: str,
	lease_seconds: int,
	task_modules: tuple[str, ...],
) -> None:
	"""Lease, execute, and acknowledge tasks until the durable queue is empty."""
	for module_name in task_modules:
		importlib.import_module(module_name)
	QueueEmptyError, TaskQueue = _require_taskqueue()
	queue = TaskQueue(queue_url, progress=False)
	while True:
		try:
			task = queue.lease(seconds=lease_seconds)
		except QueueEmptyError:
			if queue.is_empty():
				return
			time.sleep(0.1)
			continue

		stopped = threading.Event()
		renewer = threading.Thread(
			target=_renew_lease,
			args=(queue, task, lease_seconds, stopped),
			daemon=True,
		)
		renewer.start()
		try:
			task.execute()
			stopped.set()
			renewer.join()
			queue.delete(task, tally=True)
		except Exception:
			stopped.set()
			renewer.join()
			queue.cancel(task)
			raise


def drain_file_queue(
	queue_url: str,
	parallel: int,
	lease_seconds: int,
	task_modules: tuple[str, ...] = ("igneous.task_creation",),
) -> None:
	"""Execute a file queue in one or more independent worker processes."""
	if parallel < 1:
		raise ValueError("parallel must be at least 1")
	if parallel == 1:
		_drain_worker(queue_url, lease_seconds, task_modules)
		return

	context = multiprocessing.get_context("spawn")
	processes = [
		context.Process(
			target=_drain_worker,
			args=(queue_url, lease_seconds, task_modules),
		)
		for _index in range(parallel)
	]
	for process in processes:
		process.start()
	for process in processes:
		process.join()

	failed = [process.exitcode for process in processes if process.exitcode]
	if failed:
		raise RuntimeError(f"{len(failed)} task worker(s) failed with exit codes {failed}")


def run_persistent_tasks(
	queue_path: Path,
	task_fingerprint: str,
	tasks_factory,
	parallel: int,
	lease_seconds: int = 3600,
) -> dict:
	"""Insert a task set once and resume its durable queue until completion."""
	_QueueEmptyError, TaskQueue = _require_taskqueue()
	queue_path = queue_path.resolve()
	state_path = queue_path / "mctutil-state.json"
	state = read_state(state_path)
	if state is not None and state.get("fingerprint") != task_fingerprint:
		raise RuntimeError(f"persistent queue fingerprint mismatch: {queue_path}")

	queue = TaskQueue(file_queue_url(queue_path), progress=False)
	if state is None:
		state = {
			"fingerprint": task_fingerprint,
			"status": "inserting",
			"inserted": 0,
		}
		write_state(state_path, state)

	if state["status"] == "inserting":
		if queue.inserted == 0:
			# FileQueue commits its insertion counter only after the full insert
			# returns. If a process dies mid-insert, retrying the complete task
			# set can duplicate idempotent tasks but cannot accept a partial set.
			state["inserted"] = queue.insert(tasks_factory())
		else:
			state["inserted"] = queue.inserted
		state["status"] = "enqueued"
		write_state(state_path, state)

	if state["status"] == "complete":
		return state
	if queue.is_empty():
		state["status"] = "complete"
		write_state(state_path, state)
		return state

	state["status"] = "executing"
	write_state(state_path, state)
	drain_file_queue(file_queue_url(queue_path), parallel, lease_seconds)
	if not queue.is_empty():
		raise RuntimeError(f"persistent queue did not drain: {queue_path}")
	state["status"] = "complete"
	write_state(state_path, state)
	return state
