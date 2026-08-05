"""Optional transient user-systemd scope for ``ng publish``."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
from typing import Callable, Mapping, Sequence
import uuid

from mctutil.shared.log import log, LOG
from mctutil.shared.resource_monitor import _cgroup_directory


SCOPE_ACTIVE_ENV = "MCTUTIL_PUBLISH_SYSTEMD_SCOPE"
SCOPE_MARKER_ENV = "MCTUTIL_PUBLISH_SCOPE_MARKER"
PROBE_TIMEOUT_SECONDS = 3


class ScopeAction(Enum):
	"""How an opted-in publish should obtain resource accounting."""

	RELAUNCH = "relaunch"
	USE_EXISTING = "use-existing"
	FALLBACK = "fallback"


@dataclass(frozen=True)
class ScopeCapability:
	action: ScopeAction
	reason: str
	systemd_run: str | None = None


def _unified_cgroup_relative(pid: int, proc_root: Path) -> str | None:
	try:
		lines = (proc_root / str(pid) / "cgroup").read_text(
			encoding="utf-8"
		).splitlines()
	except (FileNotFoundError, OSError):
		return None
	entry = next((line for line in lines if line.startswith("0::")), None)
	return None if entry is None else entry[3:].lstrip("/")


def _controllers(path: Path) -> frozenset[str]:
	try:
		return frozenset(
			(path / "cgroup.controllers").read_text(
				encoding="utf-8"
			).split()
		)
	except (FileNotFoundError, OSError):
		return frozenset()


def _probe_failure(result) -> str:
	message = (getattr(result, "stderr", "") or "").strip().splitlines()
	return message[0] if message else f"exit status {result.returncode}"


def detect_scope_capability(  # noqa: C901
	*,
	pid: int | None = None,
	proc_root: Path = Path("/proc"),
	cgroup_root: Path = Path("/sys/fs/cgroup"),
	platform: str = sys.platform,
	which: Callable[[str], str | None] = shutil.which,
	runner: Callable = subprocess.run,
) -> ScopeCapability:
	"""Detect whether a transient user scope is useful and available."""
	pid = os.getpid() if pid is None else pid
	if not platform.startswith("linux"):
		return ScopeCapability(ScopeAction.FALLBACK, "requires Linux cgroup v2")
	if _unified_cgroup_relative(pid, proc_root) is None:
		return ScopeCapability(ScopeAction.FALLBACK, "cgroup v2 is unavailable")
	if "memory" not in _controllers(cgroup_root):
		return ScopeCapability(
			ScopeAction.FALLBACK,
			"the cgroup-v2 memory controller is unavailable",
		)
	if _cgroup_directory(pid, proc_root, cgroup_root) is not None:
		return ScopeCapability(
			ScopeAction.USE_EXISTING,
			"publish is already in a suitable isolated cgroup",
		)

	systemd_run = which("systemd-run")
	systemctl = which("systemctl")
	if systemd_run is None or systemctl is None:
		return ScopeCapability(
			ScopeAction.FALLBACK,
			"systemd-run and systemctl are required",
		)
	try:
		manager = runner(
			[
				systemctl,
				"--user",
				"show",
				"--property=ControlGroup",
				"--value",
			],
			stdin=subprocess.DEVNULL,
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE,
			text=True,
			timeout=PROBE_TIMEOUT_SECONDS,
			check=False,
		)
	except (OSError, subprocess.TimeoutExpired) as exc:
		return ScopeCapability(
			ScopeAction.FALLBACK,
			f"user systemd manager is unavailable: {exc}",
		)
	if manager.returncode != 0:
		return ScopeCapability(
			ScopeAction.FALLBACK,
			f"user systemd manager is unavailable: {_probe_failure(manager)}",
		)
	manager_relative = manager.stdout.strip().lstrip("/")
	if not manager_relative:
		return ScopeCapability(
			ScopeAction.FALLBACK,
			"user systemd manager did not report its cgroup",
		)
	manager_directory = cgroup_root / manager_relative
	if (
		"memory" not in _controllers(manager_directory)
		or not (manager_directory / "memory.current").is_file()
	):
		return ScopeCapability(
			ScopeAction.FALLBACK,
			"user systemd manager lacks delegated memory accounting",
		)
	return ScopeCapability(
		ScopeAction.RELAUNCH,
		"transient user scope is available",
		systemd_run,
	)


def reconstruct_publish_command(
	argv: Sequence[str] | None = None,
	*,
	which: Callable[[str], str | None] = shutil.which,
	python_executable: str = sys.executable,
) -> tuple[str, ...]:
	"""Reconstruct only a real ``ng publish`` command without a shell."""
	arguments = tuple(sys.argv if argv is None else argv)
	if not arguments:
		raise ValueError("publish command line is empty")
	program = arguments[0]
	program_path = (
		program
		if Path(program).is_absolute()
		else which(program)
	)
	if program_path is None:
		candidate = Path(program)
		program_path = str(candidate.resolve()) if candidate.exists() else None
	if program_path is None:
		raise ValueError(f"cannot resolve publish launcher: {program}")

	tokens = arguments[1:]
	group_invocation = any(
		tokens[index:index + 2] == ("ng", "publish")
		for index in range(max(0, len(tokens) - 1))
	)
	direct_module = Path(program_path).stem == "publish"
	if not group_invocation and not direct_module:
		raise ValueError("current command is not ng publish")

	if os.access(program_path, os.X_OK):
		return (program_path, *tokens)
	if Path(program_path).suffix == ".py":
		return (python_executable, program_path, *tokens)
	raise ValueError(f"publish launcher is not executable: {program_path}")


def propagated_exit_status(returncode: int) -> int:
	"""Convert a subprocess signal result to the conventional shell status."""
	return returncode if returncode >= 0 else 128 + abs(returncode)


def _scope_command(
	systemd_run: str,
	unit: str,
	marker: Path,
	publish_command: Sequence[str],
	python_executable: str,
) -> tuple[str, ...]:
	return (
		systemd_run,
		"--user",
		"--scope",
		"--quiet",
		"--collect",
		"--same-dir",
		f"--unit={unit}",
		"--property=Delegate=yes",
		"--property=MemoryAccounting=yes",
		"--property=TasksAccounting=yes",
		"--description=mctutil ng publish",
		"--",
		python_executable,
		"-m",
		"mctutil.ng.publish_scope",
		str(marker),
		*publish_command,
	)


def _warn_fallback(reason: str) -> None:
	log.write(
		"Publish Scope",
		f"{reason}; continuing with process-tree accounting.",
		log_level=LOG.WARN,
	)


def relaunch_publish_in_scope(  # noqa: C901
	enabled: bool,
	*,
	argv: Sequence[str] | None = None,
	environ: Mapping[str, str] | None = None,
	capability_detector: Callable[..., ScopeCapability] = detect_scope_capability,
	command_builder: Callable[..., tuple[str, ...]] = reconstruct_publish_command,
	runner: Callable = subprocess.run,
	diagnostic: Callable[[str], None] = _warn_fallback,
	announce: Callable[[str], None] | None = None,
) -> int | None:
	"""Run publish in a transient scope, or return ``None`` to fall back."""
	environment = dict(os.environ if environ is None else environ)
	if environment.get(SCOPE_ACTIVE_ENV) == "1":
		marker = environment.get(SCOPE_MARKER_ENV)
		if marker:
			try:
				Path(marker).touch(exist_ok=True)
			except OSError:
				pass
		return None
	if not enabled:
		return None

	capability = capability_detector()
	if capability.action == ScopeAction.USE_EXISTING:
		return None
	if capability.action == ScopeAction.FALLBACK:
		diagnostic(capability.reason)
		return None
	if capability.systemd_run is None:
		diagnostic("scope capability did not provide a systemd-run executable")
		return None
	try:
		publish_command = command_builder(argv)
	except ValueError as exc:
		diagnostic(str(exc))
		return None

	unit = f"mctutil-publish-{os.getpid()}-{uuid.uuid4().hex[:8]}.scope"
	if announce is None:
		def announce(name: str) -> None:
			log.write(
				"Publish Scope",
				f"Relaunching in transient user scope {name}.",
				log_level=LOG.STATUS,
			)
	with tempfile.TemporaryDirectory(prefix="mctutil-publish-scope-") as temp:
		marker = Path(temp) / "started"
		scope_environment = dict(environment)
		scope_environment[SCOPE_ACTIVE_ENV] = "1"
		scope_environment[SCOPE_MARKER_ENV] = str(marker)
		command = _scope_command(
			capability.systemd_run,
			unit,
			marker,
			publish_command,
			sys.executable,
		)
		announce(unit)
		try:
			result = runner(
				command,
				env=scope_environment,
				cwd=Path.cwd(),
				check=False,
			)
		except KeyboardInterrupt:
			raise
		except OSError as exc:
			diagnostic(f"could not create transient user scope: {exc}")
			return None
		started = marker.is_file()
		status = propagated_exit_status(result.returncode)
		if started or result.returncode < 0 or status in {
			128 + signal.SIGINT,
			128 + signal.SIGTERM,
			128 + signal.SIGHUP,
		}:
			return status
		diagnostic(
			f"could not create transient user scope ({_probe_failure(result)})"
		)
		return None


def scope_child_main(arguments: Sequence[str] | None = None) -> None:
	"""Mark the scope as started, then replace this wrapper with publish."""
	arguments = tuple(sys.argv[1:] if arguments is None else arguments)
	if len(arguments) < 2:
		raise SystemExit("scope child requires MARKER and COMMAND")
	marker = Path(arguments[0])
	command = arguments[1:]
	marker.touch(exist_ok=True)
	os.execvpe(command[0], command, os.environ)


if __name__ == "__main__":
	scope_child_main()
