from __future__ import annotations

import os
from pathlib import Path
import signal
import types

import pytest

from mctutil.ng import publish_scope as module


def write(path: Path, value: str) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(value, encoding="utf-8")


def available_scope(
	tmp_path: Path,
) -> tuple[module.ScopeCapability, list[list[str]]]:
	proc_root = tmp_path / "proc"
	cgroup_root = tmp_path / "cgroup"
	pid = 123
	manager_relative = "user.slice/user-1000.slice/user@1000.service"
	write(proc_root / str(pid) / "cgroup", "0::/user.slice/session.scope\n")
	write(cgroup_root / "cgroup.controllers", "cpu io memory pids\n")
	write(
		cgroup_root / manager_relative / "cgroup.controllers",
		"cpu memory pids\n",
	)
	write(cgroup_root / manager_relative / "memory.current", "100\n")
	calls = []

	def run(command, **kwargs):
		calls.append(command)
		return types.SimpleNamespace(
			returncode=0,
			stdout=f"/{manager_relative}\n",
			stderr="",
		)

	capability = module.detect_scope_capability(
		pid=pid,
		proc_root=proc_root,
		cgroup_root=cgroup_root,
		which=lambda name: f"/usr/bin/{name}",
		runner=run,
	)
	return capability, calls


def test_detects_available_transient_user_scope(tmp_path):
	capability, calls = available_scope(tmp_path)

	assert capability == module.ScopeCapability(
		module.ScopeAction.RELAUNCH,
		"transient user scope is available",
		"/usr/bin/systemd-run",
	)
	assert calls == [[
		"/usr/bin/systemctl",
		"--user",
		"show",
		"--property=ControlGroup",
		"--value",
	]]


@pytest.mark.parametrize(
	"relative",
	[
		"slurm/uid_1000/job_42/step_0",
		"system.slice/docker-123456.scope",
	],
)
def test_keeps_existing_isolated_cgroup_without_probing_systemd(
	tmp_path,
	relative,
):
	proc_root = tmp_path / "proc"
	cgroup_root = tmp_path / "cgroup"
	pid = 456
	write(proc_root / str(pid) / "cgroup", f"0::/{relative}\n")
	write(cgroup_root / "cgroup.controllers", "cpu memory\n")
	write(cgroup_root / relative / "memory.current", "100\n")
	write(cgroup_root / relative / "memory.max", "max\n")

	def unexpected(*_args, **_kwargs):
		raise AssertionError("existing isolated cgroup should skip systemd")

	capability = module.detect_scope_capability(
		pid=pid,
		proc_root=proc_root,
		cgroup_root=cgroup_root,
		which=unexpected,
		runner=unexpected,
	)

	assert capability.action == module.ScopeAction.USE_EXISTING


@pytest.mark.parametrize(
	("platform", "cgroup", "controllers", "expected"),
	[
		("darwin", "0::/\n", "memory\n", "requires Linux"),
		("linux", "2:memory:/\n", "memory\n", "cgroup v2 is unavailable"),
		("linux", "0::/\n", "cpu io\n", "memory controller is unavailable"),
	],
)
def test_rejects_unsupported_platform_or_cgroup(
	tmp_path,
	platform,
	cgroup,
	controllers,
	expected,
):
	proc_root = tmp_path / "proc"
	cgroup_root = tmp_path / "cgroup"
	write(proc_root / "10/cgroup", cgroup)
	write(cgroup_root / "cgroup.controllers", controllers)

	capability = module.detect_scope_capability(
		pid=10,
		proc_root=proc_root,
		cgroup_root=cgroup_root,
		platform=platform,
	)

	assert capability.action == module.ScopeAction.FALLBACK
	assert expected in capability.reason


def test_rejects_unavailable_user_manager(tmp_path):
	proc_root = tmp_path / "proc"
	cgroup_root = tmp_path / "cgroup"
	write(proc_root / "10/cgroup", "0::/user.slice/session.scope\n")
	write(cgroup_root / "cgroup.controllers", "cpu memory\n")

	capability = module.detect_scope_capability(
		pid=10,
		proc_root=proc_root,
		cgroup_root=cgroup_root,
		which=lambda name: f"/usr/bin/{name}",
		runner=lambda *_args, **_kwargs: types.SimpleNamespace(
			returncode=1,
			stdout="",
			stderr="Failed to connect to bus: No medium found\n",
		),
	)

	assert capability.action == module.ScopeAction.FALLBACK
	assert "Failed to connect to bus" in capability.reason


def test_rejects_user_manager_without_delegated_memory(tmp_path):
	proc_root = tmp_path / "proc"
	cgroup_root = tmp_path / "cgroup"
	manager_relative = "user.slice/user-1000.slice/user@1000.service"
	write(proc_root / "10/cgroup", "0::/user.slice/session.scope\n")
	write(cgroup_root / "cgroup.controllers", "cpu memory\n")
	write(cgroup_root / manager_relative / "cgroup.controllers", "cpu pids\n")
	write(cgroup_root / manager_relative / "memory.current", "100\n")

	capability = module.detect_scope_capability(
		pid=10,
		proc_root=proc_root,
		cgroup_root=cgroup_root,
		which=lambda name: f"/usr/bin/{name}",
		runner=lambda *_args, **_kwargs: types.SimpleNamespace(
			returncode=0,
			stdout=f"/{manager_relative}\n",
			stderr="",
		),
	)

	assert capability.action == module.ScopeAction.FALLBACK
	assert "lacks delegated memory accounting" in capability.reason


def test_reconstructs_group_and_direct_publish_commands(tmp_path):
	launcher = tmp_path / "mctutil"
	write(launcher, "#!/bin/sh\n")
	launcher.chmod(0o755)
	direct = tmp_path / "publish.py"
	write(direct, "# publish module\n")

	assert module.reconstruct_publish_command(
		[str(launcher), "--verbose", "ng", "publish", "/data", "--systemd-scope"]
	) == (
		str(launcher),
		"--verbose",
		"ng",
		"publish",
		"/data",
		"--systemd-scope",
	)
	assert module.reconstruct_publish_command(
		[str(direct), "/data"],
		python_executable="/usr/bin/python3",
	) == ("/usr/bin/python3", str(direct), "/data")
	with pytest.raises(ValueError, match="not ng publish"):
		module.reconstruct_publish_command([str(launcher), "ng", "shard"])


def test_scope_environment_prevents_recursion_and_marks_started(tmp_path):
	marker = tmp_path / "started"
	environment = {
		module.SCOPE_ACTIVE_ENV: "1",
		module.SCOPE_MARKER_ENV: str(marker),
	}

	def unexpected(*_args, **_kwargs):
		raise AssertionError("recursive invocation should not probe or launch")

	result = module.relaunch_publish_in_scope(
		True,
		environ=environment,
		capability_detector=unexpected,
		runner=unexpected,
	)

	assert result is None
	assert marker.is_file()


def test_capability_failure_warns_once_and_continues():
	warnings = []

	result = module.relaunch_publish_in_scope(
		True,
		capability_detector=lambda: module.ScopeCapability(
			module.ScopeAction.FALLBACK,
			"user manager unavailable",
		),
		diagnostic=warnings.append,
	)

	assert result is None
	assert warnings == ["user manager unavailable"]


def test_scope_creation_failure_falls_back_without_starting_publish():
	warnings = []

	result = module.relaunch_publish_in_scope(
		True,
		capability_detector=lambda: module.ScopeCapability(
			module.ScopeAction.RELAUNCH,
			"available",
			"/usr/bin/systemd-run",
		),
		command_builder=lambda _argv: ("/usr/bin/mctutil", "ng", "publish"),
		runner=lambda *_args, **_kwargs: types.SimpleNamespace(
			returncode=1,
			stderr="Failed to start transient scope\n",
		),
		diagnostic=warnings.append,
		announce=lambda _unit: None,
	)

	assert result is None
	assert warnings == [
		"could not create transient user scope (Failed to start transient scope)"
	]


@pytest.mark.parametrize(
	("returncode", "expected"),
	[(0, 0), (7, 7), (-signal.SIGTERM, 128 + signal.SIGTERM)],
)
def test_started_scope_preserves_invocation_and_propagates_status(
	tmp_path,
	monkeypatch,
	returncode,
	expected,
):
	recorded = {}

	def run(command, **kwargs):
		recorded["command"] = command
		recorded["kwargs"] = kwargs
		Path(kwargs["env"][module.SCOPE_MARKER_ENV]).touch()
		return types.SimpleNamespace(returncode=returncode, stderr="")

	monkeypatch.chdir(tmp_path)
	result = module.relaunch_publish_in_scope(
		True,
		environ={"PUBLISH_TEST": "kept"},
		capability_detector=lambda: module.ScopeCapability(
			module.ScopeAction.RELAUNCH,
			"available",
			"/usr/bin/systemd-run",
		),
		command_builder=lambda _argv: (
			"/usr/bin/mctutil",
			"ng",
			"publish",
			"/data",
		),
		runner=run,
		announce=lambda _unit: None,
	)

	assert result == expected
	command = recorded["command"]
	assert command[:6] == (
		"/usr/bin/systemd-run",
		"--user",
		"--scope",
		"--quiet",
		"--collect",
		"--same-dir",
	)
	assert "--property=Delegate=yes" in command
	assert "--property=MemoryAccounting=yes" in command
	assert command[-4:] == (
		"/usr/bin/mctutil",
		"ng",
		"publish",
		"/data",
	)
	kwargs = recorded["kwargs"]
	assert kwargs["cwd"] == tmp_path
	assert kwargs["env"]["PUBLISH_TEST"] == "kept"
	assert kwargs["env"][module.SCOPE_ACTIVE_ENV] == "1"
	assert "stdin" not in kwargs
	assert "stdout" not in kwargs
	assert "stderr" not in kwargs


def test_interruption_before_marker_is_not_retried():
	warnings = []

	result = module.relaunch_publish_in_scope(
		True,
		capability_detector=lambda: module.ScopeCapability(
			module.ScopeAction.RELAUNCH,
			"available",
			"/usr/bin/systemd-run",
		),
		command_builder=lambda _argv: ("/usr/bin/mctutil", "ng", "publish"),
		runner=lambda *_args, **_kwargs: types.SimpleNamespace(
			returncode=128 + signal.SIGINT,
			stderr="",
		),
		diagnostic=warnings.append,
		announce=lambda _unit: None,
	)

	assert result == 128 + signal.SIGINT
	assert warnings == []


def test_keyboard_interrupt_propagates_to_the_caller():
	def interrupt(*_args, **_kwargs):
		raise KeyboardInterrupt

	with pytest.raises(KeyboardInterrupt):
		module.relaunch_publish_in_scope(
			True,
			capability_detector=lambda: module.ScopeCapability(
				module.ScopeAction.RELAUNCH,
				"available",
				"/usr/bin/systemd-run",
			),
			command_builder=lambda _argv: (
				"/usr/bin/mctutil",
				"ng",
				"publish",
			),
			runner=interrupt,
			announce=lambda _unit: None,
		)


def test_scope_child_marks_started_then_executes(monkeypatch, tmp_path):
	marker = tmp_path / "started"
	recorded = {}

	def execvpe(program, arguments, environment):
		recorded.update(
			program=program,
			arguments=arguments,
			environment=environment,
		)
		raise RuntimeError("exec intercepted")

	monkeypatch.setattr(module.os, "execvpe", execvpe)
	with pytest.raises(RuntimeError, match="exec intercepted"):
		module.scope_child_main([str(marker), "/bin/echo", "hello"])

	assert marker.is_file()
	assert recorded["program"] == "/bin/echo"
	assert recorded["arguments"] == ("/bin/echo", "hello")
	assert recorded["environment"] is os.environ


def test_publish_flag_skips_scope_for_dry_run(load_module, tmp_path, monkeypatch):
	from click.testing import CliRunner

	publish = load_module("mctutil/ng/publish.py")
	root = tmp_path / "root"
	root.mkdir()
	dataset = root / "sample"
	dataset.mkdir()
	(dataset / "slice.tif").write_bytes(b"not read during dry run")
	calls = []
	monkeypatch.setattr(
		publish,
		"relaunch_publish_in_scope",
		lambda enabled: calls.append(enabled),
	)
	monkeypatch.setattr(publish, "module_available", lambda _name: False)

	result = CliRunner().invoke(
		publish.publish,
		[str(root), "--systemd-scope", "--dry-run", "--stop-after", "precompute"],
	)

	assert result.exit_code == 0, result.output
	assert calls == [False]
