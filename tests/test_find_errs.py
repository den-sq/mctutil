from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from mctutil.cli import main


def make_error_file(directory: Path, name: str, content: str) -> Path:
	directory.mkdir(parents=True, exist_ok=True)
	path = directory / name
	path.write_text(content, encoding="utf-8")
	return path


def test_error_directory_classification_is_sorted_and_error_wins(
	load_module,
	tmp_path,
):
	module = load_module("mctutil/parse/find_err_general.py")
	clean_job = tmp_path / "b-clean"
	errored_job = tmp_path / "a-errored"
	ignored_job = tmp_path / "c-no-error-file"
	ignored_job.mkdir()
	make_error_file(clean_job, "err-1", "")
	make_error_file(errored_job, "err-1", "")
	make_error_file(errored_job, "err-2", "traceback\n")

	errored, clean = module.classify_error_directories(tmp_path)

	assert errored == (errored_job,)
	assert clean == (clean_job,)
	assert ignored_job not in errored + clean


def test_find_errs_reports_and_optionally_writes_directory_lists(tmp_path):
	first_error = tmp_path / "jobs" / "a-error"
	second_error = tmp_path / "jobs" / "b-error"
	clean_job = tmp_path / "jobs" / "c-clean"
	make_error_file(second_error, "stderr-2", "second\n")
	make_error_file(first_error, "stderr-1", "first\n")
	make_error_file(clean_job, "stderr-3", "")
	errors_out = tmp_path / "errored.txt"
	clean_out = tmp_path / "clean.txt"

	result = CliRunner().invoke(
		main,
		[
			"parse",
			"find-errs",
			"--pattern", "stderr-*",
			"--errors-out", str(errors_out),
			"--clean-out", str(clean_out),
			str(tmp_path / "jobs"),
		],
	)

	assert result.exit_code == 0, result.output
	assert "Errored directories (2):" in result.output
	assert "Clean directories (1):" in result.output
	assert errors_out.read_text(encoding="utf-8") == (
		f"{first_error}\n{second_error}\n"
	)
	assert clean_out.read_text(encoding="utf-8") == f"{clean_job}\n"


def test_find_errs_rejects_one_file_for_both_output_lists(tmp_path):
	output = tmp_path / "jobs.txt"

	result = CliRunner().invoke(
		main,
		[
			"parse",
			"find-errs",
			"--errors-out", str(output),
			"--clean-out", str(output),
			str(tmp_path),
		],
	)

	assert result.exit_code == 2
	assert "must name different files" in result.output
	assert not output.exists()
