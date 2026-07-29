from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib
import json
from pathlib import Path
import threading
from urllib.request import Request, urlopen

from click.testing import CliRunner
import numpy as np
import pytest


VALID_INFO = {
	"type": "segmentation",
	"data_type": "uint32",
	"num_channels": 1,
	"scales": [
		{
			"key": "700_700_700",
			"encoding": "compressed_segmentation",
			"resolution": [700, 700, 700],
			"voxel_offset": [10, 20, 30],
			"size": [8, 8, 8],
			"chunk_sizes": [[4, 4, 4]],
		}
	],
}


def make_layer(tmp_path: Path) -> Path:
	layer = tmp_path / "layer"
	layer.mkdir()
	(layer / "info").write_text(json.dumps(VALID_INFO), encoding="utf-8")
	return layer


def test_serve_dry_run_is_loopback_safe_and_dependency_free(
	load_module,
	tmp_path,
	monkeypatch,
):
	module = load_module("mctutil/serve/ng.py")
	layer = make_layer(tmp_path)
	monkeypatch.setattr(
		module,
		"create_data_server",
		lambda *_args: (_ for _ in ()).throw(AssertionError("server created")),
	)

	result = CliRunner().invoke(
		module.ng,
		[
			str(layer),
			"--backend", "flask",
			"--data-port", "9000",
			"--viewer-port", "9001",
			"--dry-run",
		],
	)

	assert result.exit_code == 0, result.output
	assert "Layer type: segmentation" in result.output
	assert "Backend: flask; bind: 127.0.0.1:9000" in result.output
	assert "Viewer bind: 127.0.0.1:9001" in result.output


def test_serve_requires_explicit_exposure(load_module, tmp_path):
	module = load_module("mctutil/serve/ng.py")
	layer = make_layer(tmp_path)

	result = CliRunner().invoke(
		module.ng,
		[str(layer), "--bind", "0.0.0.0", "--dry-run"],
	)

	assert result.exit_code != 0
	assert "requires the explicit --expose option" in result.output

	exposed = CliRunner().invoke(
		module.ng,
		[str(layer), "--expose", "--dry-run"],
	)
	assert exposed.exit_code == 0, exposed.output
	assert "bind: 0.0.0.0:8000" in exposed.output


def test_serve_advertises_configured_host(load_module):
	module = load_module("mctutil/serve/ng.py")

	assert module.advertise_url(
		"http://0.0.0.0:8080/v/token/",
		"viewer.example.test",
	) == "http://viewer.example.test:8080/v/token/"
	assert module.advertise_url(
		"http://[::]:8080/v/token/",
		"2001:db8::1",
	) == "http://[2001:db8::1]:8080/v/token/"


def test_serve_wires_viewer_browser_qr_without_real_side_effects(
	load_module,
	tmp_path,
	monkeypatch,
):
	module = load_module("mctutil/serve/ng.py")
	layer = make_layer(tmp_path)
	events = []

	class FakeServer:
		server_port = 43210

		def serve_forever(self):
			events.append(("served",))

		def server_close(self):
			events.append(("closed",))

	monkeypatch.setattr(
		module,
		"create_data_server",
		lambda *args: events.append(("server", args)) or FakeServer(),
	)
	monkeypatch.setattr(
		module,
		"create_viewer",
		lambda *args: events.append(("viewer", args)) or "http://viewer/",
	)
	monkeypatch.setattr(
		module,
		"save_qr_code",
		lambda *args: events.append(("qr", args)),
	)
	monkeypatch.setattr(
		module.webbrowser,
		"open",
		lambda url: events.append(("browser", url)),
	)
	qr_path = tmp_path / "viewer.png"

	result = CliRunner().invoke(
		module.ng,
		[
			str(layer),
			"--data-port", "0",
			"--open-browser",
			"--qr", str(qr_path),
		],
	)

	assert result.exit_code == 0, result.output
	assert ("browser", "http://viewer/") in events
	assert ("qr", ("http://viewer/", qr_path)) in events
	assert ("served",) in events
	assert events[-1] == ("closed",)
	viewer_event = next(event for event in events if event[0] == "viewer")
	assert viewer_event[1][0] == "http://127.0.0.1:43210/"
	assert viewer_event[1][-1] == "127.0.0.1"


class SmokeHandler(BaseHTTPRequestHandler):
	def _respond(self, include_body):
		if self.path == "/info":
			body = b'{"type":"image"}'
			status = 200
			content_type = "application/json"
		elif self.path == "/chunk-ok":
			body = b"chunk-data"
			status = 200
			content_type = "application/octet-stream"
		else:
			body = b"missing"
			status = 404
			content_type = "text/plain"
		self.send_response(status)
		self.send_header("Content-Type", content_type)
		self.send_header("Content-Length", str(len(body)))
		self.end_headers()
		if include_body:
			self.wfile.write(body)

	def do_GET(self):
		self._respond(True)

	def do_HEAD(self):
		self._respond(False)

	def log_message(self, *_args):
		pass


@pytest.fixture()
def smoke_server():
	server = ThreadingHTTPServer(("127.0.0.1", 0), SmokeHandler)
	thread = threading.Thread(target=server.serve_forever, daemon=True)
	thread.start()
	try:
		yield f"http://127.0.0.1:{server.server_port}"
	finally:
		server.shutdown()
		server.server_close()
		thread.join()


def test_http_check_reports_success_metadata(load_module, smoke_server):
	module = load_module("mctutil/ng/http_check.py")

	result = CliRunner().invoke(
		module.http_check,
		[smoke_server, "--chunk", "chunk-ok"],
	)

	assert result.exit_code == 0, result.output
	assert f"PASS GET {smoke_server}/info status=200" in result.output
	assert "size=16 type=application/json" in result.output
	assert f"PASS GET {smoke_server}/chunk-ok status=200" in result.output
	assert "size=10 type=application/octet-stream" in result.output


def test_http_check_aggregates_failures_and_supports_head(
	load_module,
	smoke_server,
):
	module = load_module("mctutil/ng/http_check.py")

	result = CliRunner().invoke(
		module.http_check,
		[
			smoke_server,
			"--method", "head",
			"--chunk", "chunk-ok",
			"--chunk", "missing",
		],
	)

	assert result.exit_code != 0
	assert f"PASS HEAD {smoke_server}/info status=200 size=16" in result.output
	assert f"FAIL HEAD {smoke_server}/missing status=404" in result.output
	assert "1 of 3 endpoint checks failed" in result.output


class FakeBounds:
	minpt = (10, 20, 30)
	maxpt = (18, 28, 38)


class FakeVolume:
	info = VALID_INFO
	bounds = FakeBounds()

	def __init__(self, *_args, **_kwargs):
		pass

	def __getitem__(self, indices):
		shape = tuple(index.stop - index.start for index in indices)
		data = np.zeros(shape + (1,), dtype=np.uint32)
		data[0, 0, 0, 0] = 7
		return data


def test_validate_reports_metadata_and_configurable_reads(
	load_module,
	monkeypatch,
):
	module = load_module("mctutil/ng/validate.py")
	monkeypatch.setattr(module, "_require_cloudvolume", lambda: FakeVolume)
	monkeypatch.setattr(module, "patch_cloudfiles_monitoring", lambda: True)

	result = CliRunner().invoke(
		module.validate,
		[
			"file:///layer",
			"--block-size", "2,3,4",
			"--origin-at", "11,21,31",
			"--center-at", "15,25,35",
		],
	)

	assert result.exit_code == 0, result.output
	assert "Type: segmentation; dtype: uint32; channels: 1; mips: 1" in result.output
	assert "origin: (11, 21, 31)..(13, 24, 35)" in result.output
	assert "center: (14, 24, 33)..(16, 27, 37)" in result.output
	assert "unique=2" in result.output
	assert "Validation passed." in result.output


def test_validate_structural_failure_is_nonzero(load_module, monkeypatch):
	module = load_module("mctutil/ng/validate.py")

	class InvalidVolume(FakeVolume):
		info = {"type": "image", "scales": []}

	monkeypatch.setattr(module, "_require_cloudvolume", lambda: InvalidVolume)

	result = CliRunner().invoke(
		module.validate,
		["file:///bad", "--metadata-only"],
	)

	assert result.exit_code != 0
	assert "missing info field: data_type" in result.output
	assert "scales must be a non-empty list" in result.output


def test_validate_reads_real_local_cloudvolume(tmp_path):
	CloudVolume = pytest.importorskip("cloudvolume").CloudVolume
	module = importlib.import_module("mctutil.ng.validate")
	layer = tmp_path / "real-layer"
	info = CloudVolume.create_new_info(
		num_channels=1,
		layer_type="image",
		data_type="uint8",
		encoding="raw",
		resolution=[700, 700, 700],
		voxel_offset=[10, 20, 30],
		chunk_size=[4, 4, 4],
		volume_size=[8, 8, 8],
	)
	volume = CloudVolume(
		layer.resolve().as_uri(),
		info=info,
		parallel=False,
		compress=False,
	)
	volume.commit_info()
	volume.commit_provenance()
	volume[:] = np.arange(8 ** 3, dtype=np.uint8).reshape((8, 8, 8, 1))

	result = CliRunner().invoke(
		module.validate,
		[
			str(layer),
			"--block-size", "4,4,4",
		],
	)

	assert result.exit_code == 0, result.output
	assert "origin: (10, 20, 30)..(14, 24, 34)" in result.output
	assert "center: (12, 22, 32)..(16, 26, 36)" in result.output
	assert "Validation passed." in result.output


def test_range_server_supports_cors_and_byte_ranges(
	load_module,
	tmp_path,
):
	pytest.importorskip("RangeHTTPServer")
	module = load_module("mctutil/serve/ng.py")
	layer = make_layer(tmp_path)
	(layer / "chunk").write_bytes(b"0123456789")
	server = module.create_range_server(layer, "127.0.0.1", 0, True)
	thread = threading.Thread(target=server.serve_forever, daemon=True)
	thread.start()
	try:
		request = Request(
			f"http://127.0.0.1:{server.server_port}/chunk",
			headers={"Range": "bytes=2-5"},
		)
		with urlopen(request, timeout=2) as response:
			assert response.status == 206
			assert response.read() == b"2345"
			assert response.headers["Access-Control-Allow-Origin"] == "*"
	finally:
		server.shutdown()
		server.server_close()
		thread.join()


def test_flask_server_supports_cors_and_byte_ranges(
	load_module,
	tmp_path,
):
	pytest.importorskip("flask")
	pytest.importorskip("flask_cors")
	module = load_module("mctutil/serve/ng.py")
	layer = make_layer(tmp_path)
	(layer / "chunk").write_bytes(b"0123456789")
	server = module.create_flask_server(layer, "127.0.0.1", 0, True)
	thread = threading.Thread(target=server.serve_forever, daemon=True)
	thread.start()
	try:
		request = Request(
			f"http://127.0.0.1:{server.server_port}/chunk",
			headers={"Range": "bytes=2-5"},
		)
		with urlopen(request, timeout=2) as response:
			assert response.status == 206
			assert response.read() == b"2345"
			assert response.headers["Access-Control-Allow-Origin"] == "*"
	finally:
		server.shutdown()
		server.server_close()
		thread.join()


def test_server_dependencies_are_owned_by_serve_extra():
	pyproject = Path("pyproject.toml").read_text(encoding="utf-8").lower()
	ng_extra = pyproject.split("ng = [", 1)[1].split("\n]\n", 1)[0]
	serve_extra = pyproject.split("serve = [", 1)[1].split("\n]\n", 1)[0]
	for dependency in (
		'"flask"',
		'"flask-cors"',
		'"neuroglancer"',
		'"qrcode[pil]"',
		'"rangehttpserver"',
	):
		assert dependency in serve_extra
		assert dependency not in ng_extra
	assert '"cloud-volume"' in ng_extra


def test_server_is_routed_through_serve_category():
	from mctutil.cli import main

	runner = CliRunner()
	serve_help = runner.invoke(main, ["serve", "--help"])
	assert serve_help.exit_code == 0, serve_help.output
	assert "ng" in serve_help.output

	server_help = runner.invoke(main, ["serve", "ng", "--help"])
	assert server_help.exit_code == 0, server_help.output
	assert "--backend" in server_help.output

	old_route = runner.invoke(main, ["ng", "serve", "--help"])
	assert old_route.exit_code != 0
	assert "No such command 'serve'" in old_route.output
