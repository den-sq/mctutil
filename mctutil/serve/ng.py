"""Serve local precomputed data with CORS and an optional Neuroglancer viewer."""

from __future__ import annotations

from functools import partial
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
import webbrowser

import click

from mctutil.shared.deps import require

LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


def detect_layer_type(layer_root: Path, requested: str) -> str:
	if requested != "auto":
		return requested
	try:
		info = json.loads((layer_root / "info").read_text(encoding="utf-8"))
	except FileNotFoundError as exc:
		raise ValueError(f"layer root has no info file: {layer_root}") from exc
	except json.JSONDecodeError as exc:
		raise ValueError(f"layer info is not valid JSON: {layer_root / 'info'}") from exc
	layer_type = info.get("type")
	if layer_type not in {"image", "segmentation"}:
		raise ValueError(f"unsupported or missing layer type: {layer_type!r}")
	return layer_type


def resolve_bind_address(bind: str, expose: bool) -> str:
	if expose and bind in LOOPBACK_HOSTS:
		return "0.0.0.0"
	if bind not in LOOPBACK_HOSTS and not expose:
		raise ValueError(
			"non-loopback binding requires the explicit --expose option"
		)
	return bind


def exposure_warning(bind: str, advertise_host: str) -> str | None:
	normalized_host = advertise_host.strip().strip("[]").lower()
	if bind in LOOPBACK_HOSTS or normalized_host not in LOOPBACK_HOSTS:
		return None
	return (
		f"Warning: server is unauthenticated with permissive CORS on {bind}; "
		f"advertised host {advertise_host} is loopback and will be unreachable "
		"from other devices. Set --advertise-host to a reachable hostname or "
		"address, and do not expose this server on untrusted networks."
	)


def echo_exposure_warning(bind: str, advertise_host: str) -> None:
	warning = exposure_warning(bind, advertise_host)
	if warning is not None:
		click.echo(warning, err=True)


def _require_range_handler():
	return require(
		"RangeHTTPServer",
		"serve",
		purpose="the range server requires RangeHTTPServer",
	).RangeRequestHandler


def cors_range_handler(layer_root: Path, quiet: bool):
	RangeRequestHandler = _require_range_handler()

	class CorsRangeRequestHandler(RangeRequestHandler):
		def end_headers(self):
			self.send_header("Access-Control-Allow-Origin", "*")
			self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
			self.send_header("Access-Control-Allow-Headers", "Range, Content-Type")
			self.send_header("Access-Control-Expose-Headers", "Content-Length, Content-Range")
			super().end_headers()

		def do_OPTIONS(self):
			self.send_response(204)
			self.end_headers()

		def log_message(self, format, *args):
			if not quiet:
				super().log_message(format, *args)

	return partial(CorsRangeRequestHandler, directory=str(layer_root))


def create_range_server(
	layer_root: Path,
	bind: str,
	port: int,
	quiet: bool,
):
	server = ThreadingHTTPServer(
		(bind, port),
		cors_range_handler(layer_root, quiet),
	)
	server.daemon_threads = True
	return server


def create_flask_server(
	layer_root: Path,
	bind: str,
	port: int,
	quiet: bool,
):
	flask, flask_cors = require(
		("flask", "flask_cors"),
		"serve",
		purpose="the Flask server requires flask and flask-cors",
	)
	from werkzeug.serving import make_server

	Flask = flask.Flask
	send_from_directory = flask.send_from_directory
	CORS = flask_cors.CORS

	app = Flask("mctutil-serve-ng", static_folder=None)
	CORS(app)

	@app.get("/<path:relative_path>")
	def serve_file(relative_path):
		return send_from_directory(
			layer_root,
			relative_path,
			conditional=True,
		)

	@app.route("/", methods=["OPTIONS"])
	@app.route("/<path:relative_path>", methods=["OPTIONS"])
	def options_response(relative_path=""):
		del relative_path
		return "", 204

	server = make_server(bind, port, app, threaded=True)
	if quiet:
		app.logger.disabled = True
	return server


def create_data_server(
	backend: str,
	layer_root: Path,
	bind: str,
	port: int,
	quiet: bool,
):
	if backend == "range":
		return create_range_server(layer_root, bind, port, quiet)
	return create_flask_server(layer_root, bind, port, quiet)


def advertise_url(url: str, host: str) -> str:
	parts = urlsplit(url)
	normalized_host = host.strip().strip("[]")
	if not normalized_host or any(character in normalized_host for character in "/?#@"):
		raise ValueError(f"invalid advertised host: {host!r}")
	netloc = f"[{normalized_host}]" if ":" in normalized_host else normalized_host
	if parts.port is not None:
		netloc += f":{parts.port}"
	return urlunsplit(parts._replace(netloc=netloc))


def create_viewer(
	data_url: str,
	layer_type: str,
	layer_name: str,
	bind: str,
	viewer_port: int,
	advertise_host: str,
) -> str:
	neuroglancer = require(
		"neuroglancer",
		"serve",
		purpose="the viewer requires neuroglancer",
	)

	neuroglancer.set_server_bind_address(
		bind_address=bind,
		bind_port=viewer_port,
	)
	viewer = neuroglancer.Viewer()
	source = f"precomputed://{data_url}"
	layer_class = (
		neuroglancer.SegmentationLayer
		if layer_type == "segmentation"
		else neuroglancer.ImageLayer
	)
	with viewer.txn() as state:
		state.layers[layer_name] = layer_class(source=source)
	return advertise_url(str(viewer), advertise_host)


def save_qr_code(url: str, path: Path) -> None:
	qrcode = require(
		"qrcode",
		"serve",
		purpose="QR creation requires qrcode",
	)
	path.parent.mkdir(parents=True, exist_ok=True)
	qrcode.make(url).save(path)


def run_server(
	server,
	data_url: str,
	viewer_url: str | None,
	open_browser: bool,
	qr_path: Path | None,
) -> None:
	click.echo(f"Data URL: {data_url}")
	if viewer_url is not None:
		click.echo(f"Viewer URL: {viewer_url}")
		if qr_path is not None:
			save_qr_code(viewer_url, qr_path)
			click.echo(f"QR code: {qr_path.resolve()}")
		if open_browser:
			webbrowser.open(viewer_url)
	click.echo("Press Ctrl-C to stop.")
	try:
		server.serve_forever()
	except KeyboardInterrupt:
		click.echo("Stopping server.")
	finally:
		server.server_close()


@click.command("ng")
@click.argument(
	"layer_root",
	type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
	"--backend",
	type=click.Choice(("range", "flask")),
	default="range",
	show_default=True,
)
@click.option("--bind", default="127.0.0.1", show_default=True)
@click.option(
	"--expose",
	is_flag=True,
	help=(
		"Permit unauthenticated, permissive-CORS non-loopback serving; "
		"changes the default bind to 0.0.0.0. Do not use on untrusted networks."
	),
)
@click.option(
	"--advertise-host",
	default="127.0.0.1",
	show_default=True,
	help="Hostname embedded in data URLs and QR codes.",
)
@click.option("--data-port", type=click.IntRange(0, 65535), default=8000, show_default=True)
@click.option("--viewer-port", type=click.IntRange(0, 65535), default=8080, show_default=True)
@click.option("--viewer/--data-only", default=True, show_default=True)
@click.option("--layer-type", type=click.Choice(("auto", "image", "segmentation")),
				default="auto", show_default=True)
@click.option("--layer-name", default="layer", show_default=True)
@click.option("--open-browser", is_flag=True, help="Open the generated viewer URL.")
@click.option("--qr", "qr_path", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--quiet-http/--verbose-http", default=True, show_default=True)
@click.option("--execute/--dry-run", default=True, show_default=True)
def ng(
	layer_root: Path,
	backend: str,
	bind: str,
	expose: bool,
	advertise_host: str,
	data_port: int,
	viewer_port: int,
	viewer: bool,
	layer_type: str,
	layer_name: str,
	open_browser: bool,
	qr_path: Path | None,
	quiet_http: bool,
	execute: bool,
) -> None:
	"""Serve a local precomputed layer and optionally launch its viewer."""
	try:
		layer_root = layer_root.resolve()
		bind = resolve_bind_address(bind, expose)
		layer_type = detect_layer_type(layer_root, layer_type)
		if viewer and data_port != 0 and data_port == viewer_port:
			raise ValueError("--data-port and --viewer-port must differ")
		if not viewer and (open_browser or qr_path is not None):
			raise ValueError("--open-browser and --qr require --viewer")
		echo_exposure_warning(bind, advertise_host)

		click.echo(f"Layer root: {layer_root}")
		click.echo(f"Layer type: {layer_type}")
		click.echo(f"Backend: {backend}; bind: {bind}:{data_port}")
		click.echo(f"Advertised host: {advertise_host}")
		if viewer:
			click.echo(f"Viewer bind: {bind}:{viewer_port}")
		if not execute:
			return

		server = create_data_server(
			backend,
			layer_root,
			bind,
			data_port,
			quiet_http,
		)
		actual_port = server.server_port
		data_url = advertise_url(
			f"http://127.0.0.1:{actual_port}/",
			advertise_host,
		)
		try:
			viewer_url = (
				create_viewer(
					data_url,
					layer_type,
					layer_name,
					bind,
					viewer_port,
					advertise_host,
				)
				if viewer
				else None
			)
		except Exception:
			server.server_close()
			raise
		run_server(
			server,
			data_url,
			viewer_url,
			open_browser,
			qr_path,
		)
	except click.ClickException:
		raise
	except Exception as exc:
		raise click.ClickException(str(exc)) from exc


if __name__ == "__main__":
	ng()
