"""Local or remote HTTP smoke checks for precomputed layer endpoints."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import click


@dataclass(frozen=True)
class CheckResult:
	url: str
	method: str
	status: int | None
	size: int
	content_type: str | None
	error: str | None

	@property
	def ok(self) -> bool:
		return self.status is not None and 200 <= self.status < 300


def endpoint_url(base_url: str, path: str) -> str:
	return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def check_endpoint(
	url: str,
	method: str,
	timeout: float,
	byte_range: str | None = None,
) -> CheckResult:
	headers = {"Accept-Encoding": "identity"}
	if byte_range:
		headers["Range"] = f"bytes={byte_range}"
	request = Request(url, headers=headers, method=method.upper())
	try:
		with urlopen(request, timeout=timeout) as response:
			body = response.read() if method == "get" else b""
			size = (
				len(body)
				if method == "get"
				else int(response.headers.get("Content-Length", 0))
			)
			return CheckResult(
				url=url,
				method=method.upper(),
				status=response.status,
				size=size,
				content_type=response.headers.get_content_type(),
				error=None,
			)
	except HTTPError as exc:
		body = exc.read() if method == "get" else b""
		return CheckResult(
			url=url,
			method=method.upper(),
			status=exc.code,
			size=len(body),
			content_type=exc.headers.get_content_type() if exc.headers else None,
			error=str(exc.reason),
		)
	except (TimeoutError, URLError, OSError) as exc:
		reason = getattr(exc, "reason", exc)
		return CheckResult(
			url=url,
			method=method.upper(),
			status=None,
			size=0,
			content_type=None,
			error=str(reason),
		)


def echo_result(result: CheckResult) -> None:
	label = "PASS" if result.ok else "FAIL"
	status = result.status if result.status is not None else "error"
	content_type = result.content_type or "unknown"
	message = (
		f"{label} {result.method} {result.url} "
		f"status={status} size={result.size} type={content_type}"
	)
	if result.error:
		message += f" error={result.error}"
	click.echo(message)


@click.command("http-check")
@click.argument("base_url")
@click.option("--info/--no-info", "check_info", default=True, show_default=True)
@click.option("--info-path", default="info", show_default=True)
@click.option(
	"--chunk",
	"chunk_paths",
	multiple=True,
	help="Relative chunk endpoint; may be repeated.",
)
@click.option(
	"--method",
	type=click.Choice(("get", "head"), case_sensitive=False),
	default="get",
	show_default=True,
)
@click.option("--timeout", type=click.FloatRange(min=0.1), default=10.0, show_default=True)
@click.option(
	"--byte-range",
	help="Optional inclusive byte range, for example 0-1023.",
)
def http_check(
	base_url: str,
	check_info: bool,
	info_path: str,
	chunk_paths: tuple[str, ...],
	method: str,
	timeout: float,
	byte_range: str | None,
) -> None:
	"""Check a precomputed info endpoint and explicit chunk URLs."""
	try:
		paths = []
		if check_info:
			paths.append(info_path)
		paths.extend(chunk_paths)
		if not paths:
			raise ValueError("request at least one endpoint with --info or --chunk")

		results = [
			check_endpoint(
				endpoint_url(base_url, path),
				method.lower(),
				timeout,
				byte_range,
			)
			for path in paths
		]
		for result in results:
			echo_result(result)
		failed = [result for result in results if not result.ok]
		if failed:
			raise click.ClickException(
				f"{len(failed)} of {len(results)} endpoint checks failed"
			)
	except click.ClickException:
		raise
	except Exception as exc:
		raise click.ClickException(str(exc)) from exc


if __name__ == "__main__":
	http_check()
