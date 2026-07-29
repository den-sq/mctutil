# mctutil serve

Local data and visualization servers.

Run `mctutil serve --help` to list commands and
`mctutil serve <task> --help` for a command's options.

## Commands

- **`ng`** — Serve local precomputed data with Range/CORS or Flask and an
  optional Neuroglancer viewer.

Install the server dependencies with:

```console
pip install -e '.[serve]'
```

The server is loopback-only by default. `--expose` enables unauthenticated
serving with permissive CORS on non-loopback interfaces; do not use it on
untrusted networks. Set `--advertise-host` to a hostname or address reachable
by the devices opening the generated viewer URL or QR code.
