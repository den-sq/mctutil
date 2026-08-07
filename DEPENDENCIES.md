# Dependency contract

`environment.yml` is the supported full installation path. It provides the
compiled and conda-forge-first stack, including TomoPy, GDAL, and OpenCV. The
base dependencies and optional extras in `pyproject.toml` are also complete for
the features named by the shared dependency guards, so their installation hints
can be used in an ordinary Python environment.

## Version policy

- Direct runtime dependencies have a tested lower bound and an upper
  compatibility boundary. Routine patch and minor updates remain available
  inside that range; crossing a major compatibility boundary is deliberate and
  tested in one change.
- The supported numerical ABI is `numpy>=1.24,<2`, matching the TomoPy 1.x
  runtime installed from conda-forge. The same bound is present in pip metadata
  and `environment.yml` so installing an extra cannot silently upgrade the
  environment to NumPy 2. The bound is intentionally repeated in the
  environment's `pip:` subsection because pip resolves that subsection without
  applying conda's constraints; this prevents Igneous' OpenCV dependency from
  upgrading NumPy behind conda's back.
- Tifffile and Zarr are one compatibility set. Tifffile 2025.5.21 changed its
  Zarr store to require Zarr 3 and dropped Python 3.10. Because mctutil supports
  Python 3.10, all TIFF-bearing extras use
  `tifffile>=2024.8.30,<2025.5.21` and both Zarr-bearing extras use
  `zarr>=2.18,<3`. Update those bounds together.
- A distribution repeated across extras uses the same requirement everywhere.
  `environment.yml` uses the same compatibility bounds as pip metadata.
- Dragonfly APIs are excluded from pip and conda metadata because they exist
  only inside the ORS Dragonfly runtime.

## Verification

CI resolves `environment.yml`, runs `pip check`, and imports every lazy module
declared by `mctutil.shared.deps.EXTRA_MODULES`:

```console
python scripts/check_optional_dependencies.py --all
```

To check one pip extra in a clean environment, install it and select the same
name in the smoke command:

```console
python -m pip install -e '.[transform]'
python scripts/check_optional_dependencies.py --extra transform
mctutil transform memmap-prep --help
```

Google Sheets uploads use the same optional dependency contract:

```console
python -m pip install -e '.[google-sheets]'
python scripts/check_optional_dependencies.py --extra google-sheets
```

Repeat `--extra` to check a combination. TomoPy, GDAL, OpenCV, and Dragonfly
commands must be checked in their supported conda or application runtime.
