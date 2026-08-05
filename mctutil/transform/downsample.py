"""Compatibility surface for the deprecated ``transform downsample`` alias."""

from mctutil.shared import cli  # noqa: F401
from mctutil.transform.convert import downsample


if __name__ == "__main__":
	downsample()
