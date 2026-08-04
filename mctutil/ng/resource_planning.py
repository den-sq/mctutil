"""Resource planning for post-MIP-0 Neuroglancer stages."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from math import prod
import os
from pathlib import Path
import re

import numpy as np
import psutil


KIB = 1024
MIB = 1024 ** 2
GIB = 1024 ** 3
TIB = 1024 ** 4

MIP0_SMALL_LIMIT = 512 * GIB
MIP0_MEDIUM_LIMIT = TIB
SMALL_SHARD_CAPACITY = 2 * GIB
MEDIUM_SHARD_CAPACITY = 4 * GIB
LARGE_SHARD_CAPACITY = 8 * GIB

MEMORY_RESERVE = 16 * GIB
SHARD_MEMORY_MULTIPLIER = 3

LOW_CHUNK = (96, 96, 96)
MID_CHUNK = (64, 64, 64)
HIGH_CHUNK = (16, 16, 16)

_SIZE_PATTERN = re.compile(
	r"^\s*(\d+(?:\.\d+)?)\s*(B|KiB|MiB|GiB|TiB)?\s*$",
	re.IGNORECASE,
)
_SIZE_UNITS = {
	"b": 1,
	"kib": KIB,
	"mib": MIB,
	"gib": GIB,
	"tib": TIB,
}


@dataclass(frozen=True)
class ScaleShardPlan:
	"""Power-of-two shard capacity selected for one destination MIP."""

	mip: int
	chunk_size: tuple[int, int, int]
	chunk_bytes: int
	chunks_per_shard: int
	capacity: int


@dataclass(frozen=True)
class ShardCapacityPlan:
	"""Dataset tier and exact realizable capacities for selected MIPs."""

	logical_mip0_bytes: int
	capacity_ceiling: int
	scales: tuple[ScaleShardPlan, ...]

	@property
	def maximum_actual_capacity(self) -> int:
		return max(scale.capacity for scale in self.scales)

	@property
	def targets(self) -> dict[int, int]:
		return {scale.mip: scale.capacity for scale in self.scales}


@dataclass(frozen=True)
class WorkerPlan:
	"""Inputs and result of the conservative concurrency calculation."""

	available_ram: int
	reserve: int
	capacity_budget: int
	memory_limit: int
	cpu_limit: int
	requested_limit: int
	workers: int
	warning: str | None = None


def parse_binary_size(value: str) -> int:
	"""Parse an integer byte count or a binary unit such as ``8GiB``."""
	match = _SIZE_PATTERN.fullmatch(value)
	if match is None:
		raise ValueError("must be bytes or a size such as 2GiB, 4GiB, or 8GiB")
	amount = Decimal(match.group(1))
	unit = (match.group(2) or "B").lower()
	parsed = int(amount * _SIZE_UNITS[unit])
	if parsed <= 0:
		raise ValueError("must be greater than zero")
	return parsed


def format_binary_size(value: int) -> str:
	"""Format a byte count using an exact or compact binary unit."""
	for suffix, unit in (("TiB", TIB), ("GiB", GIB), ("MiB", MIB), ("KiB", KIB)):
		if value >= unit:
			amount = value / unit
			precision = 0 if value % unit == 0 else 4
			return f"{amount:.{precision}f} {suffix}"
	return f"{value} B"


def logical_mip0_bytes(info: dict) -> int:
	"""Return the uncompressed logical byte size declared by MIP 0."""
	try:
		size = tuple(int(value) for value in info["scales"][0]["size"])
		dtype = np.dtype(info["data_type"])
		channels = int(info.get("num_channels", 1))
	except (IndexError, KeyError, TypeError, ValueError) as exc:
		raise ValueError("invalid MIP-0 metadata for resource planning") from exc
	if len(size) != 3 or any(value < 0 for value in size) or channels <= 0:
		raise ValueError("invalid MIP-0 dimensions or channel count")
	return prod(size) * dtype.itemsize * channels


def select_shard_capacity(logical_bytes: int) -> int:
	"""Choose the 2/4/8 GiB tier from the logical MIP-0 size."""
	if logical_bytes <= MIP0_SMALL_LIMIT:
		return SMALL_SHARD_CAPACITY
	if logical_bytes <= MIP0_MEDIUM_LIMIT:
		return MEDIUM_SHARD_CAPACITY
	return LARGE_SHARD_CAPACITY


def chunk_for_mip(
	mip: int,
	low_chunk: tuple[int, int, int] = LOW_CHUNK,
	mid_chunk: tuple[int, int, int] = MID_CHUNK,
	high_chunk: tuple[int, int, int] = HIGH_CHUNK,
) -> tuple[int, int, int]:
	if mip <= 2:
		return low_chunk
	if mip <= 4:
		return mid_chunk
	return high_chunk


def realizable_shard_capacity(
	capacity_ceiling: int,
	chunk_size: tuple[int, int, int],
	dtype,
	num_channels: int = 1,
) -> tuple[int, int, int]:
	"""Return chunk bytes, power-of-two chunks, and exact shard capacity."""
	chunk_bytes = prod(chunk_size) * np.dtype(dtype).itemsize * num_channels
	available_chunks = capacity_ceiling // chunk_bytes
	if available_chunks < 1:
		raise ValueError(
			f"shard capacity {capacity_ceiling} is smaller than one "
			f"{chunk_size} chunk ({chunk_bytes} bytes)"
		)
	chunks_per_shard = 1 << (available_chunks.bit_length() - 1)
	return chunk_bytes, chunks_per_shard, chunks_per_shard * chunk_bytes


def plan_shard_capacities(
	info: dict,
	mips: tuple[int, ...],
	capacity_override: int | None = None,
	low_chunk: tuple[int, int, int] = LOW_CHUNK,
	mid_chunk: tuple[int, int, int] = MID_CHUNK,
	high_chunk: tuple[int, int, int] = HIGH_CHUNK,
) -> ShardCapacityPlan:
	"""Build the dataset tier and exact per-MIP Igneous byte targets."""
	if not mips:
		raise ValueError("at least one MIP is required for shard planning")
	logical_bytes = logical_mip0_bytes(info)
	ceiling = capacity_override or select_shard_capacity(logical_bytes)
	dtype = info["data_type"]
	channels = int(info.get("num_channels", 1))
	scales = []
	for mip in mips:
		chunk_size = chunk_for_mip(mip, low_chunk, mid_chunk, high_chunk)
		chunk_bytes, chunks_per_shard, capacity = realizable_shard_capacity(
			ceiling,
			chunk_size,
			dtype,
			channels,
		)
		scales.append(
			ScaleShardPlan(
				mip=mip,
				chunk_size=chunk_size,
				chunk_bytes=chunk_bytes,
				chunks_per_shard=chunks_per_shard,
				capacity=capacity,
			)
		)
	return ShardCapacityPlan(logical_bytes, ceiling, tuple(scales))


def _read_cgroup_number(path: Path) -> int | None:
	try:
		value = path.read_text(encoding="utf-8").strip()
	except (FileNotFoundError, OSError):
		return None
	if value == "max":
		return None
	try:
		parsed = int(value)
	except ValueError:
		return None
	# Cgroup v1 commonly uses a value near int64 max to mean unlimited.
	return None if parsed >= 2 ** 60 else parsed


def cgroup_available_ram(root: Path = Path("/sys/fs/cgroup")) -> int | None:
	"""Return unused bytes from common cgroup v2 or v1 memory files."""
	pairs = (
		(root / "memory.max", root / "memory.current"),
		(
			root / "memory" / "memory.limit_in_bytes",
			root / "memory" / "memory.usage_in_bytes",
		),
	)
	available = []
	for limit_path, usage_path in pairs:
		limit = _read_cgroup_number(limit_path)
		usage = _read_cgroup_number(usage_path)
		if limit is not None and usage is not None:
			available.append(max(0, limit - usage))
	return min(available) if available else None


def system_available_ram() -> int:
	"""Return process-effective available RAM, including cgroup limits."""
	host_available = int(psutil.virtual_memory().available)
	cgroup_available = cgroup_available_ram()
	if cgroup_available is None:
		return host_available
	return min(host_available, cgroup_available)


def system_cpu_count() -> int:
	"""Return the process-effective CPU count when affinity is available."""
	try:
		return max(1, len(os.sched_getaffinity(0)))
	except (AttributeError, OSError):
		return max(1, os.cpu_count() or 1)


def plan_worker_limit(
	requested_limit: int,
	capacity_budget: int,
	available_ram: int | None = None,
	cpu_limit: int | None = None,
) -> WorkerPlan:
	"""Apply the 16 GiB reserve and threefold shard-memory headroom."""
	available_ram = system_available_ram() if available_ram is None else available_ram
	cpu_limit = system_cpu_count() if cpu_limit is None else max(1, cpu_limit)
	warning = None
	if available_ram <= MEMORY_RESERVE:
		memory_limit = 1
		warning = (
			f"available RAM ({format_binary_size(available_ram)}) does not exceed "
			f"the {format_binary_size(MEMORY_RESERVE)} reserve; using one worker"
		)
	else:
		memory_limit = max(
			1,
			(available_ram - MEMORY_RESERVE)
			// (SHARD_MEMORY_MULTIPLIER * capacity_budget),
		)
	workers = max(1, min(requested_limit, cpu_limit, memory_limit))
	return WorkerPlan(
		available_ram=available_ram,
		reserve=MEMORY_RESERVE,
		capacity_budget=capacity_budget,
		memory_limit=memory_limit,
		cpu_limit=cpu_limit,
		requested_limit=requested_limit,
		workers=workers,
		warning=warning,
	)
