"""Compact shard-capacity and worker planning for Neuroglancer stages."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from math import prod
import os
from pathlib import Path
import re

import numpy as np
import psutil

from mctutil.shared.log import log, LOG


GIB = 1024 ** 3
TIB = 1024 ** 4
MEMORY_RESERVE = 16 * GIB

LOW_CHUNK = (96, 96, 96)
MID_CHUNK = (64, 64, 64)
HIGH_CHUNK = (16, 16, 16)

_SIZE_PATTERN = re.compile(
	r"^\s*(\d+(?:\.\d+)?)\s*(B|KiB|MiB|GiB|TiB)?\s*$",
	re.IGNORECASE,
)
_SIZE_UNITS = {
	"b": 1,
	"kib": 1024,
	"mib": 1024 ** 2,
	"gib": GIB,
	"tib": TIB,
}


@dataclass(frozen=True)
class ResourcePlan:
	"""All output sizing and concurrency decisions for one dataset."""

	logical_bytes: int
	shard_ceiling: int
	# (mip, chunk size, chunks per shard, actual raw capacity)
	shards: tuple[tuple[int, tuple[int, int, int], int, int], ...]
	memory_capacity: int
	requested_workers: int
	cpu_limit: int
	memory_limit: int
	workers: int
	warning: str | None


def parse_size(value: str) -> int:
	match = _SIZE_PATTERN.fullmatch(value)
	if match is None:
		raise ValueError("must be bytes or a size such as 2GiB, 4GiB, or 8GiB")
	parsed = int(
		Decimal(match.group(1))
		* _SIZE_UNITS[(match.group(2) or "B").lower()]
	)
	if parsed <= 0:
		raise ValueError("must be greater than zero")
	return parsed


def format_size(value: int) -> str:
	for suffix, unit in (
		("TiB", TIB),
		("GiB", GIB),
		("MiB", 1024 ** 2),
		("KiB", 1024),
	):
		if value >= unit:
			amount = value / unit
			precision = 0 if value % unit == 0 else 4
			return f"{amount:.{precision}f} {suffix}"
	return f"{value} B"


def _read_cgroup_value(path: Path) -> int | None:
	try:
		value = path.read_text(encoding="utf-8").strip()
		parsed = int(value)
	except (FileNotFoundError, OSError, ValueError):
		return None
	# Cgroup v1 commonly uses a value near int64 max to mean unlimited.
	return None if parsed >= 2 ** 60 else parsed


def system_resources(
	cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> tuple[int, int]:
	"""Return stable effective memory capacity and CPU count."""
	capacities = [int(psutil.virtual_memory().total)]
	for limit_path in (
		cgroup_root / "memory.max",
		cgroup_root / "memory" / "memory.limit_in_bytes",
	):
		limit = _read_cgroup_value(limit_path)
		if limit is not None:
			capacities.append(limit)
	try:
		cpus = len(os.sched_getaffinity(0))
	except (AttributeError, OSError):
		cpus = os.cpu_count() or 1
	return min(capacities), max(1, cpus)


def _shard_plan(
	info: dict,
	mips: tuple[int, ...],
	capacity_override: int | None,
	low_chunk: tuple[int, int, int],
	mid_chunk: tuple[int, int, int],
	high_chunk: tuple[int, int, int],
):
	try:
		size = tuple(int(value) for value in info["scales"][0]["size"])
		dtype = np.dtype(info["data_type"])
		channels = int(info.get("num_channels", 1))
	except (IndexError, KeyError, TypeError, ValueError) as exc:
		raise ValueError("invalid MIP-0 metadata for resource planning") from exc
	if len(size) != 3 or any(value < 0 for value in size) or channels <= 0:
		raise ValueError("invalid MIP-0 dimensions or channel count")
	if not mips:
		raise ValueError("at least one MIP is required for resource planning")

	logical_bytes = prod(size) * dtype.itemsize * channels
	if capacity_override is not None:
		shard_ceiling = capacity_override
	elif logical_bytes <= 512 * GIB:
		shard_ceiling = 2 * GIB
	elif logical_bytes <= TIB:
		shard_ceiling = 4 * GIB
	else:
		shard_ceiling = 8 * GIB

	shards = []
	for mip in mips:
		chunk = low_chunk if mip <= 2 else mid_chunk if mip <= 4 else high_chunk
		chunk_bytes = prod(chunk) * dtype.itemsize * channels
		available_chunks = shard_ceiling // chunk_bytes
		if available_chunks < 1:
			raise ValueError(
				f"shard capacity {shard_ceiling} is smaller than one "
				f"{chunk} chunk ({chunk_bytes} bytes)"
			)
		chunks_per_shard = 1 << (available_chunks.bit_length() - 1)
		shards.append(
			(mip, chunk, chunks_per_shard, chunks_per_shard * chunk_bytes)
		)
	return logical_bytes, shard_ceiling, tuple(shards)


def plan_resources(
	info: dict,
	mips: tuple[int, ...],
	requested_workers: int,
	capacity_override: int | None = None,
	memory_capacity: int | None = None,
	cpu_limit: int | None = None,
	low_chunk: tuple[int, int, int] = LOW_CHUNK,
	mid_chunk: tuple[int, int, int] = MID_CHUNK,
	high_chunk: tuple[int, int, int] = HIGH_CHUNK,
) -> ResourcePlan:
	"""Compute the capacity tier, exact shard targets, and worker limit."""
	logical_bytes, shard_ceiling, shards = _shard_plan(
		info,
		mips,
		capacity_override,
		low_chunk,
		mid_chunk,
		high_chunk,
	)

	if memory_capacity is None or cpu_limit is None:
		system_ram, system_cpus = system_resources()
		memory_capacity = system_ram if memory_capacity is None else memory_capacity
		cpu_limit = system_cpus if cpu_limit is None else cpu_limit
	memory_capacity = max(0, int(memory_capacity))
	cpu_limit = max(1, int(cpu_limit))
	capacity_budget = max(shard[3] for shard in shards)
	warning = None
	if memory_capacity <= MEMORY_RESERVE:
		memory_limit = 1
		warning = "memory capacity does not exceed the 16 GiB reserve; using one worker"
	else:
		memory_limit = max(
			1,
			(memory_capacity - MEMORY_RESERVE) // (3 * capacity_budget),
		)
	workers = max(1, min(requested_workers, cpu_limit, memory_limit))
	return ResourcePlan(
		logical_bytes=logical_bytes,
		shard_ceiling=shard_ceiling,
		shards=tuple(shards),
		memory_capacity=memory_capacity,
		requested_workers=requested_workers,
		cpu_limit=cpu_limit,
		memory_limit=memory_limit,
		workers=workers,
		warning=warning,
	)


def log_resource_plan(
	label: str,
	plan: ResourcePlan | None,
	include_shards: bool = False,
) -> None:
	"""Emit one resource line and, when useful, one grouped shard-target line."""
	if plan is None:
		log.write(
			label,
			"Resources: determined after MIP 0.",
			log_level=LOG.INFO,
		)
		return
	log.write(
		label,
		(
			f"Resources: MIP 0={format_size(plan.logical_bytes)} logical; "
			f"shards={format_size(plan.shard_ceiling)} ceiling; "
			f"workers={plan.workers}/{plan.requested_workers} requested "
			f"(CPU {plan.cpu_limit}, RAM limit {plan.memory_limit} from "
			f"{format_size(plan.memory_capacity)} capacity minus 16 GiB reserve)."
		),
		log_level=LOG.INFO,
	)
	if include_shards:
		groups = {}
		for mip, chunk, _count, capacity in plan.shards:
			groups.setdefault((chunk, capacity), []).append(mip)
		parts = []
		for (chunk, capacity), mips in groups.items():
			consecutive = mips == list(range(mips[0], mips[-1] + 1))
			if len(mips) == 1:
				mip_text = str(mips[0])
			elif consecutive:
				mip_text = f"{mips[0]}-{mips[-1]}"
			else:
				mip_text = ",".join(str(mip) for mip in mips)
			parts.append(
				f"mips {mip_text}: {format_size(capacity)} at {chunk}"
			)
		log.write(
			label,
			f"Shard targets: {'; '.join(parts)}.",
			log_level=LOG.INFO,
		)
	if plan.warning is not None:
		log.write(label, f"Warning: {plan.warning}", log_level=LOG.WARN)
