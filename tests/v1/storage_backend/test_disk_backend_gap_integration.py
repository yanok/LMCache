# SPDX-License-Identifier: Apache-2.0
"""Integration tests: LocalDiskBackend.batched_contains_gaps() inside
StorageManager.batched_contains_with_gaps().

These tests verify that the gap-reporting path works end-to-end with a real
LocalDiskBackend — not a stub backend. Each test creates a real backend,
injects keys into backend.dict (simulating the post-put state without disk I/O),
wraps it in a StorageManager, and calls batched_contains_with_gaps().

The key test is test_natural_hole_in_disk_cache: a genuinely absent key (key 2
of 4 is never inserted) is reported as a gap by the disk backend, and
StorageManager correctly propagates it as a gap interval (2, 3).
"""

# Standard
import asyncio
import shutil
import tempfile
import threading
from collections import OrderedDict
from typing import List
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.event_manager import EventManager
from lmcache.v1.memory_management import AdHocMemoryAllocator
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.local_disk_backend import LocalDiskBackend
from lmcache.v1.storage_backend.storage_manager import StorageManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_key(chunk_id: int) -> CacheEngineKey:
    return CacheEngineKey(
        model_name="test_model",
        world_size=1,
        worker_id=0,
        chunk_hash=chunk_id,
        dtype=torch.bfloat16,
    )


def make_disk_backend(disk_path: str, loop: asyncio.AbstractEventLoop) -> LocalDiskBackend:
    """Create a LocalDiskBackend using cpu device (no CUDA needed)."""
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        local_disk=disk_path,
        max_local_disk_size=1.0,
        lmcache_instance_id="test_gap_integration",
    )
    allocator = AdHocMemoryAllocator(256 * 1024 * 1024)  # 256 MiB
    cpu_config = LMCacheEngineConfig.from_legacy(chunk_size=256)
    cpu_backend = LocalCPUBackend(cpu_config, memory_allocator=allocator)
    return LocalDiskBackend(
        config=config,
        loop=loop,
        local_cpu_backend=cpu_backend,
        dst_device="cpu",
    )


def inject_keys(backend: LocalDiskBackend, keys: List[CacheEngineKey]) -> None:
    """Insert keys into backend.dict, simulating the post-put state.

    Uses MagicMock() as metadata — sufficient since all tests use pin=False.
    """
    for key in keys:
        backend.dict[key] = MagicMock()


def make_storage_manager(backend: LocalDiskBackend) -> StorageManager:
    """Create a StorageManager with only the given disk backend."""
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        local_cpu=False,
        lmcache_instance_id="test_gap_integration",
    )
    metadata = LMCacheMetadata(
        model_name="test_model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(28, 2, 256, 8, 128),
        role="scheduler",
    )
    event_manager = EventManager()
    manager = StorageManager(
        config=config,
        metadata=metadata,
        event_manager=event_manager,
    )
    # Replace backends with only our disk backend
    manager.storage_backends = OrderedDict({"LocalDiskBackend": backend})
    return manager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def async_loop():
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)
    loop.close()


@pytest.fixture
def temp_disk_path():
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def disk_backend(temp_disk_path, async_loop):
    backend = make_disk_backend(temp_disk_path, async_loop)
    yield backend
    # No explicit close needed; StorageManager.close() handles the backend


@pytest.fixture
def storage_manager(disk_backend):
    manager = make_storage_manager(disk_backend)
    yield manager
    manager.close()


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------

class TestDiskBackendGapIntegration:
    """
    Integration tests: LocalDiskBackend (real) inside StorageManager.

    These tests differ from unit tests in test_storage_manager_gaps.py (which
    use stub backends) and test_local_disk_backend.py (which test gaps in
    isolation). Here, the full call chain from StorageManager down to
    LocalDiskBackend.batched_contains_gaps() is exercised with real objects.
    """

    def test_baseline_all_present_no_gaps(
        self, disk_backend, storage_manager, monkeypatch
    ):
        """Phase 6c — regression baseline: all 4 keys present, gap_rate=0.0.

        batched_contains_with_gaps() must report furthest_hit_end=4 with no
        gaps and all keys in block_mapping, identical to batched_contains()
        prefix behavior.
        """
        monkeypatch.delenv("LMCACHE_DISK_GAP_RATE", raising=False)
        keys = [make_key(i) for i in range(4)]
        inject_keys(disk_backend, keys)

        furthest_hit_end, chunk_gaps, block_mapping = (
            storage_manager.batched_contains_with_gaps(keys)
        )

        assert furthest_hit_end == 4
        assert chunk_gaps == []
        assert set(block_mapping["LocalDiskBackend"]) == set(keys)

    def test_natural_hole_in_disk_cache(
        self, disk_backend, storage_manager, monkeypatch
    ):
        """Phase 6b — key test: a genuine absent key is reported as a gap.

        Keys 0, 1, 3 are stored; key 2 is never inserted.
        batched_contains_gaps() reports gaps=[(2,3)], end=4.
        batched_contains_with_gaps() must report:
          - furthest_hit_end = 4 (last hit is key 3 at index 3)
          - chunk_gaps = [(2, 3)]
          - block_mapping has keys 0, 1, 3 (not key 2)
        """
        monkeypatch.delenv("LMCACHE_DISK_GAP_RATE", raising=False)
        keys = [make_key(i) for i in range(4)]
        # Inject keys 0, 1, 3 — skip key 2 deliberately
        inject_keys(disk_backend, [keys[0], keys[1], keys[3]])

        furthest_hit_end, chunk_gaps, block_mapping = (
            storage_manager.batched_contains_with_gaps(keys)
        )

        assert furthest_hit_end == 4
        assert chunk_gaps == [(2, 3)]
        assert keys[2] not in block_mapping.get("LocalDiskBackend", [])
        assert keys[0] in block_mapping["LocalDiskBackend"]
        assert keys[1] in block_mapping["LocalDiskBackend"]
        assert keys[3] in block_mapping["LocalDiskBackend"]

    def test_fault_injection_full_rate_suppresses_all_hits(
        self, disk_backend, storage_manager, monkeypatch
    ):
        """Phase 6b — fault injection at rate=1.0 makes all hits appear as misses.

        Even though all 4 keys are present in the disk backend,
        LMCACHE_DISK_GAP_RATE=1.0 flips every hit to a miss.
        The result: furthest_hit_end=0, no chunk_gaps, empty block_mapping.
        """
        disk_backend.config.disk_gap_rate = 1.0
        keys = [make_key(i) for i in range(4)]
        inject_keys(disk_backend, keys)

        furthest_hit_end, chunk_gaps, block_mapping = (
            storage_manager.batched_contains_with_gaps(keys)
        )

        assert furthest_hit_end == 0
        assert chunk_gaps == []
        assert block_mapping == {}

    def test_regression_furthest_hit_matches_prefix_count(
        self, disk_backend, storage_manager, monkeypatch
    ):
        """Phase 6c — regression: furthest_hit_end == batched_contains() hit count.

        For a contiguous prefix hit (keys 0,1 present, keys 2,3 absent) with
        no fault injection, both methods must report the same extent.
        """
        monkeypatch.delenv("LMCACHE_DISK_GAP_RATE", raising=False)
        keys = [make_key(i) for i in range(4)]
        inject_keys(disk_backend, [keys[0], keys[1]])  # only first two

        # gap-aware lookup
        furthest_hit_end, chunk_gaps, _ = (
            storage_manager.batched_contains_with_gaps(keys)
        )
        # prefix-only lookup
        prefix_count, _ = storage_manager.batched_contains(keys)

        assert furthest_hit_end == prefix_count  # both = 2
        assert chunk_gaps == []  # trailing misses are not gaps

    def test_trailing_misses_not_reported_as_gaps(
        self, disk_backend, storage_manager, monkeypatch
    ):
        """Keys beyond the last hit are not gaps — they are simply uncached.

        Keys 0, 1 present; keys 2, 3 absent.
        The absence of keys 2, 3 must NOT appear as gaps because
        furthest_hit_end=2 and gaps are only within [0, furthest_hit_end).
        """
        monkeypatch.delenv("LMCACHE_DISK_GAP_RATE", raising=False)
        keys = [make_key(i) for i in range(4)]
        inject_keys(disk_backend, [keys[0], keys[1]])

        furthest_hit_end, chunk_gaps, block_mapping = (
            storage_manager.batched_contains_with_gaps(keys)
        )

        assert furthest_hit_end == 2
        assert chunk_gaps == []  # trailing absences stripped, not gaps
        assert len(block_mapping["LocalDiskBackend"]) == 2
