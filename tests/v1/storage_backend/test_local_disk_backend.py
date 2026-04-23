# SPDX-License-Identifier: Apache-2.0
# Standard
import asyncio
import os
import shutil
import tempfile

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.local_disk_backend import LocalDiskBackend


class MockLookupServer:
    def __init__(self):
        self.removed_keys = []
        self.inserted_keys = []

    def batched_remove(self, keys):
        self.removed_keys.extend(keys)

    def batched_insert(self, keys):
        self.inserted_keys.extend(keys)


class MockLMCacheWorker:
    def __init__(self):
        self.messages = []

    def put_msg(self, msg):
        self.messages.append(msg)


def create_test_config(
    disk_path: str,
    max_disk_size: float = 1.0,
    disk_gap_rate: float = 0.0,
    disk_gap_count: int = 0,
):
    """Create a test configuration for LocalDiskBackend."""
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        local_disk=disk_path,
        max_local_disk_size=max_disk_size,
        lmcache_instance_id="test_instance",
        disk_gap_rate=disk_gap_rate,
        disk_gap_count=disk_gap_count,
    )
    return config


def create_test_metadata():
    """Create a test metadata for LMCacheMetadata."""
    return LMCacheMetadata(
        model_name="test_model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(28, 2, 256, 8, 128),
    )


def create_test_key(key_id: int = 0) -> CacheEngineKey:
    """Create a test CacheEngineKey."""
    return CacheEngineKey(
        model_name="test_model",
        world_size=3,
        worker_id=1,
        chunk_hash=hash(key_id),
        dtype=torch.bfloat16,
    )


@pytest.fixture
def temp_disk_path():
    """Create a temporary directory for disk storage tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def async_loop():
    """Create an asyncio event loop for testing."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


# ----------------------------------------------------------------------------


@pytest.fixture
def local_cpu_backend(memory_allocator):
    """Create a LocalCPUBackend for testing."""
    config = LMCacheEngineConfig.from_legacy(chunk_size=256)
    return LocalCPUBackend(config, memory_allocator=memory_allocator)


@pytest.fixture
def local_disk_backend(temp_disk_path, async_loop, local_cpu_backend):
    """Create a LocalDiskBackend for testing."""
    config = create_test_config(temp_disk_path)
    return LocalDiskBackend(
        config=config,
        loop=async_loop,
        local_cpu_backend=local_cpu_backend,
        dst_device="cuda",
    )


class TestLocalDiskBackend:
    """Test cases for LocalDiskBackend."""

    def test_init(self, temp_disk_path, async_loop, local_cpu_backend):
        """Test LocalDiskBackend initialization."""
        config = create_test_config(temp_disk_path)
        backend = LocalDiskBackend(
            config=config,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cuda",
        )

        assert backend.dst_device == "cuda"
        assert backend.local_cpu_backend == local_cpu_backend
        assert backend.path == temp_disk_path
        assert os.path.exists(temp_disk_path)
        assert backend.lmcache_worker is None
        assert backend.instance_id == "test_instance"
        assert backend.usage == 0
        assert len(backend.dict) == 0

        local_cpu_backend.memory_allocator.close()

    def test_init_with_lookup_server_and_worker(
        self, temp_disk_path, async_loop, local_cpu_backend
    ):
        """Test LocalDiskBackend initialization with lookup server and worker."""
        config = create_test_config(temp_disk_path)
        lmcache_worker = MockLMCacheWorker()

        backend = LocalDiskBackend(
            config=config,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cuda",
            lmcache_worker=lmcache_worker,
        )

        assert backend.lmcache_worker == lmcache_worker

        local_cpu_backend.memory_allocator.close()

    def test_str(self, local_disk_backend):
        """Test string representation."""
        assert str(local_disk_backend) == "LocalDiskBackend"
        local_disk_backend.local_cpu_backend.memory_allocator.close()

    def test_key_to_path(self, local_disk_backend):
        """Test key to path conversion."""
        key = create_test_key(1)
        path = local_disk_backend._key_to_path(key)

        expected_filename = key.to_string().replace("/", "-") + ".pt"
        assert path == os.path.join(local_disk_backend.path, expected_filename)

        local_disk_backend.local_cpu_backend.memory_allocator.close()

    def test_contains_key_not_exists(self, local_disk_backend):
        """Test contains() when key doesn't exist."""
        key = create_test_key(2)
        assert not local_disk_backend.contains(key)
        assert not local_disk_backend.contains(key, pin=True)

        local_disk_backend.local_cpu_backend.memory_allocator.close()

    def test_get_blocking_key_not_exists(self, local_disk_backend):
        """Test get_blocking() when key doesn't exist."""
        key = create_test_key(2)
        result = local_disk_backend.get_blocking(key)

        assert result is None

        local_disk_backend.local_cpu_backend.memory_allocator.close()


# ---------------------------------------------------------------------------
# Tests for batched_contains_gaps() with fault injection
# ---------------------------------------------------------------------------
from unittest.mock import MagicMock, patch  # noqa: E402


def _make_backend_cpu(
    temp_disk_path,
    async_loop,
    disk_gap_rate: float = 0.0,
    disk_gap_count: int = 0,
) -> "LocalDiskBackend":
    """Create a LocalDiskBackend with cpu device (no CUDA needed)."""
    config = create_test_config(
        temp_disk_path,
        disk_gap_rate=disk_gap_rate,
        disk_gap_count=disk_gap_count,
    )
    cpu_config = LMCacheEngineConfig.from_legacy(chunk_size=256)
    from lmcache.v1.memory_management import AdHocMemoryAllocator
    allocator = AdHocMemoryAllocator(1024 * 1024 * 1024)  # 1 GiB
    cpu_backend = LocalCPUBackend(cpu_config, memory_allocator=allocator)
    return LocalDiskBackend(
        config=config,
        loop=async_loop,
        local_cpu_backend=cpu_backend,
        dst_device="cpu",
    )


def _inject_keys(backend: "LocalDiskBackend", keys) -> None:
    """Directly insert keys into backend.dict so contains() returns True."""
    for key in keys:
        backend.dict[key] = MagicMock()


class TestBatchedContainsGaps:
    """Tests for LocalDiskBackend.batched_contains_gaps()."""

    # ------------------------------------------------------------------
    # Basic behavior (no fault injection)
    # ------------------------------------------------------------------

    def test_empty_keys_returns_empty_gaps(self, temp_disk_path, async_loop):
        backend = _make_backend_cpu(temp_disk_path, async_loop)
        gaps, end = backend.batched_contains_gaps([])
        assert gaps == []
        assert end == 0

    def test_all_present_returns_empty_gaps(self, temp_disk_path, async_loop):
        backend = _make_backend_cpu(temp_disk_path, async_loop)
        keys = [create_test_key(i) for i in range(4)]
        _inject_keys(backend, keys)
        gaps, end = backend.batched_contains_gaps(keys)
        assert gaps == []
        assert end == 4

    def test_all_absent_returns_end_zero(self, temp_disk_path, async_loop):
        """All absent → no hits → end=0; no gaps (trailing misses not reported)."""
        backend = _make_backend_cpu(temp_disk_path, async_loop)
        keys = [create_test_key(i) for i in range(4)]
        gaps, end = backend.batched_contains_gaps(keys)
        assert gaps == []
        assert end == 0

    def test_gap_in_middle(self, temp_disk_path, async_loop):
        """Keys 0,1,3 present; key 2 absent → gap at (2,3)."""
        backend = _make_backend_cpu(temp_disk_path, async_loop)
        keys = [create_test_key(i) for i in range(4)]
        _inject_keys(backend, [keys[0], keys[1], keys[3]])
        gaps, end = backend.batched_contains_gaps(keys)
        assert gaps == [(2, 3)]
        assert end == 4

    def test_end_truncated_at_last_hit(self, temp_disk_path, async_loop):
        """Trailing misses stripped: end == last_hit_end, not len(keys)."""
        backend = _make_backend_cpu(temp_disk_path, async_loop)
        keys = [create_test_key(i) for i in range(5)]
        _inject_keys(backend, [keys[0], keys[1]])
        gaps, end = backend.batched_contains_gaps(keys)
        assert end == 2
        assert gaps == []

    def test_consecutive_gaps_merged(self, temp_disk_path, async_loop):
        """Keys 1,2 both absent → single interval (1,3), not two entries."""
        backend = _make_backend_cpu(temp_disk_path, async_loop)
        keys = [create_test_key(i) for i in range(5)]
        _inject_keys(backend, [keys[0], keys[3], keys[4]])
        gaps, end = backend.batched_contains_gaps(keys)
        assert gaps == [(1, 3)]
        assert end == 5

    def test_pin_hits_are_pinned(self, temp_disk_path, async_loop):
        backend = _make_backend_cpu(temp_disk_path, async_loop)
        keys = [create_test_key(i) for i in range(3)]
        mock_objs = [MagicMock() for _ in range(3)]
        for key, obj in zip(keys, mock_objs, strict=False):
            backend.dict[key] = obj

        gaps, end = backend.batched_contains_gaps(keys, pin=True)
        for obj in mock_objs:
            obj.pin.assert_called_once()

    # ------------------------------------------------------------------
    # Fault injection via disk_gap_rate
    # ------------------------------------------------------------------

    def test_gap_rate_zero_no_faults(self, temp_disk_path, async_loop):
        backend = _make_backend_cpu(temp_disk_path, async_loop, disk_gap_rate=0.0)
        keys = [create_test_key(i) for i in range(3)]
        _inject_keys(backend, keys)
        gaps, end = backend.batched_contains_gaps(keys)
        assert gaps == []
        assert end == 3

    def test_gap_rate_one_all_hits_become_gaps(self, temp_disk_path, async_loop):
        """disk_gap_rate=1.0 -> every real hit is suppressed; end=0."""
        backend = _make_backend_cpu(temp_disk_path, async_loop, disk_gap_rate=1.0)
        keys = [create_test_key(i) for i in range(3)]
        _inject_keys(backend, keys)
        gaps, end = backend.batched_contains_gaps(keys)
        assert gaps == []
        assert end == 0

    def test_gap_rate_one_real_misses_unaffected(self, temp_disk_path, async_loop):
        """Real misses are unaffected by fault injection -> end=0."""
        backend = _make_backend_cpu(temp_disk_path, async_loop, disk_gap_rate=1.0)
        keys = [create_test_key(i) for i in range(3)]
        gaps, end = backend.batched_contains_gaps(keys)
        assert gaps == []
        assert end == 0

    def test_gap_rate_partial_uses_random(self, temp_disk_path, async_loop):
        """disk_gap_rate=0.5: mock random to control which hits flip."""
        backend = _make_backend_cpu(temp_disk_path, async_loop, disk_gap_rate=0.5)
        keys = [create_test_key(i) for i in range(4)]
        _inject_keys(backend, keys)

        # 0.3 < 0.5 -> flip (gap), 0.7 >= 0.5 -> keep (hit)
        with patch(
            "lmcache.v1.storage_backend.local_disk_backend.random.random",
            side_effect=[0.3, 0.7, 0.3, 0.7],
        ):
            gaps, end = backend.batched_contains_gaps(keys)

        assert gaps == [(0, 1), (2, 3)]
        assert end == 4

    def test_gap_rate_not_applied_to_real_misses(self, temp_disk_path, async_loop):
        """random.random is NOT called for absent keys."""
        backend = _make_backend_cpu(temp_disk_path, async_loop, disk_gap_rate=0.5)
        keys = [create_test_key(i) for i in range(3)]
        _inject_keys(backend, [keys[0]])

        call_count = [0]

        def counting_random():
            call_count[0] += 1
            return 0.9

        with patch(
            "lmcache.v1.storage_backend.local_disk_backend.random.random",
            side_effect=counting_random,
        ):
            gaps, end = backend.batched_contains_gaps(keys)

        assert call_count[0] == 1
        assert gaps == []
        assert end == 1

    def test_fault_injected_keys_not_pinned(self, temp_disk_path, async_loop):
        """Keys reported as gaps are never pinned, even if really present."""
        backend = _make_backend_cpu(temp_disk_path, async_loop, disk_gap_rate=1.0)
        keys = [create_test_key(i) for i in range(2)]
        mock_objs = [MagicMock(), MagicMock()]
        for key, obj in zip(keys, mock_objs):
            backend.dict[key] = obj

        backend.batched_contains_gaps(keys, pin=True)
        for obj in mock_objs:
            obj.pin.assert_not_called()


class TestBatchedContainsGapsDiskRuntimeMutation:
    """Verify injection params are read from self.config at call time (runtime-mutable)."""

    def test_gap_rate_mutable_at_runtime(self, temp_disk_path, async_loop):
        """Mutating config.disk_gap_rate between calls changes injection behavior."""
        backend = _make_backend_cpu(temp_disk_path, async_loop)
        keys = [create_test_key(i) for i in range(4)]
        _inject_keys(backend, keys)

        gaps, end = backend.batched_contains_gaps(keys)
        assert gaps == [] and end == 4

        backend.config.disk_gap_rate = 1.0
        gaps, end = backend.batched_contains_gaps(keys)
        assert end == 0

        backend.config.disk_gap_rate = 0.0
        gaps, end = backend.batched_contains_gaps(keys)
        assert gaps == [] and end == 4

    def test_gap_count_mutable_at_runtime(self, temp_disk_path, async_loop):
        """Mutating config.disk_gap_count between calls changes injection behavior."""
        backend = _make_backend_cpu(temp_disk_path, async_loop)
        keys = [create_test_key(i) for i in range(6)]
        _inject_keys(backend, keys)

        gaps, end = backend.batched_contains_gaps(keys)
        assert gaps == [] and end == 6

        backend.config.disk_gap_count = 2
        with patch(
            "lmcache.v1.storage_backend.local_disk_backend.random.sample",
            return_value=[1, 3],
        ):
            gaps, end = backend.batched_contains_gaps(keys)
        assert gaps == [(1, 2), (3, 4)] and end == 6

        backend.config.disk_gap_count = 0
        gaps, end = backend.batched_contains_gaps(keys)
        assert gaps == [] and end == 6

    def test_gap_count_threshold_not_met(self, temp_disk_path, async_loop):
        """count mode is skipped when len(keys) < 2 * gap_count."""
        backend = _make_backend_cpu(temp_disk_path, async_loop, disk_gap_count=3)
        keys = [create_test_key(i) for i in range(5)]
        _inject_keys(backend, keys)
        gaps, end = backend.batched_contains_gaps(keys)
        assert gaps == [] and end == 5

    def test_gap_count_takes_precedence_over_rate(self, temp_disk_path, async_loop):
        """When both count and rate are non-zero, count mode wins."""
        backend = _make_backend_cpu(
            temp_disk_path, async_loop, disk_gap_rate=1.0, disk_gap_count=2
        )
        keys = [create_test_key(i) for i in range(6)]
        _inject_keys(backend, keys)
        with patch(
            "lmcache.v1.storage_backend.local_disk_backend.random.sample",
            return_value=[0, 2],
        ) as mock_sample, patch(
            "lmcache.v1.storage_backend.local_disk_backend.random.random"
        ) as mock_random:
            backend.batched_contains_gaps(keys)
        mock_sample.assert_called_once()
        mock_random.assert_not_called()

