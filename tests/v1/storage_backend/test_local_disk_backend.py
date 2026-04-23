# SPDX-License-Identifier: Apache-2.0
# Standard
from unittest.mock import MagicMock, patch
import asyncio
import os
import shutil
import tempfile

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey, DiskCacheMetadata
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.config_base import _parse_local_disk
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
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
    local_disk_path_sharding: str = "by_gpu",
    disk_gap_rate: float = 0.0,
    disk_gap_count: int = 0,
):
    """Create a test configuration for LocalDiskBackend."""
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        local_disk=disk_path,
        local_disk_path_sharding=local_disk_path_sharding,
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
        dst_device="cuda:0",
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
            dst_device="cuda:0",
        )

        assert backend.dst_device == "cuda:0"
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
            dst_device="cuda:0",
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


class TestMultiPathDiskBackend:
    """Test cases for multi-path (multi-device) LocalDiskBackend."""

    def test_init_multi_path(self, async_loop, local_cpu_backend):
        """Test initialisation with comma-separated paths."""
        dir_a = tempfile.mkdtemp()
        dir_b = tempfile.mkdtemp()
        try:
            combined = f"{dir_a},{dir_b}"
            config = create_test_config(combined)
            backend = LocalDiskBackend(
                config=config,
                loop=async_loop,
                local_cpu_backend=local_cpu_backend,
                dst_device="cuda:0",
            )

            # Path selected by device_id (0 % 2 = 0 -> dir_a)
            assert backend.path == dir_a
            # Both directories should exist
            assert os.path.isdir(dir_a)
            assert os.path.isdir(dir_b)
            # Block size is a plain int for the selected path
            assert isinstance(backend.os_disk_bs, int)
        finally:
            shutil.rmtree(dir_a, ignore_errors=True)
            shutil.rmtree(dir_b, ignore_errors=True)
            local_cpu_backend.memory_allocator.close()

    def test_gpu_affinity_selects_path(self, async_loop, local_cpu_backend):
        """Different cuda devices select different paths via modulo."""
        dir_a = tempfile.mkdtemp()
        dir_b = tempfile.mkdtemp()
        try:
            combined = f"{dir_a},{dir_b}"
            config = create_test_config(combined)

            dirs_by_gpu = {}
            for device in ("cuda:0", "cuda:1"):
                backend = LocalDiskBackend(
                    config=config,
                    loop=async_loop,
                    local_cpu_backend=local_cpu_backend,
                    dst_device=device,
                )
                dirs_by_gpu[device] = backend.path

            assert dirs_by_gpu["cuda:0"] == dir_a
            assert dirs_by_gpu["cuda:1"] == dir_b
        finally:
            shutil.rmtree(dir_a, ignore_errors=True)
            shutil.rmtree(dir_b, ignore_errors=True)
            local_cpu_backend.memory_allocator.close()

    def test_all_directories_created(self, async_loop, local_cpu_backend):
        """All paths in the list get their directories created."""
        base = tempfile.mkdtemp()
        try:
            paths = [os.path.join(base, f"nvme{i}") for i in range(3)]
            combined = ",".join(paths)
            config = create_test_config(combined)
            LocalDiskBackend(
                config=config,
                loop=async_loop,
                local_cpu_backend=local_cpu_backend,
                dst_device="cuda:0",
            )
            for p in paths:
                assert os.path.isdir(p), f"{p} should exist"
        finally:
            shutil.rmtree(base, ignore_errors=True)
            local_cpu_backend.memory_allocator.close()

    def test_single_path_backward_compat(
        self, temp_disk_path, async_loop, local_cpu_backend
    ):
        """A single path (no commas) works exactly as before."""
        config = create_test_config(temp_disk_path)
        backend = LocalDiskBackend(
            config=config,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cuda:0",
        )
        assert backend.path == temp_disk_path
        local_cpu_backend.memory_allocator.close()

    def test_path_sharding_default(self, temp_disk_path, async_loop, local_cpu_backend):
        """Default local_disk_path_sharding is 'by_gpu' (backend inits OK)."""
        config = create_test_config(temp_disk_path)
        backend = LocalDiskBackend(
            config=config,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cuda:0",
        )
        assert backend.path == temp_disk_path
        local_cpu_backend.memory_allocator.close()

    def test_path_sharding_explicit_by_gpu(
        self, temp_disk_path, async_loop, local_cpu_backend
    ):
        """Explicitly setting local_disk_path_sharding='by_gpu' works."""
        config = create_test_config(temp_disk_path, local_disk_path_sharding="by_gpu")
        backend = LocalDiskBackend(
            config=config,
            loop=async_loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cuda:0",
        )
        assert backend.path == temp_disk_path
        local_cpu_backend.memory_allocator.close()

    def test_path_sharding_unsupported_raises(
        self, temp_disk_path, async_loop, local_cpu_backend
    ):
        """Unsupported local_disk_path_sharding raises ValueError."""
        config = create_test_config(
            temp_disk_path, local_disk_path_sharding="round_robin"
        )
        with pytest.raises(ValueError, match="Unsupported path sharding strategy"):
            LocalDiskBackend(
                config=config,
                loop=async_loop,
                local_cpu_backend=local_cpu_backend,
                dst_device="cuda:0",
            )

    def test_cpu_dst_device_defaults_to_first_path(self, async_loop, local_cpu_backend):
        """dst_device='cpu' should fall back to device_id=0."""
        dir_a = tempfile.mkdtemp()
        dir_b = tempfile.mkdtemp()
        try:
            combined = f"{dir_a},{dir_b}"
            config = create_test_config(combined)
            backend = LocalDiskBackend(
                config=config,
                loop=async_loop,
                local_cpu_backend=local_cpu_backend,
                dst_device="cpu",
            )
            # device_id=0 -> 0 % 2 = 0 -> dir_a
            assert backend.path == dir_a
        finally:
            shutil.rmtree(dir_a, ignore_errors=True)
            shutil.rmtree(dir_b, ignore_errors=True)
            local_cpu_backend.memory_allocator.close()


class TestParseLocalDisk:
    """Unit tests for the _parse_local_disk config parser."""

    def test_none(self):
        assert _parse_local_disk(None) is None

    def test_single_raw_path(self):
        assert _parse_local_disk("/mnt/nvme0/cache/") == "/mnt/nvme0/cache/"

    def test_single_file_uri(self):
        assert _parse_local_disk("file:///mnt/nvme0/cache/") == "/mnt/nvme0/cache/"

    def test_single_file_uri_no_trailing_slash(self):
        assert _parse_local_disk("file:///mnt/nvme0/cache") == "/mnt/nvme0/cache"

    def test_comma_separated_raw(self):
        result = _parse_local_disk("/mnt/nvme0/,/mnt/nvme1/")
        assert result == "/mnt/nvme0/,/mnt/nvme1/"

    def test_comma_separated_file_uris(self):
        result = _parse_local_disk("file:///mnt/nvme0/,file:///mnt/nvme1/")
        assert result == "/mnt/nvme0/,/mnt/nvme1/"

    def test_mixed_uri_and_raw(self):
        result = _parse_local_disk("file:///mnt/nvme0/,/mnt/nvme1/")
        assert result == "/mnt/nvme0/,/mnt/nvme1/"

    def test_whitespace_around_paths(self):
        result = _parse_local_disk("  /mnt/nvme0/ , /mnt/nvme1/  ")
        assert result == "/mnt/nvme0/,/mnt/nvme1/"

    def test_empty_string(self):
        assert _parse_local_disk("") is None


class TestGetBlockingCachePolicyUpdate:
    """Regression tests for phantom cache hit in get_blocking() (issue #3015).

    ``get_blocking()`` must call ``cache_policy.update_on_hit()`` only when
    ``load_bytes_from_disk()`` returns a valid ``MemoryObj``.  Calling it
    before confirming load success records a phantom hit that skews future
    eviction decisions.
    """

    def _inject_key(
        self,
        backend: LocalDiskBackend,
        key: CacheEngineKey,
        shape: torch.Size,
        dtype: torch.dtype,
    ) -> None:
        """Insert a key into backend.dict without writing anything to disk."""
        meta = DiskCacheMetadata(
            path="/nonexistent/path.pt",
            size=0,
            shape=shape,
            dtype=dtype,
            cached_positions=None,
            fmt=MemoryFormat.KV_2LTD,
            pin_count=0,
        )
        with backend.disk_lock:
            backend.dict[key] = meta
            backend.cache_policy.update_on_put(key)

    def test_no_phantom_hit_when_load_fails(
        self, local_disk_backend: LocalDiskBackend
    ) -> None:
        """update_on_hit must NOT be called when load_bytes_from_disk returns None."""
        key = create_test_key(101)
        shape = torch.Size([28, 2, 256, 8, 128])
        self._inject_key(local_disk_backend, key, shape, torch.bfloat16)

        with patch.object(
            local_disk_backend, "load_bytes_from_disk", return_value=None
        ):
            with patch.object(
                local_disk_backend.cache_policy, "update_on_hit"
            ) as mock_update:
                result = local_disk_backend.get_blocking(key)

        assert result is None
        mock_update.assert_not_called()
        local_disk_backend.local_cpu_backend.memory_allocator.close()

    def test_updates_cache_policy_on_successful_load(
        self, local_disk_backend: LocalDiskBackend
    ) -> None:
        """update_on_hit must be called exactly once when the load succeeds."""
        key = create_test_key(102)
        shape = torch.Size([28, 2, 256, 8, 128])
        self._inject_key(local_disk_backend, key, shape, torch.bfloat16)

        fake_memory_obj = MagicMock(spec=MemoryObj)
        with patch.object(
            local_disk_backend, "load_bytes_from_disk", return_value=fake_memory_obj
        ):
            with patch.object(
                local_disk_backend.cache_policy, "update_on_hit"
            ) as mock_update:
                result = local_disk_backend.get_blocking(key)

        assert result is fake_memory_obj
        mock_update.assert_called_once_with(key, local_disk_backend.dict)
        local_disk_backend.local_cpu_backend.memory_allocator.close()

    def test_key_absent_returns_none_without_policy_update(
        self, local_disk_backend: LocalDiskBackend
    ) -> None:
        """get_blocking must return None immediately when the key is not cached."""
        key = create_test_key(103)

        with patch.object(
            local_disk_backend.cache_policy, "update_on_hit"
        ) as mock_update:
            result = local_disk_backend.get_blocking(key)

        assert result is None
        mock_update.assert_not_called()
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
