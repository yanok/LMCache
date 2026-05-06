# SPDX-License-Identifier: Apache-2.0
"""Tests for NixlStorageBackend shared-pool (CPU mode) constructor-based init.

Verifies that local_cpu_backend is wired up at construction time (no post_init),
get_allocator_backend() returns the correct backend, and error cases are caught
early. No real NIXL hardware needed — nixl package is mocked at sys.modules level.
"""

# Standard
import asyncio
import sys
import types
from unittest.mock import MagicMock, patch

# Third Party
import pytest
import torch


# ---------------------------------------------------------------------------
# Mock the nixl package before any import of nixl_storage_backend
# ---------------------------------------------------------------------------

def _make_nixl_mock() -> None:
    nixlBind_mock = MagicMock()
    nixlBind_mock.nixlRegDList = object
    nixlBind_mock.nixlXferDList = object
    nixlBind_mock.nixlBackendError = Exception

    sync_t_mock = MagicMock()
    sync_t_mock.NIXL_THREAD_SYNC_STRICT = "NIXL_THREAD_SYNC_STRICT"

    api_mock = types.ModuleType("nixl._api")
    api_mock.nixl_agent = MagicMock
    api_mock.nixl_agent_config = MagicMock
    api_mock.nixl_prepped_dlist_handle = MagicMock
    api_mock.nixl_xfer_handle = MagicMock
    api_mock.nixlBind = nixlBind_mock
    api_mock.nixl_thread_sync_t = sync_t_mock

    nixl_mock = types.ModuleType("nixl")
    nixl_mock._api = api_mock

    sys.modules.setdefault("nixl", nixl_mock)
    sys.modules.setdefault("nixl._api", api_mock)


_make_nixl_mock()

# First Party
import lmcache.v1.memory_management as memory_management_module  # noqa: E402
import lmcache.v1.storage_backend.nixl_storage_backend as nixl_module  # noqa: E402
from lmcache.v1.config import LMCacheEngineConfig  # noqa: E402
from lmcache.v1.memory_management import (  # noqa: E402
    MemoryFormat,
    MixedMemoryAllocator,
    PagedTensorMemoryAllocator,
)
from lmcache.v1.metadata import LMCacheMetadata  # noqa: E402
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_metadata() -> LMCacheMetadata:
    return LMCacheMetadata(
        model_name="test_model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(4, 2, 256, 8, 128),
    )


def _nixl_cpu_config(pool_size: int = 0) -> LMCacheEngineConfig:
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        local_cpu=True,
        lmcache_instance_id="test_nixl_shared",
    )
    config.nixl_buffer_size = 1024 * 1024
    config.nixl_buffer_device = "cpu"
    config.save_unfull_chunk = False
    config.extra_config = {
        "enable_nixl_storage": True,
        "nixl_backend": "OBJ",
        "nixl_pool_size": pool_size,
        "nixl_backend_params": {},
        "nixl_presence_cache": False,
        "nixl_async_put": False,
        "use_direct_io": False,
        "nixl_path": None,
        "nixl_use_hugepages": False,
        "nixl_enable_prog_thread": True,
    }
    return config


def _nixl_gpu_config(pool_size: int = 0) -> LMCacheEngineConfig:
    config = _nixl_cpu_config(pool_size)
    config.nixl_buffer_device = "cuda"
    return config


def _make_paged_allocator(metadata: LMCacheMetadata) -> PagedTensorMemoryAllocator:
    shapes = metadata.get_shapes()
    dtypes = metadata.get_dtypes()
    chunk_bytes = sum(s.numel() * d.itemsize for s, d in zip(shapes, dtypes, strict=True))
    buffer = torch.zeros(chunk_bytes * 4, dtype=torch.uint8)
    return PagedTensorMemoryAllocator(
        buffer, [torch.Size(metadata.kv_shape)], [metadata.kv_dtype], MemoryFormat.KV_2LTD,
    )


def _make_local_cpu_paged(monkeypatch, metadata: LMCacheMetadata) -> LocalCPUBackend:
    """LocalCPUBackend whose memory_allocator is MixedMemoryAllocator(use_paging=True)."""
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256, local_cpu=True, lmcache_instance_id="test_paged"
    )
    shapes = metadata.get_shapes()
    dtypes = metadata.get_dtypes()
    chunk_bytes = sum(s.numel() * d.itemsize for s, d in zip(shapes, dtypes, strict=True))
    aligned = chunk_bytes * 4
    config.max_local_cpu_size = aligned / (1024 ** 3)
    config.nixl_buffer_device = "cpu"
    config.extra_config = {"enable_nixl_storage": True}
    config.save_unfull_chunk = False
    real_buf = torch.zeros(aligned, dtype=torch.uint8)
    monkeypatch.setattr(
        memory_management_module, "_allocate_cpu_memory",
        lambda size, *a, **kw: real_buf,
    )
    return LocalCPUBackend(config=config, metadata=metadata, dst_device="cpu")


def _make_local_cpu_flat(metadata: LMCacheMetadata) -> LocalCPUBackend:
    """LocalCPUBackend whose memory_allocator is MixedMemoryAllocator(use_paging=False)."""
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256, local_cpu=True, lmcache_instance_id="test_flat"
    )
    config.max_local_cpu_size = 0.01
    return LocalCPUBackend(config=config, metadata=metadata, dst_device="cpu")


def _stub_dynamic_agent(monkeypatch) -> None:
    monkeypatch.setattr(nixl_module.NixlDynamicStorageAgent, "__init__",
                        lambda self, *a, **kw: None)
    monkeypatch.setattr(nixl_module.NixlDynamicStorageAgent, "close", lambda self: None)
    monkeypatch.setattr(nixl_module.NixlDynamicStorageAgent, "mem_type", "OBJ", raising=False)


def _stub_static_agent(monkeypatch) -> None:
    monkeypatch.setattr(nixl_module.NixlStaticStorageAgent, "__init__",
                        lambda self, *a, **kw: None)
    monkeypatch.setattr(nixl_module.NixlStaticStorageAgent, "close", lambda self: None)


def _stub_gpu_allocator(monkeypatch, allocator) -> None:
    monkeypatch.setattr(nixl_module.NixlStorageBackend, "initialize_allocator",
                        lambda self, config, metadata: allocator)


def _build_dynamic(monkeypatch, metadata, config, local_cpu_backend=None):
    _stub_dynamic_agent(monkeypatch)
    loop = asyncio.new_event_loop()
    nixl_config = nixl_module.NixlStorageConfig.from_cache_engine_config(config, metadata)
    backend = nixl_module.NixlDynamicStorageBackend(
        nixl_config, config, metadata, loop, local_cpu_backend=local_cpu_backend
    )
    return backend, loop


def _build_static(monkeypatch, metadata, config, local_cpu_backend=None):
    _stub_static_agent(monkeypatch)
    loop = asyncio.new_event_loop()
    nixl_config = nixl_module.NixlStorageConfig.from_cache_engine_config(config, metadata)
    backend = nixl_module.NixlStaticStorageBackend(
        nixl_config, config, metadata, loop, local_cpu_backend=local_cpu_backend
    )
    return backend, loop


# ---------------------------------------------------------------------------
# Tests: Dynamic CPU mode
# ---------------------------------------------------------------------------

class TestDynamicCpuMode:

    def test_cpu_mode_uses_local_cpu_allocator(self, monkeypatch):
        """Constructor sets memory_allocator to the inner PagedTensorMemoryAllocator
        and stores _local_cpu_backend."""
        metadata = _make_metadata()
        local_cpu = _make_local_cpu_paged(monkeypatch, metadata)
        mixed = local_cpu.get_memory_allocator()
        assert isinstance(mixed, MixedMemoryAllocator)
        assert isinstance(mixed.pin_allocator, PagedTensorMemoryAllocator)

        backend, loop = _build_dynamic(monkeypatch, metadata, _nixl_cpu_config(),
                                        local_cpu_backend=local_cpu)
        try:
            assert backend.memory_allocator is mixed.pin_allocator
            assert backend._local_cpu_backend is local_cpu
        finally:
            loop.close()
            local_cpu.memory_allocator.close()

    def test_cpu_mode_raises_without_local_cpu(self, monkeypatch):
        """Constructor raises RuntimeError when local_cpu_backend is None in CPU mode."""
        metadata = _make_metadata()
        _stub_dynamic_agent(monkeypatch)
        loop = asyncio.new_event_loop()
        try:
            nixl_config = nixl_module.NixlStorageConfig.from_cache_engine_config(
                _nixl_cpu_config(), metadata
            )
            with pytest.raises(RuntimeError, match="max_local_cpu_size"):
                nixl_module.NixlDynamicStorageBackend(
                    nixl_config, _nixl_cpu_config(), metadata, loop,
                    local_cpu_backend=None
                )
        finally:
            loop.close()

    def test_cpu_mode_raises_if_wrong_allocator_type(self, monkeypatch):
        """Constructor raises RuntimeError when LocalCPUBackend uses flat MixedMemoryAllocator."""
        metadata = _make_metadata()
        local_cpu_flat = _make_local_cpu_flat(metadata)
        _stub_dynamic_agent(monkeypatch)
        loop = asyncio.new_event_loop()
        try:
            nixl_config = nixl_module.NixlStorageConfig.from_cache_engine_config(
                _nixl_cpu_config(), metadata
            )
            with pytest.raises(RuntimeError, match="MixedMemoryAllocator\\(use_paging=True\\)"):
                nixl_module.NixlDynamicStorageBackend(
                    nixl_config, _nixl_cpu_config(), metadata, loop,
                    local_cpu_backend=local_cpu_flat
                )
        finally:
            loop.close()

    def test_get_allocator_backend_returns_local_cpu(self, monkeypatch):
        """get_allocator_backend() returns local_cpu_backend in CPU mode."""
        metadata = _make_metadata()
        local_cpu = _make_local_cpu_paged(monkeypatch, metadata)
        backend, loop = _build_dynamic(monkeypatch, metadata, _nixl_cpu_config(),
                                        local_cpu_backend=local_cpu)
        try:
            assert backend.get_allocator_backend() is local_cpu
        finally:
            loop.close()
            local_cpu.memory_allocator.close()


# ---------------------------------------------------------------------------
# Tests: Dynamic GPU mode
# ---------------------------------------------------------------------------

class TestDynamicGpuMode:

    def test_gpu_mode_uses_own_allocator(self, monkeypatch):
        """In GPU mode, constructor allocates its own buffer; _local_cpu_backend is None."""
        metadata = _make_metadata()
        fake_alloc = _make_paged_allocator(metadata)
        _stub_gpu_allocator(monkeypatch, fake_alloc)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0, raising=False)
        monkeypatch.setattr(torch.cuda, "set_device", lambda x: None, raising=False)

        backend, loop = _build_dynamic(monkeypatch, metadata, _nixl_gpu_config())
        try:
            assert backend.memory_allocator is fake_alloc
            assert backend._local_cpu_backend is None
        finally:
            loop.close()
            fake_alloc.close()

    def test_get_allocator_backend_returns_self(self, monkeypatch):
        """In GPU mode, get_allocator_backend() returns self."""
        metadata = _make_metadata()
        fake_alloc = _make_paged_allocator(metadata)
        _stub_gpu_allocator(monkeypatch, fake_alloc)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0, raising=False)
        monkeypatch.setattr(torch.cuda, "set_device", lambda x: None, raising=False)

        backend, loop = _build_dynamic(monkeypatch, metadata, _nixl_gpu_config())
        try:
            assert backend.get_allocator_backend() is backend
        finally:
            loop.close()
            fake_alloc.close()


# ---------------------------------------------------------------------------
# Tests: Static CPU mode
# ---------------------------------------------------------------------------

class TestStaticCpuMode:

    def test_cpu_mode_uses_local_cpu_allocator(self, monkeypatch):
        """Static backend: constructor sets memory_allocator to pin_allocator and
        constructs the agent immediately."""
        metadata = _make_metadata()
        local_cpu = _make_local_cpu_paged(monkeypatch, metadata)
        mixed = local_cpu.get_memory_allocator()
        paged = mixed.pin_allocator

        backend, loop = _build_static(monkeypatch, metadata, _nixl_cpu_config(pool_size=4),
                                       local_cpu_backend=local_cpu)
        try:
            assert backend.memory_allocator is paged
            assert backend._local_cpu_backend is local_cpu
            assert backend.agent is not None
        finally:
            loop.close()
            local_cpu.memory_allocator.close()

    def test_cpu_mode_raises_if_wrong_allocator_type(self, monkeypatch):
        """Static backend raises RuntimeError when LocalCPUBackend uses flat allocator."""
        metadata = _make_metadata()
        local_cpu_flat = _make_local_cpu_flat(metadata)
        _stub_static_agent(monkeypatch)
        loop = asyncio.new_event_loop()
        try:
            nixl_config = nixl_module.NixlStorageConfig.from_cache_engine_config(
                _nixl_cpu_config(pool_size=4), metadata
            )
            with pytest.raises(RuntimeError, match="MixedMemoryAllocator\\(use_paging=True\\)"):
                nixl_module.NixlStaticStorageBackend(
                    nixl_config, _nixl_cpu_config(pool_size=4), metadata, loop,
                    local_cpu_backend=local_cpu_flat
                )
        finally:
            loop.close()

    def test_gpu_mode_uses_own_allocator(self, monkeypatch):
        """Static GPU mode: constructor allocates its own buffer; _local_cpu_backend is None."""
        metadata = _make_metadata()
        fake_alloc = _make_paged_allocator(metadata)
        _stub_gpu_allocator(monkeypatch, fake_alloc)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0, raising=False)
        monkeypatch.setattr(torch.cuda, "set_device", lambda x: None, raising=False)

        backend, loop = _build_static(monkeypatch, metadata, _nixl_gpu_config(pool_size=4))
        try:
            assert backend.memory_allocator is fake_alloc
            assert backend._local_cpu_backend is None
        finally:
            loop.close()
            fake_alloc.close()
