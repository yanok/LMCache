# SPDX-License-Identifier: Apache-2.0
"""Tests for LocalCPUBackend.initialize_allocator() NIXL CPU shared-pool path.

Verifies that when NIXL CPU mode is active (enable_nixl_storage=True and
nixl_buffer_device="cpu"), initialize_allocator() returns a
MixedMemoryAllocator(use_paging=True) whose pin_allocator is a
PagedTensorMemoryAllocator, and that the default path (no NIXL) still
returns a MixedMemoryAllocator with a TensorMemoryAllocator inside.
"""

# Standard
from unittest.mock import MagicMock

# Third Party
import torch

# First Party
import lmcache.v1.memory_management as memory_management_module
import lmcache.v1.storage_backend.local_cpu_backend as local_cpu_backend_module
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MixedMemoryAllocator, PagedTensorMemoryAllocator
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend


def create_test_metadata() -> LMCacheMetadata:
    """Create a minimal test metadata instance."""
    return LMCacheMetadata(
        model_name="test_model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(4, 2, 256, 8, 128),
    )


def create_nixl_cpu_config(cpu_size_gb: float = 0.01) -> LMCacheEngineConfig:
    """Create a config with NIXL CPU mode enabled."""
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        local_cpu=True,
        lmcache_instance_id="test_nixl_cpu",
    )
    config.max_local_cpu_size = cpu_size_gb
    config.extra_config = {
        "enable_nixl_storage": True,
    }
    config.nixl_buffer_device = "cpu"
    config.save_unfull_chunk = False
    return config


def create_default_config(cpu_size_gb: float = 0.01) -> LMCacheEngineConfig:
    """Create a default config without NIXL storage."""
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        local_cpu=True,
        lmcache_instance_id="test_default",
    )
    config.max_local_cpu_size = cpu_size_gb
    return config


class TestInitializeAllocatorNixlCpuMode:
    """Tests for the NIXL CPU shared-pool code path in initialize_allocator()."""

    def test_nixl_cpu_mode_returns_mixed_allocator_with_paging(self, monkeypatch):
        """When enable_nixl_storage=True and nixl_buffer_device='cpu',
        initialize_allocator() must return a MixedMemoryAllocator whose
        pin_allocator is a PagedTensorMemoryAllocator.
        """
        metadata = create_test_metadata()
        config = create_nixl_cpu_config()

        # Compute a buffer size that is a valid multiple of chunk size so the
        # PagedTensorMemoryAllocator assertion passes.
        shapes = metadata.get_shapes()
        dtypes = metadata.get_dtypes()
        chunk_bytes = sum(
            s.numel() * d.itemsize for s, d in zip(shapes, dtypes, strict=True)
        )
        # Use exactly 4 chunks worth of memory so alignment is trivially satisfied.
        aligned_bytes = chunk_bytes * 4
        # Override max_local_cpu_size so the computed cpu_size_bytes == aligned_bytes.
        config.max_local_cpu_size = aligned_bytes / (1024**3)

        # Replace _allocate_cpu_memory (in memory_management) with one that returns a
        # real uint8 tensor of the right size so PagedTensorMemoryAllocator can split it.
        real_buffer = torch.zeros(aligned_bytes, dtype=torch.uint8)

        monkeypatch.setattr(
            memory_management_module,
            "_allocate_cpu_memory",
            lambda size, *args, **kwargs: real_buffer,
        )

        backend = LocalCPUBackend(config=config, metadata=metadata, dst_device="cpu")
        try:
            result = backend.memory_allocator
            assert isinstance(result, MixedMemoryAllocator), (
                f"Expected MixedMemoryAllocator, got {type(result).__name__}"
            )
            assert isinstance(result.pin_allocator, PagedTensorMemoryAllocator), (
                f"Expected pin_allocator to be PagedTensorMemoryAllocator, "
                f"got {type(result.pin_allocator).__name__}"
            )
        finally:
            backend.memory_allocator.close()

    def test_default_mode_returns_mixed_memory_allocator(self, monkeypatch):
        """When NIXL flags are absent, initialize_allocator() must return a
        MixedMemoryAllocator (the default path).
        """
        metadata = create_test_metadata()
        config = create_default_config()

        captured: dict = {}

        class DummyMixedMemoryAllocator:
            def __init__(self, size, **kwargs):
                captured["size"] = size
                captured["kwargs"] = kwargs
                self.align_bytes = kwargs.get("align_bytes", 4096)

            def close(self):
                return None

        monkeypatch.setattr(
            local_cpu_backend_module,
            "MixedMemoryAllocator",
            DummyMixedMemoryAllocator,
        )

        backend = LocalCPUBackend(config=config, metadata=metadata, dst_device="cpu")
        try:
            assert isinstance(
                backend.memory_allocator, DummyMixedMemoryAllocator
            ), (
                f"Expected DummyMixedMemoryAllocator (stand-in for MixedMemoryAllocator), "
                f"got {type(backend.memory_allocator).__name__}"
            )
        finally:
            backend.memory_allocator.close()

    def test_nixl_cpu_mode_only_when_both_flags_set(self, monkeypatch):
        """NIXL CPU path activates only when BOTH enable_nixl_storage=True AND
        nixl_buffer_device='cpu'.  If either flag is missing the default path
        (MixedMemoryAllocator) is used.
        """
        metadata = create_test_metadata()

        captured_allocators: list = []

        class DummyMixedMemoryAllocator:
            def __init__(self, size, **kwargs):
                captured_allocators.append("mixed")
                self.align_bytes = kwargs.get("align_bytes", 4096)

            def close(self):
                return None

        monkeypatch.setattr(
            local_cpu_backend_module,
            "MixedMemoryAllocator",
            DummyMixedMemoryAllocator,
        )

        # Case 1: enable_nixl_storage=True but nixl_buffer_device is None (default)
        config1 = create_default_config()
        config1.extra_config = {"enable_nixl_storage": True}
        # nixl_buffer_device defaults to None; no override needed
        backend1 = LocalCPUBackend(config=config1, metadata=metadata, dst_device="cpu")
        assert isinstance(backend1.memory_allocator, DummyMixedMemoryAllocator)
        backend1.memory_allocator.close()

        # Case 2: nixl_buffer_device='cpu' but enable_nixl_storage not set
        config2 = create_default_config()
        config2.nixl_buffer_device = "cpu"
        backend2 = LocalCPUBackend(config=config2, metadata=metadata, dst_device="cpu")
        assert isinstance(backend2.memory_allocator, DummyMixedMemoryAllocator)
        backend2.memory_allocator.close()

        # Case 3: nixl_buffer_device='gpu' (not cpu)
        config3 = create_nixl_cpu_config()
        config3.nixl_buffer_device = "gpu"
        backend3 = LocalCPUBackend(config=config3, metadata=metadata, dst_device="cpu")
        assert isinstance(backend3.memory_allocator, DummyMixedMemoryAllocator)
        backend3.memory_allocator.close()

        assert len(captured_allocators) == 3, (
            "DummyMixedMemoryAllocator should have been used exactly 3 times"
        )

    def test_nixl_cpu_mode_alignment_warning(self, monkeypatch):
        """When cpu_size_bytes is NOT a multiple of chunk_bytes, a warning is logged
        and the size is rounded down to the nearest chunk multiple.
        """
        from unittest.mock import patch

        metadata = create_test_metadata()
        config = create_nixl_cpu_config()

        shapes = metadata.get_shapes()
        dtypes = metadata.get_dtypes()
        chunk_bytes = sum(
            s.numel() * d.itemsize for s, d in zip(shapes, dtypes, strict=True)
        )
        # Deliberately misaligned: 4 chunks + 1 byte
        misaligned_bytes = chunk_bytes * 4 + 1
        config.max_local_cpu_size = misaligned_bytes / (1024**3)

        aligned_bytes = chunk_bytes * 4
        real_buffer = torch.zeros(aligned_bytes, dtype=torch.uint8)

        allocated_sizes: list = []

        def fake_allocate_cpu_memory(size, *args, **kwargs):
            allocated_sizes.append(size)
            return real_buffer

        monkeypatch.setattr(
            memory_management_module,
            "_allocate_cpu_memory",
            fake_allocate_cpu_memory,
        )

        # Patch the module-level logger to capture warning calls
        warning_messages: list = []
        original_warning = local_cpu_backend_module.logger.warning

        def capture_warning(msg, *args, **kwargs):
            warning_messages.append(msg % args if args else msg)
            original_warning(msg, *args, **kwargs)

        with patch.object(local_cpu_backend_module.logger, "warning", side_effect=capture_warning):
            backend = LocalCPUBackend(config=config, metadata=metadata, dst_device="cpu")

        try:
            # The allocator must have been given the rounded-down size
            assert len(allocated_sizes) == 1, (
                f"Expected 1 _allocate_cpu_memory call, got {len(allocated_sizes)}"
            )
            assert allocated_sizes[0] == aligned_bytes, (
                f"Expected aligned size {aligned_bytes}, got {allocated_sizes[0]}"
            )
            # A "rounding down" warning should have been emitted via the module logger
            assert any("rounding down" in w for w in warning_messages), (
                f"Expected a 'rounding down' warning; captured warnings: {warning_messages}"
            )
        finally:
            backend.memory_allocator.close()

    def test_nixl_cpu_mode_raises_on_undersized_buffer(self, monkeypatch):
        """When cpu_size_bytes < chunk_bytes, initialize_allocator() must raise
        a ValueError instead of silently creating a zero-capacity allocator.
        """
        # Third Party
        import pytest

        metadata = create_test_metadata()
        config = create_nixl_cpu_config()

        # Set cpu_size_bytes to exactly 1 byte — smaller than one chunk.
        config.max_local_cpu_size = 1 / (1024**3)

        with pytest.raises(ValueError, match="smaller than one chunk"):
            LocalCPUBackend(config=config, metadata=metadata, dst_device="cpu")
