# SPDX-License-Identifier: Apache-2.0
# Standard
from unittest.mock import MagicMock, patch

# Third Party
import pytest

# First Party
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend


def test_hot_cache_fill_ratio_default_is_one():
    config = LMCacheEngineConfig.from_defaults()
    config.validate()
    assert config.hot_cache_fill_ratio == 1.0


def test_hot_cache_fill_ratio_zero_raises():
    with pytest.raises(ValueError, match="hot_cache_fill_ratio"):
        config = LMCacheEngineConfig.from_defaults(hot_cache_fill_ratio=0.0)
        config.validate()


def test_hot_cache_fill_ratio_above_one_raises():
    with pytest.raises(ValueError, match="hot_cache_fill_ratio"):
        config = LMCacheEngineConfig.from_defaults(hot_cache_fill_ratio=1.1)
        config.validate()


def test_hot_cache_fill_ratio_valid_values():
    for ratio in [0.1, 0.5, 0.8, 1.0]:
        config = LMCacheEngineConfig.from_defaults(hot_cache_fill_ratio=ratio)
        config.validate()
        assert config.hot_cache_fill_ratio == ratio


def _make_backend(ratio: float, use_hot: bool = True) -> LocalCPUBackend:
    """Construct a LocalCPUBackend with a mocked allocator and fixed chunk budget."""
    config = LMCacheEngineConfig.from_defaults(
        local_cpu=use_hot,
        max_local_cpu_size=1,
        hot_cache_fill_ratio=ratio,
    )
    mock_allocator = MagicMock()
    mock_allocator.align_bytes = 4096

    with patch.object(LocalCPUBackend, "calculate_chunk_budget", return_value=10):
        backend = LocalCPUBackend(
            config=config,
            metadata=None,
            memory_allocator=mock_allocator,
        )
    return backend


def test_backend_constructs_with_fill_ratio_08():
    """LocalCPUBackend constructs without error when ratio=0.8."""
    _make_backend(ratio=0.8)


def test_backend_constructs_with_fill_ratio_one():
    """LocalCPUBackend constructs without error when ratio=1.0 (no cap)."""
    _make_backend(ratio=1.0)


def test_backend_constructs_with_use_hot_false():
    """LocalCPUBackend constructs without error when use_hot=False (cap disabled)."""
    _make_backend(ratio=0.8, use_hot=False)
