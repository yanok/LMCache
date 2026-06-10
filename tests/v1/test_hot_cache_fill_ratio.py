# SPDX-License-Identifier: Apache-2.0
# Standard
from unittest.mock import MagicMock, patch

# Third Party
import pytest

# First Party
from lmcache.utils import CacheEngineKey
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


# Test helpers


def _make_mock_memory_obj() -> MagicMock:
    """Return a MagicMock with ref_count_up and ref_count_down callables."""
    from lmcache.v1.memory_management import MemoryObj

    obj = MagicMock(spec=MemoryObj)
    obj.ref_count_up = MagicMock()
    obj.ref_count_down = MagicMock()
    return obj


def _make_cache_key(n: int) -> CacheEngineKey:
    from tests.v1.utils import dumb_cache_engine_key

    return dumb_cache_engine_key(n)


# Behavioral tests for hot_cache_fill_ratio admission control


def test_submit_put_task_drops_promotion_at_cap():
    """After cap promotions, further calls return None and do not ref_count_up."""
    backend = _make_backend(ratio=0.8)  # cap == 8 (floor(0.8 * 10))
    cap = 8

    for i in range(cap):
        obj = _make_mock_memory_obj()
        backend.submit_put_task(_make_cache_key(i), obj)
        assert obj.ref_count_up.called, f"key {i} should have been promoted"

    assert len(backend.hot_cache) == cap

    extra_obj = _make_mock_memory_obj()
    backend.submit_put_task(_make_cache_key(cap), extra_obj)
    assert not extra_obj.ref_count_up.called, "promotion beyond cap must not ref_count_up"
    assert len(backend.hot_cache) == cap


def test_submit_put_task_duplicate_key_not_blocked_by_cap():
    """Duplicate key guard fires before cap check; no double ref_count_up."""
    backend = _make_backend(ratio=0.8)
    key = _make_cache_key(0)
    obj = _make_mock_memory_obj()

    backend.submit_put_task(key, obj)
    assert obj.ref_count_up.call_count == 1

    obj2 = _make_mock_memory_obj()
    backend.submit_put_task(key, obj2)
    assert not obj2.ref_count_up.called


def test_submit_put_task_no_cap_at_ratio_one():
    """ratio=1.0 → no cap, unlimited promotions."""
    backend = _make_backend(ratio=1.0)

    for i in range(20):
        obj = _make_mock_memory_obj()
        backend.submit_put_task(_make_cache_key(i), obj)

    assert len(backend.hot_cache) == 20


def test_batched_submit_put_task_respects_cap():
    """batched_submit_put_task delegates per-key; cap is enforced transitively."""
    backend = _make_backend(ratio=0.8)
    cap = 8

    keys = [_make_cache_key(i) for i in range(cap + 3)]
    objs = [_make_mock_memory_obj() for _ in keys]

    backend.batched_submit_put_task(keys, objs)

    assert len(backend.hot_cache) == cap
    for obj in objs[:cap]:
        assert obj.ref_count_up.called
    for obj in objs[cap:]:
        assert not obj.ref_count_up.called


def test_allocate_succeeds_when_hot_cache_at_cap():
    """Pool allocation is not blocked by the promotion cap."""
    backend = _make_backend(ratio=0.8)
    cap = 8

    for i in range(cap):
        backend.submit_put_task(_make_cache_key(i), _make_mock_memory_obj())

    result = backend.memory_allocator.allocate(MagicMock(), MagicMock(), None)
    assert result is not None
