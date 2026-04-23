# SPDX-License-Identifier: Apache-2.0
"""Tests for StorageManager.batched_contains_with_gaps().

Tests the four core scenarios:
1. Backend returns None (prefix fallback)
2. All hits from a single backend
3. Gaps in middle of a single backend's response
4. Truncation triggering suffix cascade to next tier
Plus: multi-tier cascade with mixed gap/prefix backends,
      PDBackend pin special-casing, empty keys,
      no-cross-tier-gap-fill contract.
"""

import torch
import pytest
from typing import List, Optional, Tuple
from collections import OrderedDict

from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.event_manager import EventManager
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface
from lmcache.v1.storage_backend.storage_manager import StorageManager


def make_key(chunk_id: int) -> CacheEngineKey:
    return CacheEngineKey(
        model_name="test_model",
        world_size=1,
        worker_id=0,
        chunk_hash=chunk_id,
        dtype=torch.bfloat16,
    )


# ---------------------------------------------------------------------------
# Minimal concrete mock backends
# ---------------------------------------------------------------------------

class _BaseStub(StorageBackendInterface):
    """Shared abstract-method stubs — not exercised by these tests."""

    def __init__(self, present_ids: set):
        super().__init__(dst_device="cpu")
        self._present = present_ids

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        return key.chunk_hash in self._present

    def exists_in_put_tasks(self, key): raise NotImplementedError
    def batched_submit_put_task(self, keys, objs, transfer_spec=None, on_complete_callback=None): raise NotImplementedError
    def get_blocking(self, key): raise NotImplementedError
    def pin(self, key): raise NotImplementedError
    def unpin(self, key): raise NotImplementedError
    def remove(self, key, force=True): raise NotImplementedError
    def get_allocator_backend(self): raise NotImplementedError
    def close(self): pass


class PrefixOnlyBackend(_BaseStub):
    """Does NOT override batched_contains_gaps (returns None → prefix fallback)."""

    def __init__(self, present_ids: set):
        super().__init__(present_ids)
        self.was_consulted = False

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        self.was_consulted = True
        return key.chunk_hash in self._present


class GapAwareBackend(_BaseStub):
    """Overrides batched_contains_gaps to return gap intervals."""

    def batched_contains_gaps(
        self,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> Tuple[List[Tuple[int, int]], int]:
        gaps = []
        in_gap = False
        gap_start = 0
        for i, key in enumerate(keys):
            if self.contains(key, pin):
                if in_gap:
                    gaps.append((gap_start, i))
                    in_gap = False
            else:
                if not in_gap:
                    gap_start = i
                    in_gap = True
        if in_gap:
            gaps.append((gap_start, len(keys)))
        return (gaps, len(keys))


class TruncatingBackend(GapAwareBackend):
    """Like GapAwareBackend but truncates result after _max_keys entries."""

    def __init__(self, present_ids: set, max_keys: int):
        super().__init__(present_ids)
        self._max_keys = max_keys

    def batched_contains_gaps(
        self,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> Tuple[List[Tuple[int, int]], int]:
        truncated = keys[: self._max_keys]
        gaps = []
        in_gap = False
        gap_start = 0
        for i, key in enumerate(truncated):
            if self.contains(key, pin):
                if in_gap:
                    gaps.append((gap_start, i))
                    in_gap = False
            else:
                if not in_gap:
                    gap_start = i
                    in_gap = True
        if in_gap:
            gaps.append((gap_start, len(truncated)))
        return (gaps, len(truncated))  # end = max_keys < len(keys) signals truncation


class TrackingGapBackend(GapAwareBackend):
    """Records the pin argument it received."""

    def __init__(self, present_ids: set):
        super().__init__(present_ids)
        self.gaps_pin_arg = None

    def batched_contains_gaps(
        self,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> Tuple[List[Tuple[int, int]], int]:
        self.gaps_pin_arg = pin
        return super().batched_contains_gaps(keys, pin)


# ---------------------------------------------------------------------------
# Helper: build a StorageManager with injected backends
# ---------------------------------------------------------------------------

def make_storage_manager_with_backends(backends: dict) -> StorageManager:
    """Create a minimal StorageManager then replace its storage_backends."""
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        local_cpu=False,
        lmcache_instance_id="test_gaps",
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
    manager.storage_backends = OrderedDict(backends)
    return manager


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBatchedContainsWithGaps:

    def setup_method(self):
        self._managers: list = []

    def teardown_method(self):
        for m in self._managers:
            m.close()
        self._managers.clear()

    def _sm(self, backends: dict) -> StorageManager:
        sm = make_storage_manager_with_backends(backends)
        self._managers.append(sm)
        return sm

    def _keys(self, n: int) -> List[CacheEngineKey]:
        return [make_key(i) for i in range(n)]

    # 1. Prefix fallback when batched_contains_gaps returns None
    def test_fallback_to_prefix_when_gaps_returns_none(self):
        """PrefixOnlyBackend (gaps=None) → prefix-only, furthest_hit_end=3."""
        backend = PrefixOnlyBackend(present_ids={0, 1, 2, 4})
        sm = self._sm({"Tier0": backend})
        keys = self._keys(5)

        furthest, chunk_gaps, bmap = sm.batched_contains_with_gaps(keys)

        assert furthest == 3
        assert chunk_gaps == []  # prefix-only: no gap intervals reported
        assert "Tier0" in bmap
        assert len(bmap["Tier0"]) == 3

    # 2. All hits from a single gap-aware backend
    def test_all_hits_single_gap_aware_backend(self):
        """GapAwareBackend with all keys present → furthest_hit_end=5, no gaps."""
        backend = GapAwareBackend(present_ids={0, 1, 2, 3, 4})
        sm = self._sm({"Tier0": backend})
        keys = self._keys(5)

        furthest, chunk_gaps, bmap = sm.batched_contains_with_gaps(keys)

        assert furthest == 5
        assert chunk_gaps == []
        assert bmap["Tier0"] == keys

    # 3. Gap in the middle
    def test_gap_in_middle(self):
        """GapAwareBackend missing key[2] → gap at (2,3), furthest_hit_end=5."""
        backend = GapAwareBackend(present_ids={0, 1, 3, 4})
        sm = self._sm({"Tier0": backend})
        keys = self._keys(5)

        furthest, chunk_gaps, bmap = sm.batched_contains_with_gaps(keys)

        assert furthest == 5
        assert chunk_gaps == [(2, 3)]
        hit_keys = bmap["Tier0"]
        assert len(hit_keys) == 4
        assert keys[0] in hit_keys
        assert keys[1] in hit_keys
        assert keys[3] in hit_keys
        assert keys[4] in hit_keys
        assert keys[2] not in hit_keys

    # 4. Truncation triggers suffix cascade to next tier
    def test_truncation_triggers_suffix_cascade(self):
        """TruncatingBackend(max=3) with {0-4}, then PrefixOnlyBackend({3-6}), 7 keys."""
        tier0 = TruncatingBackend(present_ids=set(range(5)), max_keys=3)
        tier1 = PrefixOnlyBackend(present_ids={3, 4, 5, 6})
        sm = self._sm({"Tier0": tier0, "Tier1": tier1})
        keys = self._keys(7)

        furthest, chunk_gaps, bmap = sm.batched_contains_with_gaps(keys)

        assert furthest == 7
        assert chunk_gaps == []
        assert len(bmap["Tier0"]) == 3
        assert len(bmap["Tier1"]) == 4

    # 5. Truncation with gaps, then suffix cascade
    def test_truncation_with_gaps_then_suffix_cascade(self):
        """TruncatingBackend(present={0,2}, max=3), then PrefixOnlyBackend({3,4}), 5 keys."""
        tier0 = TruncatingBackend(present_ids={0, 2}, max_keys=3)
        tier1 = PrefixOnlyBackend(present_ids={3, 4})
        sm = self._sm({"Tier0": tier0, "Tier1": tier1})
        keys = self._keys(5)

        furthest, chunk_gaps, bmap = sm.batched_contains_with_gaps(keys)

        assert furthest == 5
        assert chunk_gaps == [(1, 2)]

    # 6. No hits anywhere
    def test_no_hits_anywhere(self):
        """GapAwareBackend(empty) → furthest_hit_end=0, no chunk_gaps."""
        backend = GapAwareBackend(present_ids=set())
        sm = self._sm({"Tier0": backend})
        keys = self._keys(3)

        furthest, chunk_gaps, bmap = sm.batched_contains_with_gaps(keys)

        assert furthest == 0
        assert chunk_gaps == []
        assert bmap == {}

    # 7. Empty keys list
    def test_empty_keys(self):
        backend = GapAwareBackend(present_ids={0, 1})
        sm = self._sm({"Tier0": backend})

        result = sm.batched_contains_with_gaps([])

        assert result == (0, [], {})

    # 8. PDBackend pin special-casing
    def test_pdbackend_pin_disabled(self):
        """When backend is named 'PDBackend', pin arg must be forced to False."""
        backend = TrackingGapBackend(present_ids={0, 1, 2})
        sm = self._sm({"PDBackend": backend})
        keys = self._keys(3)

        sm.batched_contains_with_gaps(keys, pin=True)

        assert backend.gaps_pin_arg is False

    # 9. Prefix fallback cascades suffix across two prefix-only tiers
    def test_prefix_fallback_cascades_suffix(self):
        """Two PrefixOnlyBackends: {0,1} then {2,3,4}, 5 keys → all hits."""
        tier0 = PrefixOnlyBackend(present_ids={0, 1})
        tier1 = PrefixOnlyBackend(present_ids={2, 3, 4})
        sm = self._sm({"Tier0": tier0, "Tier1": tier1})
        keys = self._keys(5)

        furthest, chunk_gaps, bmap = sm.batched_contains_with_gaps(keys)

        assert furthest == 5
        assert chunk_gaps == []
        assert len(bmap["Tier0"]) == 2
        assert len(bmap["Tier1"]) == 3

    # 10. Gap-aware all miss does NOT cascade to next tier
    def test_gap_aware_all_miss_does_not_cascade(self):
        """GapAwareBackend(empty) returns ([(0,3)], 3) — offset advances by 3.
        No remaining suffix → Tier1 not consulted.
        """
        tier0 = GapAwareBackend(present_ids=set())
        tier1 = PrefixOnlyBackend(present_ids={0, 1, 2})
        sm = self._sm({"Tier0": tier0, "Tier1": tier1})
        keys = self._keys(3)

        furthest, chunk_gaps, bmap = sm.batched_contains_with_gaps(keys)

        assert furthest == 0
        assert chunk_gaps == []
        assert bmap == {}
        assert tier1.was_consulted is False

    # 11. Trailing misses stripped from chunk_gaps
    def test_trailing_misses_not_in_chunk_gaps(self):
        """GapAwareBackend {0,1} out of 5: ([(2,5)],5) → strip (2,5) since > furthest_hit=2."""
        backend = GapAwareBackend(present_ids={0, 1})
        sm = self._sm({"Tier0": backend})
        keys = self._keys(5)

        furthest, chunk_gaps, bmap = sm.batched_contains_with_gaps(keys)

        assert furthest == 2
        assert chunk_gaps == []  # trailing absences are NOT gaps

    # 12. Gap-aware then prefix cascade
    def test_gap_aware_then_prefix_cascade(self):
        """TruncatingBackend(present={0,2,3}, max=4) + PrefixOnlyBackend({4,5}), 6 keys."""
        tier0 = TruncatingBackend(present_ids={0, 2, 3}, max_keys=4)
        tier1 = PrefixOnlyBackend(present_ids={4, 5})
        sm = self._sm({"Tier0": tier0, "Tier1": tier1})
        keys = self._keys(6)

        furthest, chunk_gaps, bmap = sm.batched_contains_with_gaps(keys)

        assert furthest == 6
        assert chunk_gaps == [(1, 2)]
