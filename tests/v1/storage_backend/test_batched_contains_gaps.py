# SPDX-License-Identifier: Apache-2.0
"""Tests for batched_contains_gaps() on StorageBackendInterface."""

from typing import List, Tuple

import torch
import pytest

from lmcache.utils import CacheEngineKey
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface


def make_key(chunk_id: int) -> CacheEngineKey:
    return CacheEngineKey(
        model_name="test_model",
        world_size=1,
        worker_id=0,
        chunk_hash=chunk_id,
        dtype=torch.bfloat16,
    )


class StubBackend(StorageBackendInterface):
    """Minimal concrete backend — does NOT override batched_contains_gaps."""

    def __init__(self, present_ids: set[int]):
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


class GapAwareStubBackend(StubBackend):
    """Backend that overrides batched_contains_gaps() to return gap intervals."""

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


class TestBatchedContainsGapsDefault:
    """Base class default returns None (= 'I don't support gap reporting')."""

    def test_returns_none_by_default(self):
        backend = StubBackend(present_ids={0, 1, 2})
        result = backend.batched_contains_gaps([make_key(i) for i in range(3)])
        assert result is None

    def test_returns_none_with_pin(self):
        backend = StubBackend(present_ids={0, 1})
        result = backend.batched_contains_gaps([make_key(i) for i in range(2)], pin=True)
        assert result is None

    def test_returns_none_for_empty_keys(self):
        backend = StubBackend(present_ids=set())
        result = backend.batched_contains_gaps([])
        assert result is None

    def test_connector_base_also_returns_none(self):
        """RemoteConnector mirrors the same default."""
        from unittest.mock import MagicMock
        from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
        connector = MagicMock(spec=RemoteConnector)
        result = RemoteConnector.batched_contains_gaps(connector, [make_key(0)])
        assert result is None


class TestBatchedContainsGapsOverride:
    """A backend that overrides batched_contains_gaps() returns (gaps, end)."""

    def test_all_hits_returns_empty_gaps(self):
        backend = GapAwareStubBackend(present_ids={0, 1, 2})
        gaps, end = backend.batched_contains_gaps([make_key(i) for i in range(3)])
        assert gaps == []
        assert end == 3

    def test_single_gap_in_middle(self):
        """Keys 0,1,3,4 present; key 2 absent → gap at (2,3)."""
        backend = GapAwareStubBackend(present_ids={0, 1, 3, 4})
        gaps, end = backend.batched_contains_gaps([make_key(i) for i in range(5)])
        assert gaps == [(2, 3)]
        assert end == 5

    def test_multiple_non_consecutive_gaps(self):
        """Keys 0,2,4 present; keys 1,3 absent → two separate gaps."""
        backend = GapAwareStubBackend(present_ids={0, 2, 4})
        gaps, end = backend.batched_contains_gaps([make_key(i) for i in range(5)])
        assert gaps == [(1, 2), (3, 4)]
        assert end == 5

    def test_consecutive_gaps_merged_into_one(self):
        """Keys 1,2 both absent → single gap (1,3), not two entries."""
        backend = GapAwareStubBackend(present_ids={0, 3, 4})
        gaps, end = backend.batched_contains_gaps([make_key(i) for i in range(5)])
        assert gaps == [(1, 3)]
        assert end == 5

    def test_all_misses_single_full_gap(self):
        backend = GapAwareStubBackend(present_ids=set())
        gaps, end = backend.batched_contains_gaps([make_key(i) for i in range(3)])
        assert gaps == [(0, 3)]
        assert end == 3

    def test_trailing_misses_included_in_gap(self):
        """Keys 0,1 present; keys 2,3,4 absent → trailing gap (2,5)."""
        backend = GapAwareStubBackend(present_ids={0, 1})
        gaps, end = backend.batched_contains_gaps([make_key(i) for i in range(5)])
        assert gaps == [(2, 5)]
        assert end == 5

    def test_truncation_end_less_than_len_keys(self):
        """Backend may return end < len(keys) to signal truncation."""
        class TruncatingBackend(StubBackend):
            def batched_contains_gaps(self, keys, pin=False):
                # All 3 covered keys are present; end=3 < len(keys)=5 → truncation
                return ([], 3)

        backend = TruncatingBackend(present_ids={0, 1, 2, 3, 4})
        gaps, end = backend.batched_contains_gaps([make_key(i) for i in range(5)])
        assert gaps == []
        assert end == 3  # truncated; suffix cascades to next tier

    def test_empty_keys(self):
        backend = GapAwareStubBackend(present_ids={0, 1})
        gaps, end = backend.batched_contains_gaps([])
        assert gaps == []
        assert end == 0
