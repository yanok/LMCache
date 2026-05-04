# SPDX-License-Identifier: Apache-2.0
"""Tests for NixlDynamicStorageBackend.batched_contains_gaps()."""
import sys
import threading
from unittest.mock import MagicMock

# nixl is a native NIXL library that is not available in CI; stub it out before
# importing any lmcache module that depends on it.
if "nixl" not in sys.modules:
    _nixl_stub = MagicMock()
    sys.modules["nixl"] = _nixl_stub
    sys.modules["nixl._api"] = _nixl_stub

import pytest
import torch

from lmcache.utils import CacheEngineKey
from lmcache.v1.storage_backend.nixl_storage_backend import NixlDynamicStorageBackend


def make_key(i: int) -> CacheEngineKey:
    return CacheEngineKey(
        model_name="test_model",
        world_size=1,
        worker_id=0,
        chunk_hash=i,
        dtype=torch.bfloat16,
    )


def make_backend(query_responses):
    """Return a NixlDynamicStorageBackend with mocked NIXL internals.

    query_responses: list returned by agent.nixl_agent.query_memory().
    Pass a list of values where None means key absent, anything else means present.
    """
    backend = object.__new__(NixlDynamicStorageBackend)
    agent = MagicMock()
    agent.backend = "DOCA_MEMOS"
    agent.mem_type = "OBJ"
    agent.nixl_agent = MagicMock()
    agent.nixl_agent.query_memory.return_value = query_responses
    backend.agent = agent
    backend._format_object_key = lambda key: f"key_{key.chunk_hash}"
    # Attributes required by batched_contains_gaps()
    backend.never_check_exists = False
    backend.enable_gap_detection = True
    backend.progress_lock = threading.RLock()
    backend.progress_set = set()
    return backend


class TestBatchedContainsGapsNixl:

    def test_empty_keys(self):
        backend = make_backend([])
        assert backend.batched_contains_gaps([]) == ([], 0)

    def test_all_hits(self):
        keys = [make_key(i) for i in range(4)]
        backend = make_backend(["d0", "d1", "d2", "d3"])
        gaps, end = backend.batched_contains_gaps(keys)
        assert gaps == []
        assert end == 4

    def test_all_misses_no_hits(self):
        # No hits → last_hit_end = 0 → no gaps reported, all cascade
        keys = [make_key(i) for i in range(4)]
        backend = make_backend([None, None, None, None])
        gaps, end = backend.batched_contains_gaps(keys)
        assert gaps == []
        assert end == 0

    def test_prefix_hits_trailing_misses_cascade(self):
        # keys[0,1] hit, keys[2,3] miss — trailing misses excluded; cascade via end=2
        keys = [make_key(i) for i in range(4)]
        backend = make_backend(["d0", "d1", None, None])
        gaps, end = backend.batched_contains_gaps(keys)
        assert gaps == []
        assert end == 2

    def test_single_gap_in_middle(self):
        # keys[0,1] hit, keys[2] miss (gap), keys[3] hit
        keys = [make_key(i) for i in range(4)]
        backend = make_backend(["d0", "d1", None, "d3"])
        gaps, end = backend.batched_contains_gaps(keys)
        assert gaps == [(2, 3)]
        assert end == 4

    def test_multiple_gaps(self):
        # hit, miss, hit, miss, hit
        keys = [make_key(i) for i in range(5)]
        backend = make_backend(["d0", None, "d2", None, "d4"])
        gaps, end = backend.batched_contains_gaps(keys)
        assert gaps == [(1, 2), (3, 4)]
        assert end == 5

    def test_consecutive_misses_merged_into_one_gap(self):
        # keys[0] hit, keys[1,2] miss (one gap), keys[3] hit
        keys = [make_key(i) for i in range(4)]
        backend = make_backend(["d0", None, None, "d3"])
        gaps, end = backend.batched_contains_gaps(keys)
        assert gaps == [(1, 3)]
        assert end == 4

    def test_gap_then_trailing_misses(self):
        # keys[0] hit, keys[1] gap, keys[2] hit, keys[3,4] trailing misses (cascade)
        keys = [make_key(i) for i in range(5)]
        backend = make_backend(["d0", None, "d2", None, None])
        gaps, end = backend.batched_contains_gaps(keys)
        assert gaps == [(1, 2)]
        assert end == 3

    def test_query_exception_returns_none(self):
        # On NIXL error, return None so StorageManager falls back to prefix-only
        keys = [make_key(i) for i in range(3)]
        backend = make_backend(None)
        backend.agent.nixl_agent.query_memory.side_effect = RuntimeError("NIXL error")
        assert backend.batched_contains_gaps(keys) is None

    def test_single_batch_call(self):
        # All N keys go in one query_memory call, not N separate calls
        keys = [make_key(i) for i in range(4)]
        backend = make_backend(["d0", "d1", "d2", "d3"])
        backend.batched_contains_gaps(keys)
        backend.agent.nixl_agent.query_memory.assert_called_once()
        reg_list = backend.agent.nixl_agent.query_memory.call_args[0][0]
        assert len(reg_list) == 4

    def test_descriptor_format(self):
        # Each descriptor is (0, 0, 0, formatted_key)
        keys = [make_key(7), make_key(42)]
        backend = make_backend(["d7", "d42"])
        backend.batched_contains_gaps(keys)
        reg_list = backend.agent.nixl_agent.query_memory.call_args[0][0]
        assert reg_list[0] == (0, 0, 0, "key_7")
        assert reg_list[1] == (0, 0, 0, "key_42")

    def test_never_check_exists_returns_none(self):
        # When never_check_exists=True, method must return None immediately (no NIXL call)
        keys = [make_key(i) for i in range(3)]
        backend = make_backend(["d0", "d1", "d2"])
        backend.never_check_exists = True
        result = backend.batched_contains_gaps(keys)
        assert result is None
        backend.agent.nixl_agent.query_memory.assert_not_called()

    def test_gap_detection_disabled_returns_none(self):
        # When enable_gap_detection=False (default), returns None immediately — no NIXL call
        keys = [make_key(i) for i in range(3)]
        backend = make_backend(["d0", "d1", "d2"])
        backend.enable_gap_detection = False
        result = backend.batched_contains_gaps(keys)
        assert result is None
        backend.agent.nixl_agent.query_memory.assert_not_called()

    def test_in_progress_keys_are_misses(self):
        # Keys whose async PUT is still in flight must be treated as misses,
        # matching the behaviour of contains() which calls exists_in_put_tasks().
        # keys[0] hit, keys[1] in-progress (demoted to miss / gap), keys[2] hit
        keys = [make_key(i) for i in range(3)]
        backend = make_backend(["d0", "d1", "d2"])
        # Put keys[1] in the in-progress set to simulate an in-flight PUT
        with backend.progress_lock:
            backend.progress_set.add(keys[1])
        gaps, end = backend.batched_contains_gaps(keys)
        # keys[1] is in-progress → treated as a miss (gap) between keys[0] and keys[2]
        assert gaps == [(1, 2)]
        assert end == 3
