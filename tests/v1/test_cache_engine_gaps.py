# SPDX-License-Identifier: Apache-2.0
"""Tests for LMCacheEngine.lookup_with_gaps().

Mocks storage_manager.batched_contains_with_gaps() to control the hit/gap
pattern and verifies that lookup_with_gaps() correctly:
  - Sets res to the token count of the furthest hit chunk
  - Emits merged token-space gap intervals (consecutive chunk-gaps → one interval)
  - Handles pinning, prefix-only fallback, and layerwise fallback

Run with: PYTHONHASHSEED=0 pytest tests/v1/test_cache_engine_gaps.py -v
"""

# Standard
import uuid
from unittest.mock import patch

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import mock_up_broadcast_fn, mock_up_broadcast_object_fn
from lmcache.v1.cache_engine import LMCacheEngineBuilder
from lmcache.v1.gpu_connector.mock_gpu_connector import MockGPUConnector

from tests.v1.utils import (
    create_test_config,
    create_test_metadata,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CHUNK_SIZE = 256


@pytest.fixture
def engine():
    """CPU-backed engine with 5-chunk capacity (1280 tokens)."""
    instance_id = f"test_gaps_{uuid.uuid4().hex[:8]}"
    config = create_test_config(
        chunk_size=CHUNK_SIZE,
        local_cpu=True,
        instance_id=instance_id,
    )
    metadata = create_test_metadata()
    connector = MockGPUConnector(kv_shape=(4, 2, CHUNK_SIZE, 8, 128))

    eng = LMCacheEngineBuilder.get_or_create(
        instance_id=instance_id,
        config=config,
        metadata=metadata,
        gpu_connector=connector,
        broadcast_fn=mock_up_broadcast_fn,
        broadcast_object_fn=mock_up_broadcast_object_fn,
    )
    eng.post_init()
    yield eng

    eng.close()
    LMCacheEngineBuilder._instances.pop(instance_id, None)
    LMCacheEngineBuilder._cfgs.pop(instance_id, None)
    LMCacheEngineBuilder._metadatas.pop(instance_id, None)
    LMCacheEngineBuilder._stat_loggers.pop(instance_id, None)


def make_tokens(num_chunks: int) -> list[int]:
    """Return a fixed token list of num_chunks * CHUNK_SIZE tokens."""
    return list(range(num_chunks * CHUNK_SIZE))


# ---------------------------------------------------------------------------
# Helper: build the expected block_mapping stub
# ---------------------------------------------------------------------------

def _stub_block_mapping(chunk_gaps: list, furthest_hit_end: int) -> dict:
    """Minimal block_mapping stub — real keys not needed for these tests."""
    gap_size = sum(e - s for s, e in chunk_gaps)
    num_hits = furthest_hit_end - gap_size
    if num_hits > 0:
        return {"LocalCPUBackend": ["stub_key"] * num_hits}
    return {}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLookupWithGapsGapBuilding:
    """Tests for gap-building logic: (furthest_hit_end, chunk_gaps) → (res, token_gaps)."""

    def _call(
        self,
        engine,
        num_chunks: int,
        furthest_hit_end: int,
        chunk_gaps: list,
    ):
        """Mock batched_contains_with_gaps and call lookup_with_gaps."""
        tokens = make_tokens(num_chunks)
        block_mapping = _stub_block_mapping(chunk_gaps, furthest_hit_end)

        with patch.object(
            engine.storage_manager,
            "batched_contains_with_gaps",
            return_value=(furthest_hit_end, chunk_gaps, block_mapping),
        ):
            return engine.lookup_with_gaps(tokens)

    def test_no_hits_returns_zero_empty_gaps(self, engine):
        """All misses → (0, [])."""
        res, gaps = self._call(engine, 5, 0, [])
        assert res == 0
        assert gaps == []

    def test_all_hits_no_gaps(self, engine):
        """Full prefix hit → (1280, [])."""
        res, gaps = self._call(engine, 5, 5, [])
        assert res == 5 * CHUNK_SIZE
        assert gaps == []

    def test_gap_in_middle(self, engine):
        """Chunks 0,1 hit; chunk 2 gap; chunks 3,4 hit → (1280, [(512, 768)])."""
        res, gaps = self._call(engine, 5, 5, [(2, 3)])
        assert res == 5 * CHUNK_SIZE
        assert gaps == [(2 * CHUNK_SIZE, 3 * CHUNK_SIZE)]

    def test_multiple_gaps(self, engine):
        """Chunks 0,2,4 hit; chunks 1,3 missing → (1280, [(256,512), (768,1024)])."""
        res, gaps = self._call(engine, 5, 5, [(1, 2), (3, 4)])
        assert res == 5 * CHUNK_SIZE
        assert gaps == [
            (1 * CHUNK_SIZE, 2 * CHUNK_SIZE),
            (3 * CHUNK_SIZE, 4 * CHUNK_SIZE),
        ]

    def test_consecutive_gaps_merged_into_one_token_interval(self, engine):
        """Chunks 1,2 both missing → single merged token interval (256, 768).

        chunk_gaps=[(1,3)] means chunks[1] and chunks[2] are absent.
        These are consecutive, so the token intervals merge:
        (chunks[1].start, chunks[2].end) = (256, 768).
        """
        res, gaps = self._call(engine, 5, 5, [(1, 3)])
        assert res == 5 * CHUNK_SIZE
        # One merged interval: tokens 256..768
        assert gaps == [(1 * CHUNK_SIZE, 3 * CHUNK_SIZE)]

    def test_trailing_misses_not_counted_as_gaps(self, engine):
        """Chunks 0,1,2 hit; chunks 3,4 miss → (768, []).
        Trailing misses are not gaps — they are simply uncached tokens
        that vLLM's normal prefill handles.
        """
        res, gaps = self._call(engine, 5, 3, [])
        assert res == 3 * CHUNK_SIZE
        assert gaps == []

    def test_leading_miss_no_hits(self, engine):
        """First chunk misses, rest present — but since furthest_hit_end = 0,
        we get (0, []) not a gap."""
        # A prefix-only backend would report furthest_hit_end=0 here.
        # A gap-aware backend would report furthest_hit_end=5 with a leading gap.
        # This tests the prefix-only (furthest_hit_end=0) case.
        res, gaps = self._call(engine, 5, 0, [])
        assert res == 0
        assert gaps == []

    def test_leading_gap_then_hits(self, engine):
        """Gap-aware backend reports chunk 0 missing, chunks 1-4 hit.
        furthest_hit_end = 5, so chunk 0 is a gap.
        """
        res, gaps = self._call(engine, 5, 5, [(0, 1)])
        assert res == 5 * CHUNK_SIZE
        assert gaps == [(0, CHUNK_SIZE)]

    def test_single_chunk_hit(self, engine):
        """Single chunk, all hit → (256, [])."""
        res, gaps = self._call(engine, 1, 1, [])
        assert res == CHUNK_SIZE
        assert gaps == []

    def test_single_chunk_miss(self, engine):
        """Single chunk, miss → (0, [])."""
        res, gaps = self._call(engine, 1, 0, [])
        assert res == 0
        assert gaps == []

    def test_partial_prefix_then_gap_then_hit(self, engine):
        """Chunks 0 hit, chunk 1 gap, chunk 2 hit — gap in the middle."""
        # furthest_hit_end = 3 (last hit is chunk 2)
        res, gaps = self._call(engine, 5, 3, [(1, 2)])
        assert res == 3 * CHUNK_SIZE
        assert gaps == [(CHUNK_SIZE, 2 * CHUNK_SIZE)]


class TestLookupWithGapsPinning:
    """Tests that lookup_pins is set correctly when pin=True."""

    def test_pin_sets_lookup_pins(self, engine):
        """When pin=True, lookup_pins[lookup_id] = block_mapping."""
        tokens = make_tokens(3)
        chunk_gaps = [(1, 2)]
        # Two hit keys — non-contiguous (chunks 0 and 2 hit)
        hit_keys = ["key0", "key2"]
        block_mapping = {"LocalCPUBackend": hit_keys}

        with patch.object(
            engine.storage_manager,
            "batched_contains_with_gaps",
            return_value=(3, chunk_gaps, block_mapping),
        ):
            engine.lookup_with_gaps(tokens, lookup_id="req-1", pin=True)

        assert engine.lookup_pins["req-1"] == block_mapping

    def test_no_pin_does_not_set_lookup_pins(self, engine):
        """When pin=False, lookup_pins is unchanged."""
        tokens = make_tokens(3)
        block_mapping = {"LocalCPUBackend": ["k0", "k1", "k2"]}
        pins_before = dict(engine.lookup_pins)

        with patch.object(
            engine.storage_manager,
            "batched_contains_with_gaps",
            return_value=(3, [], block_mapping),
        ):
            engine.lookup_with_gaps(tokens, pin=False)

        assert engine.lookup_pins == pins_before

    def test_pin_no_hits_does_not_set_lookup_pins(self, engine):
        """When pin=True but no hits, lookup_pins is NOT set."""
        tokens = make_tokens(3)
        with patch.object(
            engine.storage_manager,
            "batched_contains_with_gaps",
            return_value=(0, [], {}),
        ):
            engine.lookup_with_gaps(tokens, lookup_id="req-no-hits", pin=True)

        assert "req-no-hits" not in engine.lookup_pins


class TestLookupWithGapsFallback:
    """Verify lookup_with_gaps() degrades gracefully to prefix-only behavior
    when the storage layer reports no gaps (as with prefix-only backends).
    """

    def test_prefix_only_backend_no_gaps(self, engine):
        """When batched_contains_with_gaps returns prefix-only results (no gaps),
        lookup_with_gaps() behaves identically to lookup()."""
        tokens = make_tokens(5)
        # Prefix-only: first 3 chunks hit, no gaps
        block_mapping = {"LocalCPUBackend": ["k0", "k1", "k2"]}

        with patch.object(
            engine.storage_manager,
            "batched_contains_with_gaps",
            return_value=(3, [], block_mapping),
        ):
            res, gaps = engine.lookup_with_gaps(tokens)

        assert res == 3 * CHUNK_SIZE
        assert gaps == []


class TestLookupWithGapsLayerwise:
    """Smoke test for the layerwise fallback path in lookup_with_gaps()."""

    def test_layerwise_falls_back_to_prefix_only(self, engine):
        """When use_layerwise=True, lookup_with_gaps() falls back to prefix-only
        behavior and returns (res, []) — same as lookup().
        This tests the fallback, NOT gap support (which is deferred for layerwise)."""
        tokens = make_tokens(3)

        # Force use_layerwise=True on the engine instance
        engine.use_layerwise = True
        try:
            # Mock batched_contains to return a prefix hit of 2 chunks
            with patch.object(
                engine.storage_manager,
                "batched_contains",
                return_value=(engine.num_layers, {"LocalCPUBackend": []}),
            ):
                res, gaps = engine.lookup_with_gaps(tokens)
        finally:
            engine.use_layerwise = False

        # Gaps must always be [] in layerwise fallback path
        assert gaps == []
