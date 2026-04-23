# SPDX-License-Identifier: Apache-2.0
"""Tests for lookup_with_gaps() on the lookup client interface and implementations."""

# Standard
import os
import random
import time
import uuid
import pytest

# Third Party
import torch

# First Party
from lmcache.utils import mock_up_broadcast_fn, mock_up_broadcast_object_fn
from lmcache.v1.cache_engine import LMCacheEngineBuilder
from lmcache.v1.gpu_connector.mock_gpu_connector import MockGPUConnector
from lmcache.v1.lookup_client.abstract_client import LookupClientInterface
from lmcache.v1.lookup_client.factory import LookupClientFactory
from lmcache.v1.lookup_client.lmcache_lookup_client import (
    LMCacheLookupClient,
    LMCacheLookupServer,
)
from tests.v1.utils import (
    create_test_config,
    create_test_metadata,
    generate_kv_cache_paged_list_tensors,
    generate_tokens,
    recover_engine_states,
)


class ConcreteClient(LookupClientInterface):
    """Minimal concrete implementation for testing the default."""

    def __init__(self, lookup_result):
        self._lookup_result = lookup_result

    def lookup(self, token_ids, lookup_id, request_configs=None):
        return self._lookup_result

    def close(self):
        pass


class TestLookupWithGapsDefault:
    """The default implementation wraps lookup() and returns empty gaps."""

    def test_default_wraps_lookup_with_empty_gaps(self):
        client = ConcreteClient(lookup_result=512)
        result = client.lookup_with_gaps([1, 2, 3], "req-1")
        assert result == (512, [])

    def test_default_returns_none_when_lookup_returns_none(self):
        """Async clients return None from lookup(). lookup_with_gaps() propagates it."""
        client = ConcreteClient(lookup_result=None)
        result = client.lookup_with_gaps([1, 2, 3], "req-async")
        assert result is None

    def test_default_returns_zero_on_miss(self):
        client = ConcreteClient(lookup_result=0)
        result = client.lookup_with_gaps([1, 2, 3], "req-miss")
        assert result == (0, [])

    def test_default_passes_request_configs(self):
        """lookup_with_gaps should pass request_configs through to lookup()."""
        received_configs = {}

        class TrackingClient(LookupClientInterface):
            def lookup(self, token_ids, lookup_id, request_configs=None):
                received_configs["val"] = request_configs
                return 256

            def close(self):
                pass

        client = TrackingClient()
        client.lookup_with_gaps([1, 2, 3], "req-1", {"key": "val"})
        assert received_configs["val"] == {"key": "val"}


# ---------------------------------------------------------------------------
# Integration tests: LMCacheLookupClient + LMCacheLookupServer
# ---------------------------------------------------------------------------

CHUNK_SIZE = 256


@pytest.fixture
def lmcache_engine():
    instance_id = f"test_gaps_client_{uuid.uuid4().hex[:8]}"
    config = create_test_config(
        chunk_size=CHUNK_SIZE,
        local_cpu=True,
        instance_id=instance_id,
    )
    metadata = create_test_metadata()
    connector = MockGPUConnector(kv_shape=(4, 2, CHUNK_SIZE, 8, 128))

    engine = LMCacheEngineBuilder.get_or_create(
        instance_id=instance_id,
        config=config,
        metadata=metadata,
        gpu_connector=connector,
        broadcast_fn=mock_up_broadcast_fn,
        broadcast_object_fn=mock_up_broadcast_object_fn,
    )
    engine.post_init()
    yield engine

    engine.close()
    LMCacheEngineBuilder._instances.pop(instance_id, None)
    LMCacheEngineBuilder._cfgs.pop(instance_id, None)
    LMCacheEngineBuilder._metadatas.pop(instance_id, None)
    LMCacheEngineBuilder._stat_loggers.pop(instance_id, None)


def make_server(engine):
    transport = LookupClientFactory._create_zmq_server_transport(engine.metadata)
    return LMCacheLookupServer(engine, engine.metadata, transport)


def make_client(engine):
    transport = LookupClientFactory._create_zmq_client_transport(
        engine.config, engine.metadata
    )
    return LMCacheLookupClient(engine.config, engine.metadata, transport)


def store_tokens(engine, tokens):
    num_blocks = 500
    block_size = 16
    kv_cache = generate_kv_cache_paged_list_tensors(num_blocks, "cpu", block_size)
    slot_mapping = random.sample(range(0, num_blocks * block_size), len(tokens))
    slot_mapping = torch.tensor(slot_mapping)
    engine.store(tokens=tokens, kvcaches=kv_cache, slot_mapping=slot_mapping)
    recover_engine_states(engine)
    time.sleep(0.3)


class TestLMCacheLookupClientWithGaps:
    """Integration tests: sync client gap-aware lookup via real ZMQ transport."""

    pytestmark = pytest.mark.skipif(
        os.environ.get("PYTHONHASHSEED") is None,
        reason=(
            "PYTHONHASHSEED must be set for consistent hashing between "
            "LMCacheLookupClient and LMCacheLookupServer. "
            "Run with: PYTHONHASHSEED=0 pytest ..."
        ),
    )

    def test_lookup_with_gaps_full_prefix_hit(self, lmcache_engine):
        """Store all chunks, lookup returns (total_tokens, [])."""
        tokens = generate_tokens(CHUNK_SIZE * 3, "cpu", fixed=True)
        store_tokens(lmcache_engine, tokens)

        with make_server(lmcache_engine):
            time.sleep(0.3)
            with make_client(lmcache_engine) as client:
                result = client.lookup_with_gaps(tokens.tolist(), "req-full")
                assert result is not None
                num_hit, gaps = result
                assert num_hit == CHUNK_SIZE * 3
                assert gaps == []

    def test_lookup_with_gaps_no_data_all_miss(self, lmcache_engine):
        """Nothing stored — lookup returns (0, [])."""
        tokens = generate_tokens(CHUNK_SIZE * 3, "cpu", fixed=False)

        with make_server(lmcache_engine):
            time.sleep(0.3)
            with make_client(lmcache_engine) as client:
                result = client.lookup_with_gaps(tokens.tolist(), "req-miss")
                assert result == (0, [])

    def test_lookup_with_gaps_prefix_only_backend_returns_empty_gaps(
        self, lmcache_engine
    ):
        """LocalCPUBackend uses prefix semantics (batched_contains_mask returns None).
        lookup_with_gaps() should return (prefix_count, []) — no gaps reported."""
        # Store first 2 chunks only
        all_tokens = generate_tokens(CHUNK_SIZE * 5, "cpu", fixed=True)
        partial_tokens = all_tokens[: CHUNK_SIZE * 2]
        store_tokens(lmcache_engine, partial_tokens)

        with make_server(lmcache_engine):
            time.sleep(0.3)
            with make_client(lmcache_engine) as client:
                result = client.lookup_with_gaps(all_tokens.tolist(), "req-prefix")
                assert result is not None
                num_hit, gaps = result
                # Prefix match: only 2 chunks hit, no gaps (3rd chunk not present)
                assert num_hit == CHUNK_SIZE * 2
                assert gaps == []

    def test_lookup_with_gaps_caches_result(self, lmcache_engine):
        """lookup_cache() works after lookup_with_gaps()."""
        tokens = generate_tokens(CHUNK_SIZE, "cpu", fixed=True)
        store_tokens(lmcache_engine, tokens)

        with make_server(lmcache_engine):
            time.sleep(0.3)
            with make_client(lmcache_engine) as client:
                client.lookup_with_gaps(tokens.tolist(), "req-cache-check")
                cached = client.lookup_cache("req-cache-check")
                assert cached == CHUNK_SIZE

    def test_lookup_with_gaps_empty_tokens(self, lmcache_engine):
        """Empty token list returns (0, [])."""
        with make_server(lmcache_engine):
            time.sleep(0.3)
            with make_client(lmcache_engine) as client:
                result = client.lookup_with_gaps([], "req-empty")
                assert result == (0, [])
