# lmcache/integration/vllm/tests/test_connector_plumbing.py
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for connector plumbing: LoadSpec.gaps, _req_to_gaps, get_computed_token_gaps.

These tests avoid instantiating LMCacheConnectorV1Impl fully (which requires
live vllm+lmcache infrastructure). Instead they construct stub instances via
object.__new__ and set the minimal state needed to exercise the methods under test.
"""

# Standard
import dataclasses
import sys
from typing import Optional
from unittest.mock import MagicMock

# Third Party
import pytest

# Stub out vllm and all submodules that vllm_v1_adapter imports at module level,
# so the test can run without a real vllm installation.
_VLLM_MODULES = [
    "vllm",
    "vllm.config",
    "vllm.distributed",
    "vllm.distributed.kv_transfer",
    "vllm.distributed.kv_transfer.kv_connector",
    "vllm.distributed.kv_transfer.kv_connector.v1",
    "vllm.distributed.kv_transfer.kv_connector.v1.base",
    "vllm.distributed.parallel_state",
    "vllm.sampling_params",
    "vllm.v1",
    "vllm.v1.core",
    "vllm.v1.core.sched",
    "vllm.v1.core.sched.output",
    "vllm.v1.request",
    "vllm.version",
    "uvicorn",
]
for _mod in _VLLM_MODULES:
    sys.modules.setdefault(_mod, MagicMock())

# vllm.version needs __version__ to be a real string, not a MagicMock attribute.
sys.modules["vllm.version"].__version__ = "0.0.0-stub"

# KVConnectorMetadata and KVConnectorBase_V1 are used as base classes in
# vllm_v1_adapter. They must be real Python classes for @dataclass and class
# inheritance machinery to work (both inspect __mro__).
class _KVConnectorMetadata:
    pass


class _KVConnectorBase_V1:
    pass


class _KVConnectorRole:
    pass


_base_mod = sys.modules["vllm.distributed.kv_transfer.kv_connector.v1.base"]
_base_mod.KVConnectorMetadata = _KVConnectorMetadata
_base_mod.KVConnectorBase_V1 = _KVConnectorBase_V1
_base_mod.KVConnectorRole = _KVConnectorRole

# First Party
from lmcache.integration.vllm.vllm_v1_adapter import LoadSpec  # noqa: E402


class TestLoadSpecGaps:
    """LoadSpec carries gap intervals alongside the token counts."""

    def test_default_gaps_is_empty_list(self):
        spec = LoadSpec(vllm_cached_tokens=0, lmcache_cached_tokens=256, can_load=False)
        assert spec.gaps == []

    def test_gaps_can_be_set_at_construction(self):
        gaps = [(256, 512), (768, 1024)]
        spec = LoadSpec(
            vllm_cached_tokens=0,
            lmcache_cached_tokens=1280,
            can_load=False,
            gaps=gaps,
        )
        assert spec.gaps == [(256, 512), (768, 1024)]

    def test_gaps_field_is_independent_per_instance(self):
        """Verify default_factory gives each instance its own list."""
        spec_a = LoadSpec(vllm_cached_tokens=0, lmcache_cached_tokens=256, can_load=False)
        spec_b = LoadSpec(vllm_cached_tokens=0, lmcache_cached_tokens=512, can_load=False)
        spec_a.gaps.append((0, 256))
        assert spec_b.gaps == []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl  # noqa: E402


@dataclasses.dataclass
class FakeRequest:
    """Minimal Request substitute for testing."""
    request_id: str
    all_token_ids: list
    num_tokens: int
    sampling_params: Optional[object] = None

    @property
    def priority(self):
        return 0


def make_stub_impl() -> LMCacheConnectorV1Impl:
    """Create a bare LMCacheConnectorV1Impl without invoking its __init__.

    Sets only the state fields that the methods under test read or write.
    lookup_client is a @property that delegates to self._manager.lookup_client,
    so we mock _manager to expose it.
    """
    impl = object.__new__(LMCacheConnectorV1Impl)
    impl._req_to_gaps = {}
    impl.load_specs = {}
    impl._requests_priority = {}
    impl.kv_role = "kv_consumer"
    impl.skip_last_n_tokens = 0
    impl.config = MagicMock()
    impl.config.min_retrieve_tokens = 0  # must be int, not MagicMock
    # lookup_client is a @property reading from self._manager.lookup_client
    impl._manager = MagicMock()
    impl._manager.lookup_client = MagicMock()
    return impl


# ---------------------------------------------------------------------------
# Tests for _req_to_gaps population
# ---------------------------------------------------------------------------


class TestReqToGapsPopulation:
    """get_num_new_matched_tokens() stores gap intervals in _req_to_gaps."""

    def test_gaps_stored_when_lookup_returns_gaps(self):
        """lookup_with_gaps returns (hit_count, gaps) → gaps stored in _req_to_gaps."""
        impl = make_stub_impl()
        gaps = [(256, 512)]
        impl._manager.lookup_client.lookup_cache.return_value = -1
        impl._manager.lookup_client.lookup_with_gaps.return_value = (768, gaps)

        request = FakeRequest(
            request_id="req-1",
            all_token_ids=list(range(1024)),
            num_tokens=1024,
        )
        result = impl.get_num_new_matched_tokens(request, num_computed_tokens=0)

        assert result is not None
        assert impl._req_to_gaps["req-1"] == gaps

    def test_gaps_stored_in_load_spec(self):
        """Gaps from lookup_with_gaps are stored in load_specs[req_id].gaps."""
        impl = make_stub_impl()
        gaps = [(512, 768)]
        impl._manager.lookup_client.lookup_cache.return_value = -1
        impl._manager.lookup_client.lookup_with_gaps.return_value = (1024, gaps)

        request = FakeRequest(
            request_id="req-2",
            all_token_ids=list(range(1280)),
            num_tokens=1280,
        )
        impl.get_num_new_matched_tokens(request, num_computed_tokens=0)

        assert impl.load_specs["req-2"].gaps == gaps

    def test_empty_gaps_when_lookup_returns_no_gaps(self):
        """Prefix-only hit returns empty gaps list."""
        impl = make_stub_impl()
        impl._manager.lookup_client.lookup_cache.return_value = -1
        impl._manager.lookup_client.lookup_with_gaps.return_value = (512, [])

        request = FakeRequest(
            request_id="req-3",
            all_token_ids=list(range(768)),
            num_tokens=768,
        )
        impl.get_num_new_matched_tokens(request, num_computed_tokens=0)

        assert impl._req_to_gaps["req-3"] == []

    def test_none_from_async_lookup_does_not_store_gaps(self):
        """When lookup_with_gaps returns None (async in progress), _req_to_gaps is unchanged."""
        impl = make_stub_impl()
        impl._manager.lookup_client.lookup_cache.return_value = -1
        impl._manager.lookup_client.lookup_with_gaps.return_value = None

        request = FakeRequest(
            request_id="req-async",
            all_token_ids=list(range(512)),
            num_tokens=512,
        )
        result = impl.get_num_new_matched_tokens(request, num_computed_tokens=0)

        assert result is None
        assert "req-async" not in impl._req_to_gaps

    def test_cached_path_preserves_existing_gaps(self):
        """When lookup_cache() returns a cached hit count, _req_to_gaps is unchanged.
        Gaps from the original lookup_with_gaps() call are preserved.
        """
        impl = make_stub_impl()
        # Simulate prior lookup stored gaps
        impl._req_to_gaps["req-cached"] = [(256, 512)]
        # Cached path returns hit count directly
        impl._manager.lookup_client.lookup_cache.return_value = 768

        request = FakeRequest(
            request_id="req-cached",
            all_token_ids=list(range(1024)),
            num_tokens=1024,
        )
        impl.get_num_new_matched_tokens(request, num_computed_tokens=0)

        # lookup_with_gaps should NOT have been called
        impl._manager.lookup_client.lookup_with_gaps.assert_not_called()
        # gaps preserved from original lookup
        assert impl._req_to_gaps["req-cached"] == [(256, 512)]

    def test_finished_request_gaps_cleared_in_build_connector_meta(self):
        """build_connector_meta() pops finished-request entries from _req_to_gaps."""
        impl = make_stub_impl()
        impl._req_to_gaps["req-done"] = [(256, 512)]
        impl._request_trackers = {}
        impl._unfinished_requests = {}
        impl._requests_priority = {}
        impl.load_specs = {}
        impl.force_skip_save = False
        impl.kv_role = "kv_consumer"
        impl.config.priority_limit = None

        scheduler_output = MagicMock()
        scheduler_output.finished_req_ids = ["req-done"]
        scheduler_output.scheduled_new_reqs = []
        # Simulate the newer CachedRequestData object (not a list)
        cached_reqs = MagicMock()
        cached_reqs.req_ids = []
        cached_reqs.new_block_ids = []
        scheduler_output.scheduled_cached_reqs = cached_reqs

        impl.build_connector_meta(scheduler_output)

        assert "req-done" not in impl._req_to_gaps


class TestGetComputedTokenGaps:
    """get_computed_token_gaps() reads from _req_to_gaps."""

    def test_returns_gaps_for_known_request(self):
        impl = make_stub_impl()
        impl._req_to_gaps["req-x"] = [(256, 512), (768, 1024)]
        request = FakeRequest(request_id="req-x", all_token_ids=[], num_tokens=1280)

        gaps = impl.get_computed_token_gaps(request)

        assert gaps == [(256, 512), (768, 1024)]

    def test_returns_none_for_unknown_request(self):
        impl = make_stub_impl()
        request = FakeRequest(request_id="req-unknown", all_token_ids=[], num_tokens=256)

        gaps = impl.get_computed_token_gaps(request)

        assert gaps is None

    def test_returns_empty_list_when_no_gaps(self):
        impl = make_stub_impl()
        impl._req_to_gaps["req-y"] = []
        request = FakeRequest(request_id="req-y", all_token_ids=[], num_tokens=512)

        gaps = impl.get_computed_token_gaps(request)

        assert gaps == []


# ---------------------------------------------------------------------------
# Tests for Layers 5+6: metadata propagation and gap masking
# ---------------------------------------------------------------------------

import torch  # noqa: E402 (must come after sys.modules patching at top of file)
from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorMetadata  # noqa: E402


class TestLMCacheConnectorMetadataGaps:
    """LMCacheConnectorMetadata carries req_to_gaps from scheduler to worker."""

    def test_req_to_gaps_field_exists_and_defaults_empty(self):
        meta = LMCacheConnectorMetadata()
        assert meta.req_to_gaps == {}

    def test_req_to_gaps_is_independent_per_instance(self):
        meta_a = LMCacheConnectorMetadata()
        meta_b = LMCacheConnectorMetadata()
        meta_a.req_to_gaps["req-1"] = [(0, 256)]
        assert meta_b.req_to_gaps == {}

    def test_build_connector_meta_copies_req_to_gaps(self):
        """build_connector_meta() copies _req_to_gaps into metadata."""
        impl = make_stub_impl()
        impl._req_to_gaps["req-a"] = [(256, 512)]
        impl._req_to_gaps["req-b"] = []
        impl._request_trackers = {}
        impl._unfinished_requests = {}
        impl._requests_priority = {}
        impl.load_specs = {}
        impl.force_skip_save = False
        impl.kv_role = "kv_consumer"
        impl.config.priority_limit = None

        scheduler_output = MagicMock()
        scheduler_output.finished_req_ids = []
        scheduler_output.scheduled_new_reqs = []
        cached_reqs = MagicMock()
        cached_reqs.req_ids = []
        cached_reqs.new_block_ids = []
        scheduler_output.scheduled_cached_reqs = cached_reqs

        meta = impl.build_connector_meta(scheduler_output)

        assert isinstance(meta, LMCacheConnectorMetadata)
        assert meta.req_to_gaps["req-a"] == [(256, 512)]
        assert meta.req_to_gaps["req-b"] == []

    def test_build_connector_meta_does_not_clear_req_to_gaps(self):
        """_req_to_gaps is NOT cleared by build_connector_meta().
        This preserves gaps for preempted requests that hit the lookup_cache()
        fast path on reschedule."""
        impl = make_stub_impl()
        impl._req_to_gaps["req-live"] = [(0, 256)]
        impl._request_trackers = {}
        impl._unfinished_requests = {}
        impl._requests_priority = {}
        impl.load_specs = {}
        impl.force_skip_save = False
        impl.kv_role = "kv_consumer"
        impl.config.priority_limit = None

        scheduler_output = MagicMock()
        scheduler_output.finished_req_ids = []
        scheduler_output.scheduled_new_reqs = []
        cached_reqs = MagicMock()
        cached_reqs.req_ids = []
        cached_reqs.new_block_ids = []
        scheduler_output.scheduled_cached_reqs = cached_reqs

        impl.build_connector_meta(scheduler_output)

        assert "req-live" in impl._req_to_gaps


class TestApplyGapMasks:
    """_apply_gap_masks() zeros out gap positions in token_mask and slot_mapping."""

    def test_no_gaps_leaves_masks_unchanged(self):
        token_mask = torch.ones(8, dtype=torch.bool)
        slot_mapping = torch.arange(8, dtype=torch.long)
        original_slot = slot_mapping.clone()

        LMCacheConnectorV1Impl._apply_gap_masks(token_mask, slot_mapping, gaps=[])

        assert token_mask.all()
        assert torch.equal(slot_mapping, original_slot)

    def test_gap_positions_zeroed_in_token_mask(self):
        token_mask = torch.ones(8, dtype=torch.bool)
        slot_mapping = torch.arange(8, dtype=torch.long)

        LMCacheConnectorV1Impl._apply_gap_masks(
            token_mask, slot_mapping, gaps=[(2, 4)]
        )

        expected_mask = torch.tensor(
            [True, True, False, False, True, True, True, True]
        )
        assert torch.equal(token_mask, expected_mask)

    def test_gap_positions_use_last_slot_value_in_slot_mapping(self):
        """Gap positions in slot_mapping are set to slot_mapping[-1] (dummy)."""
        token_mask = torch.ones(8, dtype=torch.bool)
        slot_mapping = torch.arange(8, dtype=torch.long)  # last value = 7

        LMCacheConnectorV1Impl._apply_gap_masks(
            token_mask, slot_mapping, gaps=[(2, 4)]
        )

        assert slot_mapping[2].item() == 7
        assert slot_mapping[3].item() == 7
        assert slot_mapping[0].item() == 0
        assert slot_mapping[4].item() == 4

    def test_multiple_gaps(self):
        token_mask = torch.ones(10, dtype=torch.bool)
        slot_mapping = torch.arange(10, dtype=torch.long)

        LMCacheConnectorV1Impl._apply_gap_masks(
            token_mask, slot_mapping, gaps=[(1, 3), (6, 8)]
        )

        expected_mask = torch.tensor(
            [True, False, False, True, True, True, False, False, True, True]
        )
        assert torch.equal(token_mask, expected_mask)
        assert slot_mapping[1].item() == 9   # last slot value
        assert slot_mapping[2].item() == 9
        assert slot_mapping[6].item() == 9
        assert slot_mapping[7].item() == 9


class TestVirtualRequestFiltering:
    """build_connector_meta() skips virtual request IDs.

    Virtual request IDs are created by the segmented-prefill scheduler
    for gap chunks. They look like '<parent_req_id>.<gap_start_token>'
    (e.g., 'chatcmpl-abc.256'). They must not create RequestTracker entries
    since their KV blocks are shared with the parent request.
    """

    def _make_fake_new_req(self, req_id: str, num_computed_tokens: int = 0):
        req = MagicMock()
        req.req_id = req_id
        req.num_computed_tokens = num_computed_tokens
        req.sampling_params = None
        req.prompt_token_ids = []
        req.block_ids = [[]]
        req.kv_transfer_params = None
        req.mm_features = None  # prevent extract_mm_features from treating MagicMock as truthy
        return req

    def test_real_request_creates_tracker(self):
        """Ordinary request IDs go through the normal path."""
        impl = make_stub_impl()
        impl._request_trackers = {}
        impl._unfinished_requests = {}
        impl._requests_priority = {}
        impl.force_skip_save = False
        impl.kv_role = "kv_consumer"
        impl.config.priority_limit = None
        impl._lmcache_chunk_size = 256
        impl._block_size = 16
        impl._discard_partial_chunks = False
        impl.config.save_decode_cache = False

        scheduler_output = MagicMock()
        scheduler_output.finished_req_ids = []
        scheduler_output.scheduled_new_reqs = [
            self._make_fake_new_req("real-req-abc")
        ]
        scheduler_output.num_scheduled_tokens = {"real-req-abc": 512}
        cached_reqs = MagicMock()
        cached_reqs.req_ids = []
        scheduler_output.scheduled_cached_reqs = cached_reqs

        impl.build_connector_meta(scheduler_output)

        assert "real-req-abc" in impl._request_trackers

    def test_virtual_request_does_not_create_tracker(self):
        """Virtual request IDs (parent.start format) are skipped."""
        impl = make_stub_impl()
        impl._request_trackers = {"real-req-abc": MagicMock()}  # parent exists
        impl._unfinished_requests = {}
        impl._requests_priority = {}
        impl.force_skip_save = False
        impl.kv_role = "kv_consumer"
        impl.config.priority_limit = None

        scheduler_output = MagicMock()
        scheduler_output.finished_req_ids = []
        scheduler_output.scheduled_new_reqs = [
            self._make_fake_new_req("real-req-abc.256")
        ]
        scheduler_output.num_scheduled_tokens = {"real-req-abc.256": 256}
        cached_reqs = MagicMock()
        cached_reqs.req_ids = []
        scheduler_output.scheduled_cached_reqs = cached_reqs

        impl.build_connector_meta(scheduler_output)

        assert "real-req-abc.256" not in impl._request_trackers

    def test_dotted_real_request_not_mistaken_for_virtual(self):
        """A real request ID with a dot but whose base is NOT in
        _request_trackers is treated as a real request, not virtual."""
        impl = make_stub_impl()
        impl._request_trackers = {}  # "chatcmpl" NOT in trackers
        impl._unfinished_requests = {}
        impl._requests_priority = {}
        impl.force_skip_save = False
        impl.kv_role = "kv_consumer"
        impl.config.priority_limit = None
        impl._lmcache_chunk_size = 256
        impl._block_size = 16
        impl._discard_partial_chunks = False
        impl.config.save_decode_cache = False

        scheduler_output = MagicMock()
        scheduler_output.finished_req_ids = []
        scheduler_output.scheduled_new_reqs = [
            self._make_fake_new_req("chatcmpl.abc")
        ]
        scheduler_output.num_scheduled_tokens = {"chatcmpl.abc": 512}
        cached_reqs = MagicMock()
        cached_reqs.req_ids = []
        scheduler_output.scheduled_cached_reqs = cached_reqs

        impl.build_connector_meta(scheduler_output)

        assert "chatcmpl.abc" in impl._request_trackers

    def test_parent_and_virtual_in_same_batch(self):
        """Both parent and virtual appear in the same scheduled_new_reqs list.

        The scheduler always places the parent before its virtual children,
        so the parent tracker is registered before the virtual is processed.
        This test documents that ordering assumption.
        """
        impl = make_stub_impl()
        impl._request_trackers = {}
        impl._unfinished_requests = {}
        impl._requests_priority = {}
        impl.force_skip_save = False
        impl.kv_role = "kv_consumer"
        impl.config.priority_limit = None
        impl._lmcache_chunk_size = 256
        impl._block_size = 16
        impl._discard_partial_chunks = False
        impl.config.save_decode_cache = False

        scheduler_output = MagicMock()
        scheduler_output.finished_req_ids = []
        # Parent first, then virtual — ordering guaranteed by the scheduler
        scheduler_output.scheduled_new_reqs = [
            self._make_fake_new_req("real-req-abc"),
            self._make_fake_new_req("real-req-abc.256"),
        ]
        scheduler_output.num_scheduled_tokens = {
            "real-req-abc": 512,
            "real-req-abc.256": 256,
        }
        cached_reqs = MagicMock()
        cached_reqs.req_ids = []
        scheduler_output.scheduled_cached_reqs = cached_reqs

        impl.build_connector_meta(scheduler_output)

        assert "real-req-abc" in impl._request_trackers
        assert "real-req-abc.256" not in impl._request_trackers
