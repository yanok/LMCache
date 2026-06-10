# SPDX-License-Identifier: Apache-2.0
# Third Party
import pytest

# First Party
from lmcache.v1.config import LMCacheEngineConfig


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
