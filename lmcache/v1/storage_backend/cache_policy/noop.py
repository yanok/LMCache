# SPDX-License-Identifier: Apache-2.0
# First Party
from lmcache.v1.storage_backend.cache_policy.base_policy import BaseCachePolicy, KeyType


class NOOPCachePolicy(BaseCachePolicy[KeyType, dict]):
    """
    NOOP cache policy — never evicts anything.

    Intended for "fill once, keep forever" scenarios where entries should
    never be displaced once written.  When the cache is full, callers that
    request eviction candidates receive an empty list and are responsible for
    dropping the incoming write rather than freeing existing entries.
    """

    def init_mutable_mapping(self) -> dict:
        """
        Initialize the backing store for the cache.

        Returns:
            An empty plain dict.
        """
        return {}

    def update_on_hit(
        self,
        key: KeyType,
        cache_dict: dict,
    ) -> None:
        """
        Called when a cached entry is accessed.

        Args:
            key: The key that was accessed.
            cache_dict: The current cache mapping.
        """

    def update_on_put(
        self,
        key: KeyType,
    ) -> None:
        """
        Called when a new entry is stored in the cache.

        Args:
            key: The key that was stored.
        """

    def update_on_force_evict(
        self,
        key: KeyType,
    ) -> None:
        """
        Called when an entry is forcibly removed from the cache.

        Args:
            key: The key that was force-evicted.
        """

    def get_evict_candidates(
        self,
        cache_dict: dict,
        num_candidates: int = 1,
    ) -> list[KeyType]:
        """
        Return entries to evict when the cache is full.

        This policy never evicts, so the returned list is always empty.
        Callers must drop incoming writes when they receive no candidates.

        Args:
            cache_dict: The current cache mapping.
            num_candidates: Maximum number of candidates requested.

        Returns:
            An empty list.
        """
        return []
