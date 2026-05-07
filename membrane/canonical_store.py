"""CanonicalStore: deduplicated shared KV pool across tenants."""

import logging

logger = logging.getLogger(__name__)

from dataclasses import dataclass

from membrane.fragment import Fragment
from membrane.lru_cache import LRUCache


@dataclass(frozen=True)
class CanonicalRef:
    """Reference to a canonical deduplicated fragment.

    Attributes:
        canonical_hash: Hash of the canonical stored fragment.
        tenant_ids: Set of tenants sharing this fragment.
    """

    canonical_hash: str
    tenant_ids: frozenset[str]


class CanonicalStore:
    """Shared deduplicated pool of fragments across multiple tenants.

    Stores one physical copy per unique content hash and tracks which
    tenants reference it.  When *max_entries* is set, LRU eviction
    removes the least-recently-accessed fragment on insert overflow.
    """

    def __init__(self, max_entries: int | None = None) -> None:
        """Initialize an empty canonical store.

        Args:
            max_entries: Maximum number of unique fragments to retain.
        """
        self.canonical_fragments: dict[str, Fragment] = {}
        self.tenant_refs: dict[str, set[str]] = {}
        self.lru = LRUCache(capacity=max_entries)
        self.max_entries = max_entries

    def store_canonical(
        self,
        fragment: Fragment,
        tenant_id: str,
    ) -> CanonicalRef:
        """Store a fragment in the canonical pool.

        Args:
            fragment: Fragment to deduplicate.
            tenant_id: Tenant contributing the fragment.

        Returns:
            CanonicalRef pointing to the shared copy.
        """
        h = fragment.content_hash
        if h not in self.canonical_fragments:
            self.canonical_fragments[h] = fragment
            self.tenant_refs[h] = set()
        self.tenant_refs[h].add(tenant_id)
        self.lru.touch(h)
        for evicted in self.lru.evict_if_over():
            self.canonical_fragments.pop(evicted, None)
            self.tenant_refs.pop(evicted, None)

        return CanonicalRef(
            canonical_hash=h,
            tenant_ids=frozenset(self.tenant_refs[h]),
        )

    def retrieve_canonical(self, ref: CanonicalRef) -> Fragment | None:
        """Retrieve a fragment from the canonical pool.

        Args:
            ref: Canonical reference.

        Returns:
            Fragment if found, else None.
        """
        frag = self.canonical_fragments.get(ref.canonical_hash)
        if frag is not None:
            self.lru.touch(ref.canonical_hash)
        return frag

    def get_shared_fragments(self, tenant_id: str) -> list[Fragment]:
        """Return all fragments shared with a given tenant.

        Args:
            tenant_id: Tenant to query.

        Returns:
            List of fragments accessible to the tenant.
        """
        result: list[Fragment] = []
        for h, tenants in self.tenant_refs.items():
            if tenant_id in tenants:
                frag = self.canonical_fragments.get(h)
                if frag is not None:
                    self.lru.touch(h)
                    result.append(frag)
        return result
