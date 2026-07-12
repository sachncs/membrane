"""CanonicalStore: deduplicated shared KV pool across tenants.

This module defines :class:`CanonicalStore` and its supporting
:class:`CanonicalRef` dataclass. The canonical store is the
*cross-tenant* layer of the deduplication fabric: a single
physical fragment is shared by every tenant that has produced or
consumed it, regardless of which physical node stores it.

The store maintains three pieces of state:

* ``canonical_fragments`` — the actual deduplicated fragment
  payloads, keyed by ``content_hash``.
* ``tenant_refs`` — the set of tenants that currently reference
  each canonical hash.
* An :class:`LRUCache` used to bound the store when
  ``max_entries`` is configured.

References returned by :meth:`store_canonical` carry the
``canonical_hash`` and the snapshot of tenants at insert time, so
callers can cheaply reason about sharing even after further
insertions evict unrelated fragments.

Thread safety:
    The class is **not thread-safe**. Provide external locking
    when sharing across threads.
"""

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
        tenant_ids: Snapshot of tenants sharing this fragment at
            the time the reference was issued. Stored as a
            ``frozenset`` so the reference is hashable.
    """

    canonical_hash: str
    tenant_ids: frozenset[str]


class CanonicalStore:
    """Shared deduplicated pool of fragments across multiple tenants.

    Stores one physical copy per unique content hash and tracks
    which tenants reference it. When ``max_entries`` is set, LRU
    eviction removes the least-recently-accessed fragment on
    insert overflow.

    Attributes:
        canonical_fragments: Mapping from ``content_hash`` to the
            canonical :class:`Fragment`.
        tenant_refs: Mapping from ``content_hash`` to the set of
            tenant IDs that reference it.
        lru: :class:`LRUCache` used to bound the store when
            ``max_entries`` is set.
        max_entries: Configured upper bound on canonical entries
            (or ``None`` for unbounded).
    """

    def __init__(self, max_entries: int | None = None) -> None:
        """Initialize an empty canonical store.

        Args:
            max_entries: Maximum number of unique fragments to
                retain. ``None`` means unbounded (LRU tracking
                remains active but no eviction is triggered).
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
        """Store (or update) a fragment in the canonical pool.

        If the fragment's ``content_hash`` is new, it is added to
        ``canonical_fragments`` and a fresh empty tenant-ref set
        is created. The calling tenant is then added to that set
        unconditionally, and the LRU is touched. When the LRU
        exceeds ``max_entries``, the oldest entries are evicted
        (along with their tenant-ref sets).

        Args:
            fragment: Fragment to deduplicate.
            tenant_id: Tenant contributing the fragment.

        Returns:
            CanonicalRef: Reference to the canonical (possibly
            pre-existing) entry.
        """
        h = fragment.content_hash
        if h not in self.canonical_fragments:
            # First time we've seen this hash — install it as
            # the canonical copy.
            self.canonical_fragments[h] = fragment
            self.tenant_refs[h] = set()
        # Add the tenant regardless of whether the entry was new
        # or existing; deduplication is per-hash, not per-tenant.
        self.tenant_refs[h].add(tenant_id)
        self.lru.touch(h)
        # Evict any overflow entries *after* registering the
        # current touch so the just-inserted fragment cannot be
        # immediately evicted.
        for evicted in self.lru.evict_if_over():
            self.canonical_fragments.pop(evicted, None)
            self.tenant_refs.pop(evicted, None)

        return CanonicalRef(
            canonical_hash=h,
            # Snapshot of the post-insert tenant set.
            tenant_ids=frozenset(self.tenant_refs[h]),
        )

    def retrieve_canonical(self, ref: CanonicalRef) -> Fragment | None:
        """Retrieve a fragment from the canonical pool.

        Touches the LRU on a successful hit so the entry stays
        warm against future eviction.

        Args:
            ref: Canonical reference.

        Returns:
            Fragment | None: The fragment, or ``None`` if the
            referenced hash has been evicted.
        """
        frag = self.canonical_fragments.get(ref.canonical_hash)
        if frag is not None:
            # Refresh LRU only on hit so misses don't pollute the
            # tracker.
            self.lru.touch(ref.canonical_hash)
        return frag

    def get_shared_fragments(self, tenant_id: str) -> list[Fragment]:
        """Return all fragments shared with ``tenant_id``.

        Iterates ``tenant_refs`` (rather than ``canonical_fragments``)
        so that the iteration cost is proportional to the number of
        references, not the number of distinct hashes.

        Args:
            tenant_id: Tenant to query.

        Returns:
            list[Fragment]: Fragments accessible to the tenant,
            in iteration order. Each successful lookup touches
            the LRU so frequently-shared fragments are protected
            from eviction.
        """
        result: list[Fragment] = []
        for h, tenants in self.tenant_refs.items():
            if tenant_id in tenants:
                frag = self.canonical_fragments.get(h)
                if frag is not None:
                    self.lru.touch(h)
                    result.append(frag)
        return result
