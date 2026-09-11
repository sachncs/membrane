# Consistency levels

Membrane offers three write-consistency levels for
``op_store`` (the canonical ``POST /fragment`` write path):

| Level        | Default? | Acks required                      | Failure mode                                       |
|--------------|----------|------------------------------------|----------------------------------------------------|
| ``strong``   | yes      | all configured replicas            | 503 + ``Retry-After`` on timeout (fail closed)     |
| ``quorum``   | no       | ``quorum_count`` replicas          | 503 + ``Retry-After`` if quorum not met            |
| ``eventual`` | no       | local write, gossip propagates     | silent convergence; no typed failure               |

The ``Fragment.consistency`` field carries the literal level
in the v5 wire envelope; the cluster's
``ClusterConfig.default_consistency`` (default ``"strong"``)
is applied by ``op_store`` when the incoming fragment's
field is missing.

## Defaults and timing

The relationship between the failure-detection timing and
the quorum timeout is enforced by configuration:

| Field                         | Default | Meaning                                                  |
|-------------------------------|---------|----------------------------------------------------------|
| ``heartbeat_interval_sec``    | 2.0     | Seconds between heartbeats.                              |
| ``failure_remove_threshold``  | 4       | Missed heartbeats before a peer is removed.              |
| ``failure_suspect_threshold`` | 2       | Missed heartbeats before a peer is marked suspect.       |
| ``quorum_count``              | 2       | Required replica acks for ``strong`` / ``quorum``.       |
| ``cluster_quorum_timeout_sec``| 5.0     | Wall-clock budget for the quorum fan-out wait.          |

The defaults satisfy ``failure_remove_threshold *
heartbeat_interval_sec`` (= 8 s) **>** ``cluster_quorum_timeout_sec``
(= 5 s); in practice a peer is held around long enough for
the quorum write to either succeed or fail-closed with a
typed error.

## Failure modes

* ``strong`` timeout: the writer returns ``503`` with a
  ``Retry-After`` header; the local write is rolled back so
  gossip does not propagate an un-acked fragment. This is
  the cluster's fail-closed contract.
* ``quorum`` timeout: same behaviour as ``strong`` except
  the number of required acks is ``quorum_count`` instead of
  the full replica set.
* ``eventual``: the local write succeeds immediately. There
  is no typed error; convergence happens on the next gossip
  round (within ``gossip_interval_sec`` seconds).

The cluster state machine consumes these outcomes and
either accepts the fragment, retries on the next write, or
suspects the offending peer (incrementing the
``membrane_gossip_failures_total`` counter).

## Relationship to security surfaces

* ``MTLSConfig.allowed_cns`` controls which peer CNs may
  participate in the quorum fan-out. A peer not in the
  allow-list is rejected at the TLS handshake before the
  consistency timer starts.
* ``TenantAuthorizer`` gates the per-op authorisation on
  every ``op_store`` regardless of the consistency level;
  ``strong`` does not bypass tenant scoping.
* The Prometheus counters
  ``membrane_quorum_failures_total{level="strong|quorum"}``
  and ``membrane_gossip_failures_total{reason="timeout|5xx|4xx"}``
  are the only operational signals; they are not exposed as
  typed errors to callers.

## See also

* ``membrane/network/config.py`` — ``ClusterConfig`` defaults
* ``membrane/transport/ops.py`` — ``op_store`` handler
* ``docs/wire-format.md`` — on-wire envelope
* ``docs/operations/slo.md`` — operational targets
