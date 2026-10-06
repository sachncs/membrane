#!/usr/bin/env bash
# End-to-end cluster test on kind: the shipped Kubernetes manifests, a
# 3-node StatefulSet with Redis and the NetworkPolicy, and payload-bearing
# strong writes.
#
#   1. Rolling restart under continuous writes: no write may fail.
#   2. Primary loss: kill a pod; the others must serve the bytes it held.
#   3. Scale 3 -> 4: the new pod must receive primaries (rebalancing).
#   4. Scale 4 -> 5: ownership spreads out; no node may own more than
#      MAX_PRIMARY_SHARE of the primaries (a third each with 3 nodes).
#
# Usage: IMAGE=membrane:ci scripts/kind_e2e.sh   (builds nothing; load the
# image first or let this script `kind load` it). Set KEEP_CLUSTER=1 to
# keep the cluster for debugging.
set -euo pipefail

IMAGE="${IMAGE:-membrane:ci}"
CLUSTER="${CLUSTER:-membrane-e2e}"
WRITE_SECONDS="${WRITE_SECONDS:-150}"
MAX_PRIMARY_SHARE="${MAX_PRIMARY_SHARE:-0.30}"
LOG_TAG=kind-e2e
# shellcheck source=scripts/kind_lib.sh
source "$(dirname "$0")/kind_lib.sh"

kind_up
deploy_membrane 1 1Gi

log "1/4 rolling restart under ${WRITE_SECONDS}s of strong writes"
run_pod writer write --duration "$WRITE_SECONDS"
sleep 20
$K rollout restart statefulset/membrane >/dev/null
$K rollout status statefulset/membrane --timeout=400s >/dev/null
wait_pod writer || { log "FAIL: writes failed during the rolling restart"; exit 1; }
SAMPLE="$($K -n ingress logs writer | sed -n 's/^SUMMARY //p' | python3 -c 'import json,sys; print(",".join(json.load(sys.stdin)["sample"]))')"

log "2/4 primary loss: force-deleting membrane-0"
$K delete pod membrane-0 --grace-period=0 --force >/dev/null 2>&1
run_pod verifier verify --hashes "$SAMPLE" --pods 3 --skip 0
wait_pod verifier || { log "FAIL: replicas did not serve the bytes"; exit 1; }
$K rollout status statefulset/membrane --timeout=300s >/dev/null

log "3/4 scale 3 -> 4: waiting for membrane-3 to receive primaries"
$K scale statefulset/membrane --replicas=4 >/dev/null
$K rollout status statefulset/membrane --timeout=300s >/dev/null
deadline=$((SECONDS + 240))
count=0
while (( SECONDS < deadline )); do
    count="$(pod_get membrane-3 inventory 2>/dev/null | python3 -c 'import json,sys; print(len(json.load(sys.stdin)["digest"]))' 2>/dev/null || echo 0)"
    [ "$count" -gt 0 ] && break
    sleep 5
done
[ "$count" -gt 0 ] || { log "FAIL: membrane-3 received no fragments after scaling"; exit 1; }
log "membrane-3 holds $count fragments"

primary_counts() {  # prints one primary_count per pod
    for pod in $($K get pods -l app=membrane -o name); do
        pod_get "${pod#pod/}" heartbeat 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin)["primary_count"])' 2>/dev/null || echo 0
    done
}
max_share() { python3 -c 'import sys; c=[int(x) for x in sys.stdin.read().split()]; print(f"{max(c)/max(1,sum(c)):.3f} {sum(c)} {c}")'; }

log "4/4 scale 4 -> 5: primaries must spread (max share <= $MAX_PRIMARY_SHARE)"
log "primary share before: $(primary_counts | max_share)"
$K scale statefulset/membrane --replicas=5 >/dev/null
$K rollout status statefulset/membrane --timeout=300s >/dev/null
deadline=$((SECONDS + 300))
share=1
while (( SECONDS < deadline )); do
    read -r share total counts < <(primary_counts | max_share) || true
    log "primary share: $share of $total ($counts)"
    python3 -c "import sys; sys.exit(0 if float('$share') <= float('$MAX_PRIMARY_SHARE') else 1)" && break
    sleep 10
done
python3 -c "import sys; sys.exit(0 if float('$share') <= float('$MAX_PRIMARY_SHARE') else 1)" \
    || { log "FAIL: the busiest of 5 nodes still owns $share of the primaries"; exit 1; }
# Counts sampled pod by pod during hand-offs can miss a fragment in flight;
# the bytes themselves must all still be served.
run_pod verifier-final verify --hashes "$SAMPLE" --pods 5
wait_pod verifier-final || { log "FAIL: fragments became unavailable while rebalancing to 5 nodes"; exit 1; }

for pod in $($K get pods -l app=membrane -o name); do
    if $K logs "$pod" | grep -q 'unhandled exception\|loop crashed'; then
        log "FAIL: $pod logged a crashed background thread"; exit 1
    fi
done
log "PASS"
