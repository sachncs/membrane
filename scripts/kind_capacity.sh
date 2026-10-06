#!/usr/bin/env bash
# Cluster read capacity on kind: does adding nodes add capacity?
#
# Every Membrane pod gets the same fixed CPU budget (CPU_LIMIT, requests =
# limits), the way capacity is planned on Kubernetes. Without a budget all
# pods share the host's cores, so a bigger cluster cannot serve more on the
# same machine. The read load drives every pod to its budget at once.
#
#   1. 3 nodes: seed SEED_FRAGMENTS payload-bearing fragments, measure.
#   2. Scale to 5 and wait until rebalancing settles (each new pod holds
#      fragments and the primary counts stop changing), measure again.
#   3. Pass when 5-node reads/s >= MIN_SCALING x 3-node reads/s
#      (5/3 = 1.67 is linear).
#
# Needs about CPU_LIMIT x 5 + 1.5 CPUs for Docker. Usage:
#   IMAGE=membrane:ci scripts/kind_capacity.sh
set -euo pipefail

IMAGE="${IMAGE:-membrane:ci}"
CLUSTER="${CLUSTER:-membrane-capacity}"
CPU_LIMIT="${CPU_LIMIT:-250m}"
SEED_FRAGMENTS="${SEED_FRAGMENTS:-600}"
MEASURE_SECONDS="${MEASURE_SECONDS:-30}"
CONNECTIONS="${CONNECTIONS:-8}"
MIN_SCALING="${MIN_SCALING:-1.5}"
LOG_TAG=kind-capacity
KIND_GUARANTEED=1
# shellcheck source=scripts/kind_lib.sh
source "$(dirname "$0")/kind_lib.sh"

kind_up
deploy_membrane "$CPU_LIMIT" 512Mi

measure() {  # measure PODS NAME : prints the reads/s of a capacity run
    run_pod "$2" capacity --pods "$1" --duration "$MEASURE_SECONDS" --connections "$CONNECTIONS"
    wait_pod "$2" || { log "FAIL: capacity run on $1 pods failed"; exit 1; }
    $K -n ingress logs "$2" | sed -n 's/^CAPACITY //p' | python3 -c 'import json,sys; print(json.load(sys.stdin)["rps"])'
}

primary_counts() {
    for pod in $($K get pods -l app=membrane -o name); do
        pod_get "${pod#pod/}" heartbeat 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin)["primary_count"])' 2>/dev/null || echo 0
    done | tr '\n' ' '
}

log "seeding $SEED_FRAGMENTS fragments"
run_pod seed seed --count "$SEED_FRAGMENTS" --pods 3
wait_pod seed || { log "FAIL: seeding failed"; exit 1; }

log "measuring 3 nodes (cpu $CPU_LIMIT each)"
three="$(measure 3 capacity-3)"
log "3 nodes: $three reads/s"

log "scaling to 5 and waiting for rebalancing to settle"
$K scale statefulset/membrane --replicas=5 >/dev/null
$K rollout status statefulset/membrane --timeout=300s >/dev/null
deadline=$((SECONDS + 400))
previous=""
settled=0
while (( SECONDS < deadline )); do
    counts="$(primary_counts)"
    log "primaries per pod: $counts"
    read -r -a c <<< "$counts"
    if [ "${#c[@]}" -eq 5 ] && [ "${c[3]}" -gt 0 ] && [ "${c[4]}" -gt 0 ] && [ "$counts" == "$previous" ]; then
        settled=1
        break
    fi
    previous="$counts"
    sleep 15
done
(( settled )) || { log "FAIL: primary ownership did not settle within 400s"; exit 1; }

log "measuring 5 nodes"
five="$(measure 5 capacity-5)"
log "5 nodes: $five reads/s"

ratio="$(python3 -c "print(f'{$five / $three:.2f}')")"
log "5 nodes / 3 nodes: ${ratio}x (need >= $MIN_SCALING; linear is 1.67)"
python3 -c "import sys; sys.exit(0 if $five / $three >= $MIN_SCALING else 1)" \
    || { log "FAIL: 5 nodes serve only ${ratio}x the reads of 3"; exit 1; }
log "PASS"
