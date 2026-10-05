#!/usr/bin/env bash
# End-to-end cluster test on kind: the shipped Kubernetes manifests, a
# 3-node StatefulSet with Redis and the NetworkPolicy, and payload-bearing
# strong writes.
#
#   1. Rolling restart under continuous writes: no write may fail.
#   2. Primary loss: kill a pod; the others must serve the bytes it held.
#   3. Scale 3 -> 4: the new pod must receive primaries (rebalancing).
#
# Usage: IMAGE=membrane:ci scripts/kind_e2e.sh   (builds nothing; load the
# image first or let this script `kind load` it). Set KEEP_CLUSTER=1 to
# keep the cluster for debugging.
set -euo pipefail

IMAGE="${IMAGE:-membrane:ci}"
CLUSTER="${CLUSTER:-membrane-e2e}"
WRITE_SECONDS="${WRITE_SECONDS:-150}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
K="kubectl --context kind-${CLUSTER}"

log() { printf '[kind-e2e] %s\n' "$*" >&2; }
cleanup() {
    if [ "${KEEP_CLUSTER:-0}" != 1 ]; then kind delete cluster --name "$CLUSTER" >/dev/null 2>&1 || true; fi
    rm -rf "$WORK"
}
trap cleanup EXIT

log "creating cluster $CLUSTER"
kind create cluster --name "$CLUSTER" --wait 120s >/dev/null
kind load docker-image "$IMAGE" --name "$CLUSTER" >/dev/null

log "generating keys"
gen() { docker run --rm "$IMAGE" membrane keys generate --subject "$1" --scope admin 2>/dev/null; }
gen membrane-peers > "$WORK/peers.txt"
gen kind-load > "$WORK/load.txt"
gen metrics-scraper > "$WORK/metrics.txt"
for f in peers load metrics; do sed -n 2p "$WORK/$f.txt"; done > "$WORK/api-keys"
sed -n 1p "$WORK/peers.txt" | tr -d '\n' > "$WORK/peer-api-key"
sed -n 1p "$WORK/metrics.txt" | tr -d '\n' > "$WORK/metrics-token"
LOAD_KEY="$(sed -n 1p "$WORK/load.txt")"

log "deploying Redis and Membrane"
$K apply -f - >/dev/null <<'EOF'
apiVersion: apps/v1
kind: Deployment
metadata: {name: membrane-redis}
spec:
  selector: {matchLabels: {app: redis}}
  template:
    metadata: {labels: {app: redis}}
    spec: {containers: [{name: redis, image: "redis:7-alpine", ports: [{containerPort: 6379}]}]}
---
apiVersion: v1
kind: Service
metadata: {name: membrane-redis}
spec: {selector: {app: redis}, ports: [{port: 6379}]}
EOF
$K create secret generic membrane-secrets --from-file=api-keys="$WORK/api-keys" \
    --from-file=peer-api-key="$WORK/peer-api-key" --from-file=metrics-token="$WORK/metrics-token" >/dev/null
if command -v uv >/dev/null; then PY=(uv run --quiet --no-project --with pyyaml python); else PY=(python3); fi
"${PY[@]}" - "$ROOT" "$IMAGE" > "$WORK/manifests.json" <<'EOF'
import json, sys, pathlib
root, image = sys.argv[1], sys.argv[2]
try:
    import yaml
except ImportError:
    sys.exit("PyYAML is required (pip install pyyaml)")
docs = []
for name in ("configmap.yaml", "service.yaml", "pdb.yaml", "statefulset.yaml", "networkpolicy.yaml"):
    for doc in yaml.safe_load_all(pathlib.Path(root, "deployment/k8s", name).read_text()):
        if not doc or doc["kind"] == "Secret":
            continue
        if doc["kind"] == "StatefulSet":
            container = doc["spec"]["template"]["spec"]["containers"][0]
            container["image"] = image
            container["imagePullPolicy"] = "Never"
            container["resources"] = {"requests": {"cpu": "100m", "memory": "256Mi"}, "limits": {"cpu": "1", "memory": "1Gi"}}
            doc["spec"]["volumeClaimTemplates"][0]["spec"]["resources"]["requests"]["storage"] = "1Gi"
        docs.append(doc)
json.dump({"apiVersion": "v1", "kind": "List", "items": docs}, sys.stdout)
EOF
$K apply -f "$WORK/manifests.json" >/dev/null
$K rollout status statefulset/membrane --timeout=300s >/dev/null
log "3 pods ready"

$K create namespace ingress >/dev/null
$K -n ingress create configmap kind-load --from-file=kind_load.py="$ROOT/scripts/kind_load.py" >/dev/null
run_pod() {  # run_pod NAME ARGS... : runs kind_load.py in the ingress namespace
    local name="$1"; shift
    local args; args="$(printf '"%s",' "$@")"; args="[${args%,}]"
    $K -n ingress run "$name" --image="$IMAGE" --image-pull-policy=Never --restart=Never --overrides="$(cat <<JSON
{"spec": {"volumes": [{"name": "s", "configMap": {"name": "kind-load"}}],
  "containers": [{"name": "$name", "image": "$IMAGE", "imagePullPolicy": "Never",
    "command": ["python", "/s/kind_load.py"], "args": $args,
    "env": [{"name": "KEY", "value": "$LOAD_KEY"}, {"name": "POD_NAMESPACE", "value": "default"}],
    "volumeMounts": [{"name": "s", "mountPath": "/s"}]}]}}
JSON
)" >/dev/null
}
wait_pod() {  # wait_pod NAME : waits for completion, prints logs, returns the pod's exit status
    local name="$1" phase
    until phase="$($K -n ingress get pod "$name" -o jsonpath='{.status.phase}')" && [[ "$phase" == Succeeded || "$phase" == Failed ]]; do sleep 3; done
    $K -n ingress logs "$name" >&2
    [ "$phase" == Succeeded ]
}

log "1/3 rolling restart under ${WRITE_SECONDS}s of strong writes"
run_pod writer write --duration "$WRITE_SECONDS"
sleep 20
$K rollout restart statefulset/membrane >/dev/null
$K rollout status statefulset/membrane --timeout=400s >/dev/null
wait_pod writer || { log "FAIL: writes failed during the rolling restart"; exit 1; }
SAMPLE="$($K -n ingress logs writer | sed -n 's/^SUMMARY //p' | python3 -c 'import json,sys; print(",".join(json.load(sys.stdin)["sample"]))')"

log "2/3 primary loss: force-deleting membrane-0"
$K delete pod membrane-0 --grace-period=0 --force >/dev/null 2>&1
run_pod verifier verify --hashes "$SAMPLE" --pods 3 --skip 0
wait_pod verifier || { log "FAIL: replicas did not serve the bytes"; exit 1; }
$K rollout status statefulset/membrane --timeout=300s >/dev/null

log "3/3 scale 3 -> 4: waiting for membrane-3 to receive primaries"
$K scale statefulset/membrane --replicas=4 >/dev/null
$K rollout status statefulset/membrane --timeout=300s >/dev/null
deadline=$((SECONDS + 240))
count=0
while (( SECONDS < deadline )); do
    count="$($K exec membrane-3 -- python -c '
import json, urllib.request
key = open("/run/secrets/membrane/peer-api-key").read().strip()
req = urllib.request.Request("http://127.0.0.1:8080/inventory", headers={"Authorization": "Bearer " + key})
print(len(json.loads(urllib.request.urlopen(req).read())["digest"]))' 2>/dev/null || echo 0)"
    [ "$count" -gt 0 ] && break
    sleep 5
done
[ "$count" -gt 0 ] || { log "FAIL: membrane-3 received no fragments after scaling"; exit 1; }
log "membrane-3 holds $count fragments"

for pod in $($K get pods -l app=membrane -o name); do
    if $K logs "$pod" | grep -q 'unhandled exception\|loop crashed'; then
        log "FAIL: $pod logged a crashed background thread"; exit 1
    fi
done
log "PASS"
