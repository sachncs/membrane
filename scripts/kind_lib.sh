# shellcheck shell=bash
# Shared setup for the kind scripts (kind_e2e.sh, kind_capacity.sh); source it.
#
# Expects IMAGE and CLUSTER to be set. Provides $K (kubectl for the cluster),
# $WORK (a temporary directory removed on exit), $ROOT, $LOAD_KEY, and:
#
#   kind_up                     create the cluster and load $IMAGE
#   deploy_membrane CPU MEMORY  Redis plus the shipped manifests, each pod
#                               limited to CPU and MEMORY (requests = limits
#                               when KIND_GUARANTEED=1)
#   run_pod NAME ARGS...        run scripts/kind_load.py in a pod
#   wait_pod NAME               wait for it, print its logs, return its status
#   pod_get POD PATH            GET http://127.0.0.1:8080/PATH inside a pod (peer key)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
K="kubectl --context kind-${CLUSTER}"

log() { printf '[%s] %s\n' "${LOG_TAG:-kind}" "$*" >&2; }
cleanup() {
    if [ "${KEEP_CLUSTER:-0}" != 1 ]; then kind delete cluster --name "$CLUSTER" >/dev/null 2>&1 || true; fi
    rm -rf "$WORK"
}
trap cleanup EXIT

kind_up() {
    log "creating cluster $CLUSTER"
    kind create cluster --name "$CLUSTER" --wait 120s >/dev/null
    kind load docker-image "$IMAGE" --name "$CLUSTER" >/dev/null
}

deploy_membrane() {  # deploy_membrane CPU MEMORY
    local cpu="$1" memory="$2"
    log "generating keys"
    gen() { docker run --rm "$IMAGE" membrane keys generate --subject "$1" --scope admin 2>/dev/null; }
    gen membrane-peers > "$WORK/peers.txt"
    gen kind-load > "$WORK/load.txt"
    gen metrics-scraper > "$WORK/metrics.txt"
    for f in peers load metrics; do sed -n 2p "$WORK/$f.txt"; done > "$WORK/api-keys"
    sed -n 1p "$WORK/peers.txt" | tr -d '\n' > "$WORK/peer-api-key"
    sed -n 1p "$WORK/metrics.txt" | tr -d '\n' > "$WORK/metrics-token"
    LOAD_KEY="$(sed -n 1p "$WORK/load.txt")"

    log "deploying Redis and Membrane (cpu $cpu, memory $memory per pod)"
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
    local py
    if command -v uv >/dev/null; then py=(uv run --quiet --no-project --with pyyaml python); else py=(python3); fi
    "${py[@]}" - "$ROOT" "$IMAGE" "$cpu" "$memory" "${KIND_GUARANTEED:-0}" > "$WORK/manifests.json" <<'EOF'
import json, sys, pathlib
root, image, cpu, memory, guaranteed = sys.argv[1:6]
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
            requests = {"cpu": cpu, "memory": memory} if guaranteed == "1" else {"cpu": "100m", "memory": "256Mi"}
            container["resources"] = {"requests": requests, "limits": {"cpu": cpu, "memory": memory}}
            doc["spec"]["volumeClaimTemplates"][0]["spec"]["resources"]["requests"]["storage"] = "1Gi"
        docs.append(doc)
json.dump({"apiVersion": "v1", "kind": "List", "items": docs}, sys.stdout)
EOF
    $K apply -f "$WORK/manifests.json" >/dev/null
    $K rollout status statefulset/membrane --timeout=300s >/dev/null
    log "3 pods ready"
    $K create namespace ingress >/dev/null
    $K -n ingress create configmap kind-load --from-file=kind_load.py="$ROOT/scripts/kind_load.py" >/dev/null
}

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

pod_get() {  # pod_get POD PATH : prints the JSON body of GET /PATH, asked inside POD with the peer key
    $K exec "$1" -- python -c '
import sys, urllib.request
key = open("/run/secrets/membrane/peer-api-key").read().strip()
req = urllib.request.Request("http://127.0.0.1:8080/" + sys.argv[1], headers={"Authorization": "Bearer " + key})
print(urllib.request.urlopen(req).read().decode())' "$2"
}
