#!/usr/bin/env bash
# Run the Redis backend tests against Sentinel and a 3-node Redis Cluster.
#
# Everything runs in Docker on a private network (Redis Cluster nodes must
# reach each other and the client by the same addresses), so this works on
# Linux and Docker Desktop alike:
#
#   scripts/redis_topologies.sh
set -euo pipefail

REDIS_IMAGE=${REDIS_IMAGE:-redis:8-alpine}
PYTHON_IMAGE=${PYTHON_IMAGE:-python:3.14-slim}
NET=membrane-redis-topologies
ROOT=$(cd "$(dirname "$0")/.." && pwd)

cleanup() {
  docker rm -f rt-master rt-sentinel rt-c1 rt-c2 rt-c3 >/dev/null 2>&1 || true
  docker network rm "$NET" >/dev/null 2>&1 || true
}
trap cleanup EXIT
cleanup
docker network create "$NET" >/dev/null

docker run -d --name rt-master --network "$NET" "$REDIS_IMAGE" >/dev/null
docker run -d --name rt-sentinel --network "$NET" --entrypoint sh "$REDIS_IMAGE" -c '
  printf "port 26379\nsentinel resolve-hostnames yes\nsentinel monitor mymaster rt-master 6379 1\n" > /tmp/sentinel.conf
  exec redis-sentinel /tmp/sentinel.conf' >/dev/null

for n in 1 2 3; do
  docker run -d --name "rt-c$n" --network "$NET" "$REDIS_IMAGE" \
    redis-server --cluster-enabled yes --cluster-announce-hostname "rt-c$n" \
      --cluster-preferred-endpoint-type hostname --appendonly no >/dev/null
done
sleep 2
docker exec rt-c1 sh -c 'yes yes | redis-cli --cluster create rt-c1:6379 rt-c2:6379 rt-c3:6379 --cluster-replicas 0' >/dev/null
for _ in $(seq 1 30); do
  docker exec rt-c1 redis-cli cluster info | grep -q 'cluster_state:ok' && break
  sleep 1
done
docker exec rt-c1 redis-cli cluster info | grep -q 'cluster_state:ok'

docker run --rm --network "$NET" -v "$ROOT:/src:ro" -w /src \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e MEMBRANE_TEST_SENTINEL_URL="redis+sentinel://rt-sentinel:26379/mymaster" \
  -e MEMBRANE_TEST_CLUSTER_URL="redis+cluster://rt-c1:6379,rt-c2:6379,rt-c3:6379" \
  "$PYTHON_IMAGE" sh -c '
    pip install -q --root-user-action=ignore redis pytest >/dev/null &&
    python -m pytest -q -p no:cacheprovider tests/membrane/persistence/test_redis_topologies.py'
