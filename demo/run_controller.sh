set -euo pipefail

NUM_NODES="${1:-2}"
NUM_WORKERS_PER_NODE="${2:-1}"
NUM_GROUPS="${3:-1}"
HOST="${4:-127.0.0.1}"
PORT="${5:-18000}"
BUILD_RING="${6:-true}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="$REPO_ROOT/python:${PYTHONPATH:-}"

echo "[run_controller] nodes=$NUM_NODES workers/node=$NUM_WORKERS_PER_NODE groups=$NUM_GROUPS" \
     "host=$HOST port=$PORT build_ring=$BUILD_RING"
exec python "$SCRIPT_DIR/TestSocketSessionTorchMLP.py" \
  --num-nodes "$NUM_NODES" \
  --num-workers-per-node "$NUM_WORKERS_PER_NODE" \
  --num-groups "$NUM_GROUPS" \
  --host "$HOST" \
  --port "$PORT" \
  --build-ring "$BUILD_RING"
