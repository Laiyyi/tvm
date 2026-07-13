set -euo pipefail

HOST="${1:-localhost}"
PORT="${2:-18000}"
NUM_WORKERS="${3:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"


echo "[run_worker] -> $HOST:$PORT  num_workers=$NUM_WORKERS"
exec python -m tvm.exec.disco_remote_socket_session "$HOST" "$PORT" "$NUM_WORKERS"
