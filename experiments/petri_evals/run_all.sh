#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Activate venv and load environment
source "$PROJECT_ROOT/.venv/bin/activate"
source "$PROJECT_ROOT/.env"

# --- Start the model organism server in the background ---
echo "Starting model organism server on port 8192..."
python "$SCRIPT_DIR/start_server.py" &
SERVER_PID=$!

cleanup() {
    echo "Shutting down server (PID $SERVER_PID)..."
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

# Wait for server to be ready
echo "Waiting for server to be ready..."
for i in $(seq 1 30); do
    if curl -s http://localhost:8192/v1/models > /dev/null 2>&1; then
        echo "Server is ready."
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Server process died. Check logs above."
        exit 1
    fi
    sleep 1
done

if ! curl -s http://localhost:8192/v1/models > /dev/null 2>&1; then
    echo "Server failed to start after 30 seconds."
    exit 1
fi

# --- Run the evals ---
bash "$SCRIPT_DIR/run_petri.sh" "$@"
