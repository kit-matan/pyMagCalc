#!/bin/bash

# Configuration
SERVER_PORT=8000
CLIENT_PORT=5173
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$PROJECT_DIR/.magcalc_logs"
BACKEND_LOG="$LOG_DIR/backend.log"
FRONTEND_LOG="$LOG_DIR/frontend.log"

mkdir -p "$LOG_DIR"

echo "🚀 Starting MagCalc Designer..."
echo "   Logs: $LOG_DIR"

# 1. Stop ALL running MagCalc processes before starting new ones.
#    Killing by port alone is not enough: stale backends that failed to bind the
#    port (e.g. a second `python gui/server.py` launched while one was already
#    running) keep lingering and can serve old code. We match by command, send
#    SIGTERM, then SIGKILL any survivor (uvicorn ignores SIGTERM mid-shutdown).
echo "🧹 Stopping any running MagCalc processes..."

stop_matching() {
    local pattern="$1" label="$2" pids
    pids=$(pgrep -f "$pattern" 2>/dev/null)
    [ -z "$pids" ] && return 0
    echo "   • $label (PIDs: $(echo $pids | tr '\n' ' '))"
    kill $pids 2>/dev/null
    sleep 1
    pids=$(pgrep -f "$pattern" 2>/dev/null)   # force any survivor
    [ -n "$pids" ] && kill -9 $pids 2>/dev/null
}

# Backend: every `python .../gui/server.py`
stop_matching "gui/server.py" "backend server(s)"
# Frontend: the Vite dev server for THIS project (scoped by project path so we
# don't touch unrelated Vite instances)
stop_matching "$PROJECT_DIR/gui/node_modules.*vite" "frontend dev server(s)"

# Final catch-all: anything still holding the ports
lsof -ti:$SERVER_PORT | xargs kill -9 2>/dev/null
lsof -ti:$CLIENT_PORT | xargs kill -9 2>/dev/null
sleep 1

# 2. Start Backend
echo "🔥 Starting Backend Server (Port $SERVER_PORT)..."
cd "$PROJECT_DIR"
python gui/server.py > "$BACKEND_LOG" 2>&1 &
BACKEND_PID=$!

# 3. Start Frontend
if [ ! -d "$PROJECT_DIR/gui/node_modules" ]; then
    echo "⚠️  gui/node_modules not found — running 'npm install' in gui/..."
    (cd "$PROJECT_DIR/gui" && npm install) || {
        echo "❌ npm install failed. Aborting."
        kill $BACKEND_PID 2>/dev/null
        exit 1
    }
fi
echo "✨ Starting Frontend (Port $CLIENT_PORT)..."
npm --prefix gui run dev > "$FRONTEND_LOG" 2>&1 &
FRONTEND_PID=$!

# Trap Ctrl+C to kill both processes
trap "echo '🛑 Stopping MagCalc...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" SIGINT SIGTERM

# 4. Wait for services to be actually listening (up to ~20 s) before opening the browser
# $4 = timeout in seconds (default 20). The backend needs a generous budget:
# it imports pymatgen/spglib/ase/matplotlib/sympy and the magcalc package
# (~13 s warm, and much longer cold when files are pulled from cloud storage
# like OneDrive/Google Drive). The server had actually bound the port before
# the old 20 s limit fired, but the cleanup path below then killed it.
wait_for_port() {
    local port=$1
    local name=$2
    local pid=$3
    local timeout=${4:-20}
    local iterations=$(( timeout * 2 ))   # loop sleeps 0.5 s per iteration
    local i
    for i in $(seq 1 "$iterations"); do
        if lsof -iTCP:$port -sTCP:LISTEN -P -n >/dev/null 2>&1; then
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "❌ $name process exited before binding port $port. Tail of log:"
            local log
            if [ "$name" = "Backend" ]; then log="$BACKEND_LOG"; else log="$FRONTEND_LOG"; fi
            tail -20 "$log"
            return 1
        fi
        sleep 0.5
    done
    echo "⏰ $name did not bind port $port within ${timeout} s. Tail of log:"
    if [ "$name" = "Backend" ]; then tail -20 "$BACKEND_LOG"; else tail -20 "$FRONTEND_LOG"; fi
    return 1
}

echo "⏳ Waiting for backend on $SERVER_PORT..."
wait_for_port $SERVER_PORT "Backend" $BACKEND_PID 60 || { kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 1; }
echo "⏳ Waiting for frontend on $CLIENT_PORT..."
wait_for_port $CLIENT_PORT "Frontend" $FRONTEND_PID 20 || { kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 1; }

URL="http://localhost:$CLIENT_PORT"
echo "🌐 Opening $URL ..."
if command -v open >/dev/null 2>&1; then
    open "$URL" || echo "   (browser launch failed — open the URL manually)"
elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$URL" || echo "   (browser launch failed — open the URL manually)"
else
    echo "   (no browser launcher detected — open $URL manually)"
fi

echo "✅ MagCalc is running at $URL. Press Ctrl+C to stop."
echo "   Backend log: $BACKEND_LOG"
echo "   Frontend log: $FRONTEND_LOG"

# Wait for the backend process. If it exits (e.g., via /shutdown), the script will continue.
wait $BACKEND_PID
# Once backend is gone, kill frontend and exit
kill $FRONTEND_PID 2>/dev/null
