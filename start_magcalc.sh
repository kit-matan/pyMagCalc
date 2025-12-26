#!/bin/bash

# Configuration
SERVER_PORT=8000
CLIENT_PORT=5173
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🚀 Starting MagCalc Designer..."

# 1. Cleanup existing ports
echo "🧹 Cleaning up existing processes..."
lsof -ti:$SERVER_PORT | xargs kill -9 2>/dev/null
lsof -ti:$CLIENT_PORT | xargs kill -9 2>/dev/null

# 2. Start Backend
echo "🔥 Starting Backend Server (Port $SERVER_PORT)..."
cd "$PROJECT_DIR"
python gui/server.py > /dev/null 2>&1 &
BACKEND_PID=$!

# 3. Start Frontend
echo "✨ Starting Frontend (Port $CLIENT_PORT)..."
npm --prefix gui run dev > /dev/null 2>&1 &
FRONTEND_PID=$!

# Trap Ctrl+C to kill both processes
trap "echo '🛑 Stopping MagCalc...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" SIGINT SIGTERM

# 4. Wait for startup and open browser
echo "⏳ Waiting for services to initialize..."
sleep 3
echo "🌐 Opening Browser..."
open "http://localhost:$CLIENT_PORT"

echo "✅ MagCalc is running! Press Ctrl+C to stop."
wait
