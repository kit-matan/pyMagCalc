import os
import sys
import threading
import uvicorn
import webview
import socket
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Ensure we can import from current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the existing backend app
# This assumes server.py initializes 'app'
from server import app as backend_app

def find_free_port():
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def create_app():
    """
    Create the main FastAPI app that combines the backend and the static frontend.
    """
    # Check if frontend build exists
    gui_dir = os.path.dirname(os.path.abspath(__file__))
    dist_dir = os.path.join(gui_dir, "dist")
    
    if not os.path.exists(dist_dir):
        print("Error: Frontend build directory 'gui/dist' not found.")
        print("Please run 'npm install' and 'npm run build' in the 'gui' directory first.")
        sys.exit(1)

    # Wrapper App
    main_app = FastAPI(lifespan=lifespan)

    # CORS (Optional but good for safety if we change ports)
    main_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount the backend at /api
    # Note: Vite proxy rewrites /api/foo -> /foo, so our backend expects /foo, /run-calculation etc.
    # But here we are mounting it.
    # If we mount backend_app at /api, a request to /api/run-calculation goes to backend_app as /run-calculation.
    # This matches the 'rewrite' rule in vite.config.js perfectly.
    main_app.mount("/api", backend_app)
    
    # We also need to expose the /files mount/route from the backend if it's not relative to the mounting point.
    # backend_app in server.py mounts "/files".
    # When mounted under /api, it becomes /api/files.
    # But the frontend might expect /files directly if the proxy was configured that way.
    # Let's check vite.config.js again.
    # Vite proxy: '/files': { target: 'http://localhost:8000', changeOrigin: true } (NO REWRITE)
    # So frontend requests /files/foo.
    # So we must mount the same static files at /files on the main app too.
    
    project_root = os.path.dirname(gui_dir) # parent of gui
    if os.path.exists(project_root):
        main_app.mount("/files", StaticFiles(directory=project_root), name="files_root")
        
    # Also we need to handle the WebSocket for logs.
    # backend_app has @app.websocket("/ws/logs").
    # When mounted at /api, it becomes ws://host/api/ws/logs.
    # Vite proxy: '/ws': { target: 'ws://localhost:8000', changeOrigin: true, ws: true } (NO REWRITE)
    # This implies frontend connects to /ws/logs (or whatever path).
    # Wait, server.py defines @app.websocket("/ws/logs").
    # If we mount backend_app at /api, it is at /api/ws/logs.
    # But frontend likely tries /ws/logs.
    # We might need to mount the websocket route specifically or mount the backend at root?
    # If we mount backend at root, it conflicts with static files.
    # Solution: Add a specific route in main_app that delegates to backend?
    # Or just mount backend at /api and tell frontend (via window object?) where API is?
    # But we want to use the EXISTING build.
    
    # Let's look at App.jsx websocket connection:
    # const wsUrl = `${protocol}//${window.location.host}/ws/logs`;
    # So it expects /ws/logs at the root.
    
    # We can mount the backend's router's websocket endpoint specifically?
    # Or cleaner: Just mount the same backend_app logic at /ws too? No.
    
    # Workaround: Re-declare the websocket route here that delegates?
    # Or better: Isolate the websocket logic in server.py so it can be attached to this app too?
    # server.py has 'websocket_logs' function. We can import it.
    from server import websocket_logs
    main_app.add_api_websocket_route("/ws/logs", websocket_logs)

    # 3. Static Files (Frontend)
    # Must be last to avoid capturing API routes (if we used a catch-all)
    # But StaticFiles works on directory.
    # We want SPA fallback (index.html for 404s).
    # StaticFiles doesn't support SPA fallback natively easily.
    # Better approach: Mount /assets to dist/assets, then a catch-all route for index.html.
    
    main_app.mount("/assets", StaticFiles(directory=os.path.join(dist_dir, "assets")), name="assets")
    
    # Serve other root files like favicon, etc?
    # For simplicity, we can just serve index.html for everything else.
    
    @main_app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        # Check if file exists in dist (e.g. vite.svg)
        file_path = os.path.join(dist_dir, full_path)
        if os.path.exists(file_path) and os.path.isfile(file_path):
             return FileResponse(file_path)
        
        # Otherwise return index.html
        return FileResponse(os.path.join(dist_dir, "index.html"))

    return main_app
    
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Manually trigger backend startup to initialize logging loop
    # We import inside function to avoid circular imports if any, 
    # though here it's fine.
    from server import startup_event
    await startup_event()
    yield
    # Shutdown logic if needed


def run_native():
    port = find_free_port()
    
    # Create the app
    app = create_app()
    
    # Run Server in Thread
    def start_server():
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
        
    t = threading.Thread(target=start_server, daemon=True)
    t.start()
    
    # Launch Webview
    window_title = "pyMagCalc Studio"
    url = f"http://127.0.0.1:{port}"
    
    webview.create_window(window_title, url, width=1200, height=900, resizable=True)
    webview.start()

if __name__ == "__main__":
    run_native()
