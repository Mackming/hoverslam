"""
Lightweight robust development server for Hoverslam web console.
Handles browser connection resets gracefully.
"""

import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WEB_DIR = Path(__file__).resolve().parent

class QuietHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def log_message(self, format, *args):
        # Quiet standard logging to keep stdout clean
        pass

def run(port=3000):
    server = ThreadingHTTPServer(("0.0.0.0", port), QuietHandler)
    print(f"Hoverslam dev server listening on http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    run(port)
