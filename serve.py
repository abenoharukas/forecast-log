"""Static server that never caches, so a reload always shows the current build.

    python3 serve.py 8731

Plain `python3 -m http.server` lets the browser reuse a cached page, which makes a
changed chart look unchanged.
"""

from __future__ import annotations

import http.server
import sys


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, *args) -> None:
        pass


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8731
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"http://127.0.0.1:{port}/web/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
