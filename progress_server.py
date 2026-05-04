from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


class TrainingStatus:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.data: dict[str, Any] = {
            "version": "unknown",
            "status": "starting",
            "episode": 0,
            "train_episodes": None,
            "last_length": None,
            "avg100": None,
            "last_loss": None,
            "last_eval": None,
            "model_path": "models/policy_model.pt",
            "model_exists": False,
            "done": False,
        }

    def update(self, **kwargs: Any) -> None:
        with self._lock:
            self.data.update(kwargs)

            model_path = self.data.get("model_path")
            if model_path:
                self.data["model_exists"] = os.path.exists(str(model_path))

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            data = dict(self.data)

        model_path = data.get("model_path")
        if model_path:
            path = Path(str(model_path))
            data["model_exists"] = path.exists()
            data["model_size_bytes"] = path.stat().st_size if path.exists() else 0

        return data


STATUS = TrainingStatus()


def _html_page(data: dict[str, Any]) -> str:
    last_eval = data.get("last_eval") or {}

    model_exists = data.get("model_exists")
    download_link = (
        '<p><a href="/download">Download model</a></p>'
        if model_exists
        else "<p>Model checkpoint not available yet.</p>"
    )

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>TREE3 Training Progress</title>
  <meta http-equiv="refresh" content="5">
  <style>
    body {{
      font-family: system-ui, Segoe UI, Arial, sans-serif;
      margin: 2rem;
      max-width: 900px;
      line-height: 1.45;
    }}
    .card {{
      border: 1px solid #ccc;
      border-radius: 12px;
      padding: 1rem 1.25rem;
      margin-bottom: 1rem;
    }}
    code {{
      background: #eee;
      padding: 0.1rem 0.3rem;
      border-radius: 4px;
    }}
    table {{
      border-collapse: collapse;
    }}
    td {{
      padding: 0.25rem 0.75rem 0.25rem 0;
    }}
  </style>
</head>
<body>
  <h1>TREE3 Training Progress</h1>

  <div class="card">
    <table>
      <tr><td>Version</td><td><code>{data.get("version")}</code></td></tr>
      <tr><td>Status</td><td><b>{data.get("status")}</b></td></tr>
      <tr><td>Episode</td><td>{data.get("episode")} / {data.get("train_episodes")}</td></tr>
      <tr><td>Last length</td><td>{data.get("last_length")}</td></tr>
      <tr><td>Avg100</td><td>{data.get("avg100")}</td></tr>
      <tr><td>Last loss</td><td>{data.get("last_loss")}</td></tr>
      <tr><td>Done</td><td>{data.get("done")}</td></tr>
    </table>
  </div>

  <div class="card">
    <h2>Last evaluation</h2>
    <table>
      <tr><td>Episodes</td><td>{last_eval.get("episodes")}</td></tr>
      <tr><td>Average length</td><td>{last_eval.get("avg")}</td></tr>
      <tr><td>Best</td><td>{last_eval.get("best")}</td></tr>
      <tr><td>Worst</td><td>{last_eval.get("worst")}</td></tr>
    </table>
  </div>

  <div class="card">
    <h2>Model</h2>
    <p>Path: <code>{data.get("model_path")}</code></p>
    <p>Exists: {data.get("model_exists")}</p>
    <p>Size: {data.get("model_size_bytes", 0)} bytes</p>
    {download_link}
  </div>

  <p>JSON status: <a href="/status">/status</a></p>
</body>
</html>
"""


class ProgressHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        # Silence default request logging.
        return

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            data = STATUS.snapshot()
            html = _html_page(data).encode("utf-8")

            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
            return

        if self.path == "/status":
            data = json.dumps(STATUS.snapshot(), indent=2).encode("utf-8")

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if self.path == "/download":
            data = STATUS.snapshot()
            model_path = Path(str(data.get("model_path", "")))

            if not model_path.exists():
                body = b"Model checkpoint not available yet.\n"
                self.send_response(404)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            content = model_path.read_bytes()

            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header(
                "Content-Disposition",
                f'attachment; filename="{model_path.name}"',
            )
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return

        body = b"Not found\n"
        self.send_response(404)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def start_progress_server(
    host: str = "0.0.0.0",
    port: int = 80,
    model_path: str = "models/policy_model.pt",
) -> ThreadingHTTPServer:
    STATUS.update(
        status="server_running",
        model_path=model_path,
    )

    server = ThreadingHTTPServer((host, port), ProgressHandler)

    thread = threading.Thread(
        target=server.serve_forever,
        daemon=True,
    )
    thread.start()

    print(f"progress server running on http://{host}:{port}")
    return server