import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
import os
import sys
from pathlib import Path

# Ensure we can import `rag_core.py` regardless of the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rag_core import ask_llm, build_prompt, retrieve_chunks  # noqa: E402


def _send_json(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))

    # CORS for local React dev.
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")

    handler.end_headers()
    handler.wfile.write(data)


class ApiHandler(BaseHTTPRequestHandler):
    server_version = "DevOpsRAGApi/1.0"

    def do_OPTIONS(self):  # noqa: N802 (BaseHTTPRequestHandler naming)
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):  # noqa: N802
        if self.path == "/api/health":
            _send_json(self, 200, {"status": "ok"})
            return

        _send_json(self, 404, {"error": "Not found"})

    def do_POST(self):  # noqa: N802
        if self.path != "/api/ask":
            _send_json(self, 404, {"error": "Not found"})
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_length) if content_length > 0 else b""
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            _send_json(self, 400, {"error": "Invalid JSON"})
            return

        question = str(body.get("question") or "").strip()
        top_k = int(body.get("top_k") or 5)
        debug = bool(body.get("debug") or False)

        if not question:
            _send_json(self, 400, {"error": "Missing 'question' field"})
            return

        try:
            retrieved = retrieve_chunks(question, top_k=top_k)
            chunks = retrieved["documents"]
            metadatas = retrieved["metadatas"]
            distances = retrieved["distances"]

            prompt = build_prompt(question, chunks)
            answer = ask_llm(prompt)

            response: dict[str, Any] = {
                "answer": answer,
                "retrieval": {
                    "n_chunks": len(chunks),
                    "best_distance": distances[0] if distances else None,
                },
            }

            if debug:
                response["retrieval"].update(
                    {
                        "chunks": chunks,
                        "metadatas": metadatas,
                        "distances": distances,
                        "prompt": prompt,
                    }
                )

            _send_json(self, 200, response)
        except Exception as e:
            _send_json(self, 500, {"error": str(e)})


def main() -> None:
    port = int(os.getenv("RAG_API_PORT", "8000"))
    host = os.getenv("RAG_API_HOST", "0.0.0.0")

    httpd = ThreadingHTTPServer((host, port), ApiHandler)
    print(f"RAG API listening on http://{host}:{port}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()

