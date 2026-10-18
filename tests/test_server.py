"""Chat UI server: streaming answers, uploads, static files, and the localhost-only protections."""

import http.client
import json
import threading
from urllib.parse import quote

import pytest

from locallens import config
from locallens.server import Backend, make_server, safe_upload_name


class FakeBackend(Backend):
    def __init__(self):
        super().__init__(config.Config())
        self.ingested = []

    def ask(self, question, rerank=None):
        yield {"type": "sources", "sources": [{"n": 1, "document": "a.pdf", "page": 2, "snippet": "x",
                                               "source": "textlayer", "confidence": None}]}
        for t in ["The answer ", "is 42 [1]."]:
            yield {"type": "token", "text": t}
        yield {"type": "done", "cited": [1], "bogus": [], "rerank": rerank, "q": question}

    def ingest(self, path, vlm_mode, log):
        self.ingested.append((path, vlm_mode, path.read_bytes()))
        log("[bold]indexed[/] " + path.name)
        return {"files_indexed": 1, "chunks": 3, "errors": []}

    def documents(self):
        return [{"name": "a.pdf", "pages": 2, "digital_pages": 2, "chunks": 9, "indexed": "2026-10-18T00:00"}]

    def health(self):
        return {"ok": True, "documents": 3, "chunks": 9}


@pytest.fixture
def env(tmp_path):
    web = tmp_path / "web"
    web.mkdir()
    (web / "index.html").write_text("<html>chat</html>")
    (web / "main.dart.js").write_text("console.log(1)")
    (tmp_path / "secret.txt").write_text("nope")
    backend = FakeBackend()
    srv = make_server(backend, port=0, web_dir=web, uploads_dir=tmp_path / "uploads")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv, backend, tmp_path / "uploads"
    srv.shutdown()
    srv.server_close()


def request(srv, method, path, body=None, host=None, ctype="application/json"):
    port = srv.server_address[1]
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    headers = {"Host": host or f"127.0.0.1:{port}"}
    if body is not None:
        headers["Content-Type"] = ctype
        if not isinstance(body, bytes):
            body = json.dumps(body).encode()
    conn.request(method, path, body=body, headers=headers)
    resp = conn.getresponse()
    return resp.status, resp.getheader("Content-Type"), resp.read()


def events(raw):
    return [json.loads(line) for line in raw.decode().splitlines()]


def test_ask_streams_ndjson_events(env):
    srv, _, _ = env
    code, ctype, raw = request(srv, "POST", "/api/ask", {"question": " what? ", "rerank": False})
    assert code == 200 and ctype == "application/x-ndjson"
    evs = events(raw)
    assert [e["type"] for e in evs] == ["sources", "token", "token", "done"]
    assert "".join(e["text"] for e in evs if e["type"] == "token") == "The answer is 42 [1]."
    assert evs[-1]["q"] == "what?" and evs[-1]["rerank"] is False


def test_upload_saves_and_indexes(env):
    srv, backend, uploads = env
    url = "/api/upload?name=" + quote("my notes.pdf") + "&vlm=all-scans"
    code, _, raw = request(srv, "POST", url, b"%PDF-fake", ctype="application/octet-stream")
    assert code == 200
    evs = events(raw)
    assert [e["type"] for e in evs] == ["saved", "log", "done"]
    assert evs[1]["text"] == "indexed my notes.pdf"   # rich markup stripped
    assert evs[-1]["report"]["chunks"] == 3
    path, vlm, data = backend.ingested[0]
    assert path == uploads / "my notes.pdf" and vlm == "all-scans" and data == b"%PDF-fake"
    assert not list(uploads.glob(".upload-*"))   # temp file cleaned up


def test_upload_same_name_different_content_gets_new_name(env):
    srv, _, uploads = env
    for body in (b"one", b"two", b"one"):
        request(srv, "POST", "/api/upload?name=scan.png", body, ctype="application/octet-stream")
    assert sorted(p.name for p in uploads.iterdir()) == ["scan-2.png", "scan.png"]


def test_upload_rejections(env):
    srv, backend, _ = env
    octet = "application/octet-stream"
    assert request(srv, "POST", "/api/upload?name=a.exe", b"x", ctype=octet)[0] == 400
    assert request(srv, "POST", "/api/upload?name=a.pdf&vlm=bogus", b"x", ctype=octet)[0] == 400
    assert request(srv, "POST", "/api/upload?name=a.pdf", b"x", ctype="text/plain")[0] == 415
    assert request(srv, "POST", "/api/upload?name=a.pdf", b"x", ctype=octet, host="evil.example")[0] == 403
    assert backend.ingested == []


def test_safe_upload_name_strips_folders():
    assert safe_upload_name("../../etc/passwd.pdf") == "passwd.pdf"
    assert safe_upload_name("C:\\Users\\me\\scan.JPG") == "scan.JPG"
    assert safe_upload_name(".hidden.pdf") is None
    assert safe_upload_name("notes.txt") is None


def test_health_documents_and_static_files(env):
    srv, _, _ = env
    code, _, raw = request(srv, "GET", "/api/health")
    assert code == 200 and json.loads(raw)["documents"] == 3
    assert json.loads(request(srv, "GET", "/api/documents")[2])[0]["name"] == "a.pdf"
    code, ctype, raw = request(srv, "GET", "/")
    assert code == 200 and raw == b"<html>chat</html>" and ctype.startswith("text/html")
    code, ctype, _ = request(srv, "GET", "/main.dart.js")
    assert code == 200 and ctype == "text/javascript"
    assert request(srv, "GET", "/some/route")[2] == b"<html>chat</html>"   # client-side route
    assert request(srv, "GET", "/missing.js")[0] == 404


def test_rejects_foreign_host_header(env):
    srv, _, _ = env
    # what a DNS-rebinding page would send
    assert request(srv, "GET", "/api/health", host="evil.example:8765")[0] == 403
    assert request(srv, "POST", "/api/ask", {"question": "x"}, host="evil.example")[0] == 403


def test_rejects_non_json_and_bad_bodies(env):
    srv, _, _ = env
    assert request(srv, "POST", "/api/ask", {"question": "x"}, ctype="text/plain")[0] == 415
    assert request(srv, "POST", "/api/ask", {"nope": 1})[0] == 400
    assert request(srv, "POST", "/api/ask", {"question": "   "})[0] == 400
    assert request(srv, "POST", "/api/other", {"question": "x"})[0] == 404


def test_no_path_traversal(env):
    srv, _, _ = env
    code, _, raw = request(srv, "GET", "/../secret.txt")
    assert b"nope" not in raw and code == 404


def test_errors_mid_stream_become_events(tmp_path):
    class Broken(FakeBackend):
        def ask(self, question, rerank=None):
            yield {"type": "sources", "sources": []}
            raise RuntimeError("model crashed")

    srv = make_server(Broken(), port=0, web_dir=tmp_path)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        _, _, raw = request(srv, "POST", "/api/ask", {"question": "x"})
        assert events(raw)[-1] == {"type": "error", "message": "RuntimeError: model crashed"}
        # no build yet → the instructions page instead of a crash
        assert b"not built yet" in request(srv, "GET", "/")[2]
    finally:
        srv.shutdown()
        srv.server_close()
