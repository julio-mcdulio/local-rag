"""MCP server tests: JWT (JWKS) auth end-to-end, offline.

A local RSA key pair + in-process JWKS HTTP stand-in simulate an external
IdP. The server app is exercised with Starlette's TestClient (no network
beyond localhost JWKS fetch).
"""

import http.server
import json
import threading
import time
from datetime import datetime, timedelta, timezone

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from starlette.testclient import TestClient

from conftest import make_config, make_registry
from rag.embeddings import slugify
from rag.mcp_server import JwtAuth, build_app
from rag.store import Store
from rag.text_extract import OcrClient

ISSUER = "https://idp.example.com"
AUDIENCE = "rag-mcp"


def _make_keypair(kid: str = "test-key"):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = json.loads(pyjwt.algorithms.RSAAlgorithm.to_jwk(key.public_key()))
    jwk["kid"] = kid
    jwk["alg"] = "RS256"
    jwk["use"] = "sig"
    return key, jwk


class JwksServer:
    """Serves a JWKS JSON document on a localhost port."""

    def __init__(self, jwk: dict):
        payload = json.dumps({"keys": [jwk]}).encode()

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *a):
                pass

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(
            target=self.httpd.serve_forever, daemon=True
        )
        self.thread.start()

    def uri(self):
        return f"http://127.0.0.1:{self.port}/.well-known/jwks.json"

    def stop(self):
        self.httpd.shutdown()
        self.thread.join(timeout=5)


@pytest.fixture(scope="module")
def jwks():
    key, jwk = _make_keypair()
    server = JwksServer(jwk)
    yield key, server
    server.stop()


def _token(key, claims=None, **over):
    now = datetime.now(timezone.utc)
    c = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "sub": "pi-agent",
        "iat": now,
        "exp": now + timedelta(hours=1),
    }
    c.update(claims or {})
    c.update(over)
    return pyjwt.encode(c, key, algorithm="RS256", headers={"kid": "test-key"})


def test_jwt_verify_accepts_valid(jwks):
    key, server = jwks
    auth = JwtAuth(server.uri(), audience=AUDIENCE, issuer=ISSUER)
    assert auth.verify(_token(key)) is True


def test_jwt_verify_rejects_wrong_issuer(jwks):
    key, server = jwks
    auth = JwtAuth(server.uri(), audience=AUDIENCE, issuer="https://other-idp")
    assert auth.verify(_token(key)) is False


def test_jwt_verify_rejects_wrong_audience(jwks):
    key, server = jwks
    auth = JwtAuth(server.uri(), audience="something-else", issuer=ISSUER)
    assert auth.verify(_token(key)) is False


def test_jwt_verify_rejects_expired(jwks):
    key, server = jwks
    now = datetime.now(timezone.utc)
    tok = _token(key, exp=now - timedelta(hours=1), iat=now - timedelta(hours=2))
    auth = JwtAuth(server.uri(), audience=AUDIENCE, issuer=ISSUER)
    assert auth.verify(tok) is False


def test_jwt_verify_rejects_garbage(jwks):
    key, server = jwks
    auth = JwtAuth(server.uri(), audience=AUDIENCE, issuer=ISSUER)
    assert auth.verify("not-a-token") is False


@pytest.fixture
def mcp_env(tmp_path, jwks):
    cfg = make_config(str(tmp_path))
    store = Store(cfg.db_path_obj)
    providers = make_registry()
    ocr = OcrClient("http://localhost:11434", "maternion/LightOnOCR-2")
    yield cfg, store, providers, ocr
    store.close()


def test_mcp_auth_middleware_401(mcp_env, jwks):
    cfg, store, providers, ocr = mcp_env
    key, server = jwks
    cfg.auth_jwks_uri = server.uri()
    cfg.auth_issuer = ISSUER
    cfg.auth_audience = AUDIENCE

    app = build_app(cfg, store, providers)
    with TestClient(app) as client:
        # no token
        assert client.get("/mcp").status_code == 401
        # bad token
        assert client.get("/mcp", headers={"Authorization": "Bearer nope"}).status_code == 401
        # wrong-issuer token
        other_key, _ = _make_keypair(kid="other")
        bad = pyjwt.encode(
            {"iss": "https://other-idp", "aud": AUDIENCE, "sub": "x",
             "iat": datetime.now(timezone.utc), "exp": datetime.now(timezone.utc) + timedelta(hours=1)},
            other_key, algorithm="RS256",
        )
        r = client.get("/mcp", headers={"Authorization": f"Bearer {bad}"})
        assert r.status_code == 401
        # valid token passes the middleware (MCP handshake may 4xx, but not 401)
        good = _token(key)
        r = client.get("/mcp", headers={"Authorization": f"Bearer {good}"})
        assert r.status_code != 401
        # health is not behind auth
        assert client.get("/health").status_code == 200


def test_mcp_no_auth_when_unconfigured(mcp_env):
    cfg, store, providers, ocr = mcp_env
    # no auth_jwks_uri -> middleware absent
    assert cfg.auth_jwks_uri is None
    app = build_app(cfg, store, providers)
    with TestClient(app) as client:
        r = client.get("/mcp")
        assert r.status_code != 401
