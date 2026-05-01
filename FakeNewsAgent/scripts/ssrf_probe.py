"""SSRF vulnerability probe for cross_modal_tool._ensure_base64_uri.

Phase 1 — PRE-FIX:
  Starts a real HTTP server on 127.0.0.1, calls _ensure_base64_uri with that
  URL, and proves the server's payload was fetched and returned as base64.
  Also uses mocked DNS to show private/IMDS addresses are not blocked.

Phase 2 — POST-FIX (run after _validate_image_url is implemented):
  Tests _validate_image_url directly — it raises ValueError on blocked URLs.
  Also confirms _ensure_base64_uri returns None (not raises) for blocked URLs,
  which is the correct caller-fallback behaviour.

Run:
    python scripts/ssrf_probe.py

─────────────────────────────────────────────────────────────
FUTURE CONSIDERATIONS FOR THE _validate_image_url FIX
─────────────────────────────────────────────────────────────

1. DNS rebinding (not covered by current fix)
   The current guard resolves the hostname once before the request. An
   attacker who controls a DNS server can return a public IP at validation
   time, then flip the record to 127.0.0.1 before urllib opens the TCP
   connection. Mitigation: pass the pre-resolved IP directly as the
   connection target (via a custom HTTPHandler) rather than letting urllib
   re-resolve the hostname at connect time.

2. HTTP redirect following
   urllib.request.urlopen follows 301/302 redirects by default. A server
   at a public IP could redirect to http://192.168.1.1/secret and bypass
   the one-time URL check. Mitigation: disable redirect following
   (build a custom opener with no HTTPRedirectHandler) or re-run
   _validate_image_url on every Location header before following.

3. Response size and content-type not validated
   A malicious server could return a 2 GB response or a non-image payload.
   The current code reads the full body into memory. Mitigation: add a
   Content-Length check (reject > ~10 MB) and verify Content-Type starts
   with "image/" before decoding.

4. IPv6 coverage
   _validate_image_url checks all getaddrinfo records, so IPv6 private
   ranges (::1, fc00::/7, fe80::/10) are covered. However, some
   environments return only IPv4 records. Verify getaddrinfo behaviour
   on the deployment host if IPv6 dual-stack is in use.
"""

import base64
import http.server
import socket
import sys
import threading
from pathlib import Path
from unittest.mock import patch

# ── Path setup so fact_check_agent is importable ──────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scraper_preprocessing_memory"))

from fact_check_agent.src.tools.cross_modal_tool import (  # noqa: E402
    _ensure_base64_uri,
    _validate_image_url,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def ok(msg):    print(f"  [PASS] {msg}")
def fail(msg):  print(f"  [FAIL] {msg}")
def section(title): print(f"\n{'='*60}\n  {title}\n{'='*60}")


def fake_resolve(ip):
    """Fake getaddrinfo result for a given IP — no real DNS lookup."""
    return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", (ip, 0))]


# ── Local HTTP server ─────────────────────────────────────────────────────────

SECRET = b"SENSITIVE_CREDENTIAL=s3cr3t_key_12345"


class _SecretHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(SECRET)))
        self.end_headers()
        self.wfile.write(SECRET)

    def log_message(self, *args):
        pass


def start_local_server():
    server = http.server.HTTPServer(("127.0.0.1", 0), _SecretHandler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, port


# ── Phase 1: demonstrate the vulnerability ───────────────────────────────────

def run_pre_fix_probes():
    section("PHASE 1 — PRE-FIX: demonstrating SSRF vulnerability")
    results = []

    # ── Probe 1: real loopback fetch ─────────────────────────────────────────
    print("\nProbe 1: fetch from 127.0.0.1 (real loopback HTTP server)")
    server, port = start_local_server()
    url = f"http://127.0.0.1:{port}/secret"
    try:
        result = _ensure_base64_uri(url)
        if result:
            decoded = base64.b64decode(result.split(",", 1)[1])
            if SECRET in decoded:
                ok(f"Fetched loopback server — payload recovered: {decoded.decode()!r}")
                results.append(True)
            else:
                fail(f"Response came back but payload missing. Got: {decoded[:80]}")
                results.append(False)
        else:
            fail("_ensure_base64_uri returned None — fetch was blocked (already fixed?)")
            results.append(False)
    except ValueError as e:
        fail(f"Blocked by validation (already fixed?): {e}")
        results.append(False)
    except Exception as e:
        fail(f"Unexpected error: {e}")
        results.append(False)
    finally:
        server.shutdown()

    # ── Probe 2: private IP 192.168.x.x (mocked DNS) ─────────────────────────
    print("\nProbe 2: DNS resolves to 192.168.1.1 (private RFC 1918)")
    with (
        patch("socket.getaddrinfo", return_value=fake_resolve("192.168.1.1")),
        patch("urllib.request.urlopen") as mock_open,
    ):
        mock_open.return_value.__enter__ = lambda s: s
        mock_open.return_value.__exit__ = lambda *a: False
        mock_open.return_value.headers.get = lambda *a, **kw: "text/plain"
        mock_open.return_value.read.return_value = b"router-admin-panel-data"
        try:
            _ensure_base64_uri("http://internal.corp/logo.png")
            if mock_open.called:
                ok("urlopen WAS called — private IP 192.168.1.1 not blocked")
                results.append(True)
            else:
                fail("urlopen not reached (already protected?)")
                results.append(False)
        except ValueError as e:
            fail(f"Blocked by validation (already fixed?): {e}")
            results.append(False)

    # ── Probe 3: AWS IMDS 169.254.169.254 (mocked DNS) ───────────────────────
    print("\nProbe 3: DNS resolves to 169.254.169.254 (AWS/GCP metadata endpoint)")
    with (
        patch("socket.getaddrinfo", return_value=fake_resolve("169.254.169.254")),
        patch("urllib.request.urlopen") as mock_open,
    ):
        mock_open.return_value.__enter__ = lambda s: s
        mock_open.return_value.__exit__ = lambda *a: False
        mock_open.return_value.headers.get = lambda *a, **kw: "text/plain"
        mock_open.return_value.read.return_value = b"ami-id=ami-0abcdef iam-security-credentials/..."
        try:
            _ensure_base64_uri("http://169.254.169.254/latest/meta-data/")
            if mock_open.called:
                ok("urlopen WAS called — IMDS 169.254.169.254 not blocked (credentials would leak)")
                results.append(True)
            else:
                fail("urlopen not reached")
                results.append(False)
        except ValueError as e:
            fail(f"Blocked by validation (already fixed?): {e}")
            results.append(False)

    # ── Probe 4: file:// scheme ───────────────────────────────────────────────
    print("\nProbe 4: file:// scheme (local filesystem access)")
    try:
        result = _ensure_base64_uri("file:///etc/passwd")
        if result is not None:
            ok("No error raised and returned content — file:// not scheme-blocked")
            results.append(True)
        else:
            fail("Returned None — scheme was blocked (already fixed?)")
            results.append(False)
    except ValueError as e:
        fail(f"Blocked at scheme check (already fixed?): {e}")
        results.append(False)
    except Exception as e:
        ok(f"Reached urlopen before failing — file:// not scheme-blocked. Error: {type(e).__name__}")
        results.append(True)

    # ── Summary ───────────────────────────────────────────────────────────────
    passed = sum(results)
    print(f"\n  Vulnerabilities confirmed: {passed}/{len(results)}")
    if passed == len(results):
        print("  ✗ System is VULNERABLE — all SSRF probes succeeded\n")
    else:
        print("  Partial results — some probes may already be patched\n")

    return results


# ── Phase 2: verify the fix blocks all restricted addresses ──────────────────

def run_post_fix_probes():
    """Tests _validate_image_url directly — it raises ValueError on blocked URLs.
    _ensure_base64_uri wraps it in try/except and returns None (correct pipeline
    behaviour), so we test the guard itself rather than the wrapper.
    """
    section("PHASE 2 — POST-FIX: verifying SSRF is blocked")
    results = []

    cases = [
        ("127.0.0.1 loopback",   "127.0.0.1",       "http://localhost/img.png"),
        ("10.x.x.x private",     "10.0.0.5",        "http://internal.corp/img.png"),
        ("192.168.x.x private",  "192.168.1.100",   "http://router.local/admin"),
        ("169.254.169.254 IMDS", "169.254.169.254", "http://169.254.169.254/latest/meta-data/"),
    ]

    for label, ip, url in cases:
        print(f"\nProbe: {label} ({ip})")
        with patch("socket.getaddrinfo", return_value=fake_resolve(ip)):
            try:
                _validate_image_url(url)
                fail(f"NOT blocked — {ip} was allowed through")
                results.append(False)
            except ValueError as e:
                ok(f"Blocked with ValueError: {e}")
                results.append(True)

    # file:// scheme (no DNS lookup — blocked at scheme check)
    print("\nProbe: file:// scheme")
    try:
        _validate_image_url("file:///etc/passwd")
        fail("file:// scheme NOT blocked")
        results.append(False)
    except ValueError as e:
        ok(f"Blocked with ValueError: {e}")
        results.append(True)

    # ftp:// scheme
    print("\nProbe: ftp:// scheme")
    try:
        _validate_image_url("ftp://files.example.com/img.jpg")
        fail("ftp:// scheme NOT blocked")
        results.append(False)
    except ValueError as e:
        ok(f"Blocked with ValueError: {e}")
        results.append(True)

    # Public IP must still be allowed
    print("\nProbe: public IP 93.184.216.34 (example.com) — must be ALLOWED")
    with patch("socket.getaddrinfo", return_value=fake_resolve("93.184.216.34")):
        try:
            _validate_image_url("https://example.com/image.jpg")
            ok("Public IP allowed through — not over-blocked")
            results.append(True)
        except ValueError as e:
            fail(f"Public IP incorrectly blocked: {e}")
            results.append(False)

    # Confirm _ensure_base64_uri returns None (not raises) for blocked URLs
    print("\nProbe: _ensure_base64_uri returns None for loopback (caller fallback check)")
    server, port = start_local_server()
    url = f"http://127.0.0.1:{port}/secret"
    try:
        result = _ensure_base64_uri(url)
        if result is None:
            ok("_ensure_base64_uri returned None — caller falls back to caption mode safely")
            results.append(True)
        else:
            fail("Still returned base64 payload — fix not wired into _ensure_base64_uri")
            results.append(False)
    except Exception as e:
        fail(f"Unexpected exception from _ensure_base64_uri: {e}")
        results.append(False)
    finally:
        server.shutdown()

    passed = sum(results)
    print(f"\n  Probes passed: {passed}/{len(results)}")
    if passed == len(results):
        print("  ✓ SSRF fully mitigated — all addresses blocked, public IPs allowed\n")
    else:
        print("  ✗ Fix incomplete — some probes still vulnerable\n")

    return results


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pre = run_pre_fix_probes()
    print("\n" + "─" * 60)
    post = run_post_fix_probes()
