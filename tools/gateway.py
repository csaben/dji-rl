#!/usr/bin/env python3
"""Authenticated public gateway for the Flight Lab app (for phones without Tailscale), meant to sit
behind the Cloudflare tunnel:

    internet -> Cloudflare -> cloudflared (cradle) -> 127.0.0.1:8979 (this) -> planner :8968 / pilot :8971

Everything requires a key (created on first run, never logged): the owner key ~/.config/dji-gateway/token
can do everything; the guest key ~/.config/dji-gateway/guest_token can use the app, sim and local splats
but not World Labs splats, race training or the live pilot link (403, see guest_denied). Keys are sent as:
  - native calls:  `Authorization: Bearer <token>`
  - WebView/browser: open `/login?key=<token>&next=/` once (or the form at /login); it sets an HttpOnly
    Secure cookie and redirects, so pages, <img src=/sim/frame...> and WebSockets just work afterwards
  - `?key=<token>` is also accepted on any request (stripped before forwarding)
Unauthenticated: 401 (HTML login form for browser navigations). Repeated failures from one client IP
(CF-Connecting-IP) get 429 for a while.

Routes:
  /pilot          WebSocket -> ws://100.80.98.4:8971/ (the pilot link, docs/pilot-protocol.md)
  /upload         -> http://100.80.98.4:8966/upload (dji-upload; use &offset=&total= parts, Cloudflare caps
                     request bodies at 100 MB)
  /healthz        no auth, "ok"
  everything else -> http://127.0.0.1:8968 (planner: app pages, /sim/* sim viewer + splat jobs, missions,
                     videos), HTTP and WebSocket, streamed both ways

    .venv/bin/python tools/gateway.py --port 8979
"""
from __future__ import annotations

import argparse
import asyncio
import hmac
import secrets
import time
from collections import defaultdict, deque
from pathlib import Path
from urllib.parse import urlencode

import aiohttp
from aiohttp import web

TOKEN_FILE = Path.home() / ".config" / "dji-gateway" / "token"            # owner: everything
GUEST_FILE = Path.home() / ".config" / "dji-gateway" / "guest_token"      # guest: see GUEST_DENY
COOKIE = "djigw"
HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
       "transfer-encoding", "upgrade", "host", "content-length", "expect"}

LOGIN = """<!doctype html><meta name=viewport content="width=device-width,initial-scale=1">
<title>Flight Lab</title><body style="font:16px system-ui;max-width:420px;margin:15vh auto;padding:0 16px">
<h3>Flight Lab</h3><form method=post action=/login><input type=hidden name=next value="{next}">
<input name=key type=password placeholder="access key" autofocus style="width:100%;padding:10px;font-size:16px">
<button style="margin-top:10px;padding:10px 16px;font-size:16px">Enter</button></form>{err}</body>"""


def load_token(f: Path = TOKEN_FILE) -> str:
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        f.touch(mode=0o600)
        f.write_text(secrets.token_urlsafe(32))
    f.chmod(0o600)
    return f.read_text().strip()


def guest_denied(method: str, path: str, body: bytes) -> str | None:
    """things a guest key can't do: anything that spends money, burns long GPU time or flies the drone"""
    if path == "/pilot":
        return "the live pilot link is owner-only"
    if method == "POST" and path.endswith("/race/train"):
        return "race training is owner-only"
    if method == "POST" and "/splat" in path and b"worldlabs" in body.lower():
        return "World Labs splats cost money: owner-only (local splats are fine)"
    return None


class Gate:
    def __init__(self, keys: dict[str, str], planner: str, pilot: str, upload: str):
        self.keys, self.planner, self.pilot, self.upload = keys, planner.rstrip("/"), pilot, upload.rstrip("/")
        self.fails: dict[str, deque] = defaultdict(deque)
        self.http: aiohttp.ClientSession | None = None

    # ------------------------------------------------------------------ auth
    def ok(self, s: str | None) -> str | None:
        """key -> role ('owner' | 'guest') or None"""
        if not s:
            return None
        hit = None
        for k, role in self.keys.items():
            if hmac.compare_digest(s.encode(), k.encode()):
                hit = role
        return hit

    def client_ip(self, r: web.Request) -> str:
        return r.headers.get("CF-Connecting-IP") or r.remote or "?"

    def blocked(self, ip: str) -> bool:
        q = self.fails[ip]
        while q and q[0] < time.time() - 600:
            q.popleft()
        return len(q) >= 20

    def authed(self, r: web.Request) -> str | None:
        h = r.headers.get("Authorization", "")
        return (self.ok(h[7:].strip()) if h.lower().startswith("bearer ") else None) \
            or self.ok(r.cookies.get(COOKIE)) or self.ok(r.query.get("key"))

    def set_cookie(self, resp: web.StreamResponse, key: str):
        resp.set_cookie(COOKIE, key, max_age=180 * 86400, httponly=True, secure=True, samesite="Lax", path="/")

    @staticmethod
    def safe_next(n: str | None) -> str:
        return n if n and n.startswith("/") and not n.startswith("//") else "/"

    async def login(self, r: web.Request):
        ip = self.client_ip(r)
        if self.blocked(ip):
            return web.Response(status=429, text="too many attempts, try again later")
        if r.method == "POST":
            f = await r.post()
            key, nxt = f.get("key"), self.safe_next(f.get("next"))
        else:
            key, nxt = r.query.get("key"), self.safe_next(r.query.get("next"))
        if key is None:
            return web.Response(text=LOGIN.format(next=nxt, err=""), content_type="text/html")
        if not self.ok(key):
            self.fails[ip].append(time.time())
            return web.Response(status=401, text=LOGIN.format(next=nxt, err="<p style=color:#c33>wrong key</p>"),
                                content_type="text/html")
        resp = web.HTTPFound(nxt)
        self.set_cookie(resp, key)
        return resp

    # ------------------------------------------------------------------ proxy
    async def handle(self, r: web.Request):
        if r.path == "/healthz":
            return web.Response(text="ok")
        if r.path == "/login":
            return await self.login(r)
        ip = self.client_ip(r)
        if self.blocked(ip):
            return web.Response(status=429, text="too many attempts, try again later")
        role = self.authed(r)
        if not role:
            if r.headers.get("Authorization") or r.query.get("key") or r.cookies.get(COOKIE):
                self.fails[ip].append(time.time())      # count wrong keys, not missing ones
            if r.method == "GET" and "text/html" in r.headers.get("Accept", ""):
                return web.Response(status=401, text=LOGIN.format(next=r.path_qs, err=""), content_type="text/html")
            return web.json_response({"error": "unauthorized"}, status=401)
        query = {k: v for k, v in r.query.items() if k != "key"}
        qs = ("?" + urlencode(query)) if query else ""
        body = None
        if role != "owner":
            if r.method == "POST" and r.body_exists:
                body = await r.read()
            why = guest_denied(r.method, r.path, body or b"")
            if why:
                return web.json_response({"error": why, "ok": False}, status=403)
        if r.path == "/pilot":
            return await self.ws(r, self.pilot + qs)
        if r.path == "/upload":
            return await self.forward(r, self.upload + r.path + qs, body)
        if r.headers.get("Upgrade", "").lower() == "websocket":
            return await self.ws(r, self.planner.replace("http", "ws", 1) + r.path + qs)
        return await self.forward(r, self.planner + r.path + qs, body)

    async def forward(self, r: web.Request, url: str, body: bytes | None = None):
        hdrs = {k: v for k, v in r.headers.items() if k.lower() not in HOP and k.lower() not in ("authorization", "cookie")}
        if body is None and r.content_length is not None:
            hdrs["Content-Length"] = str(r.content_length)   # the planner reads bodies by Content-Length, not chunked
        hdrs["X-Forwarded-Proto"] = "https"
        hdrs["X-Forwarded-For"] = self.client_ip(r)
        data = body if body is not None else (r.content if r.body_exists else None)
        try:
            async with self.http.request(r.method, url, headers=hdrs, data=data, allow_redirects=False) as up:
                resp = web.StreamResponse(status=up.status)
                for k, v in up.headers.items():
                    if k.lower() not in HOP and k.lower() != "content-encoding":
                        resp.headers.add(k, v)
                if r.query.get("key") and self.ok(r.query.get("key")):
                    self.set_cookie(resp, r.query["key"])
                if up.headers.get("Content-Length") and "Content-Encoding" not in up.headers:
                    resp.content_length = int(up.headers["Content-Length"])
                await resp.prepare(r)
                async for chunk in up.content.iter_chunked(1 << 16):
                    await resp.write(chunk)
                await resp.write_eof()
                return resp
        except aiohttp.ClientError as e:
            return web.json_response({"error": f"upstream unavailable: {type(e).__name__}"}, status=502)

    async def ws(self, r: web.Request, url: str):
        client = web.WebSocketResponse(max_msg_size=16 * 1024 * 1024, heartbeat=20)
        await client.prepare(r)
        try:
            async with self.http.ws_connect(url, max_msg_size=16 * 1024 * 1024) as up:
                async def pump(src, dst):
                    async for m in src:
                        if m.type == aiohttp.WSMsgType.TEXT:
                            await dst.send_str(m.data)
                        elif m.type == aiohttp.WSMsgType.BINARY:
                            await dst.send_bytes(m.data)
                        else:
                            break
                    await dst.close()
                await asyncio.gather(pump(client, up), pump(up, client), return_exceptions=True)
        except aiohttp.ClientError:
            await client.close(code=1011, message=b"upstream unavailable")
        return client


async def main_async(a):
    g = Gate({load_token(TOKEN_FILE): "owner", load_token(GUEST_FILE): "guest"}, a.planner, a.pilot, a.upload)
    g.http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=600),
                                   auto_decompress=False)
    app = web.Application(client_max_size=4 * 1024 ** 3)
    app.router.add_route("*", "/{tail:.*}", g.handle)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    await web.TCPSite(runner, a.host, a.port).start()
    print(f"gateway on http://{a.host}:{a.port} -> {a.planner}, /pilot -> {a.pilot}", flush=True)
    await asyncio.Event().wait()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8979)
    ap.add_argument("--planner", default="http://127.0.0.1:8968")
    ap.add_argument("--pilot", default="ws://100.80.98.4:8971/")
    ap.add_argument("--upload", default="http://100.80.98.4:8966")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
