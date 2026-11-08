"""Minimal World Labs Atlas client over the local ac proxy (mirrors @worldlabsai/client@0.5.0), plus the
two reconstruction tasks we use: images2PosedRGBD -> posedImages2Splats. See docs/worldlabs.md.

The proxy token comes from .env (WORLDLABS_BASE_URL, WORLDLABS_API_KEY) and is never printed or written.
Every request/response is saved (minus image bytes) under <workdir>/api/ for auditing.
"""
from __future__ import annotations

import base64
import json
import time
import uuid
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[1]


class Stop(Exception):
    """The ac proxy refused (approval needed, no grant, key host down): only Clark can fix it."""


def _env():
    d = {}
    for line in (REPO / ".env").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            d[k.strip()] = v.strip().strip('"').strip("'")
    return d


class Atlas:
    def __init__(self, workdir: Path, log=print):
        e = _env()
        self.base = e["WORLDLABS_BASE_URL"].rstrip("/")
        self.s = requests.Session()
        self.s.headers["Authorization"] = "Bearer " + e["WORLDLABS_API_KEY"]
        self.api = Path(workdir) / "api"
        self.api.mkdir(parents=True, exist_ok=True)
        self.log = log

    def save(self, name, obj):
        (self.api / f"{name}.json").write_text(json.dumps(obj, indent=1))

    def req(self, method, path, body=None, query=None, headers=None, tag=None, timeout=120, retries=3):
        for attempt in range(retries + 1):
            r = self.s.request(method, self.base + path, json=body, params=query, headers=headers or {}, timeout=timeout)
            txt = r.text
            try:
                j = r.json()
            except ValueError:
                j = {"_raw": txt[:2000]}
            low = txt[:2000].lower()
            for code in ("approval_required", "no_grant", "key_host_unreachable"):
                if code in low:
                    if tag:
                        self.save(tag + ".ERR", {"status": r.status_code, "body": j})
                    raise Stop(f"ac proxy refused: {code} (HTTP {r.status_code})")
            if r.status_code in (429, 502, 503, 504) and attempt < retries:
                ra = r.headers.get("retry-after")
                wait = float(ra) if ra and ra.replace(".", "").isdigit() else 2 ** attempt * 2
                if wait > 10 and r.status_code == 429:
                    break
                time.sleep(wait)
                continue
            break
        if tag:
            self.save(tag, {"status": r.status_code, "body": j})
        if not r.ok:
            raise RuntimeError(f"{method} {path} -> HTTP {r.status_code}: {txt[:800]}")
        return j

    def wait(self, op_id, tag, poll=15, max_s=4 * 3600, on_poll=None):
        t0 = time.time()
        while True:
            try:
                op = self.req("GET", f"/api/v2/operations/{op_id}", query={"wait": "true"}, timeout=90)
            except RuntimeError as e:
                if any(f"HTTP {c}" in str(e) for c in (502, 503, 504)):
                    time.sleep(poll)
                    continue
                raise
            if op.get("done"):
                self.save(tag + ".result", op)
                return op
            if on_poll:
                on_poll(time.time() - t0)
            if time.time() - t0 > max_s:
                raise RuntimeError(f"timeout waiting for {tag}")
            time.sleep(poll)

    # ------------------------------------------------------------------ tasks
    def posed_rgbd(self, images: list[Path], mime="image/webp", key=None, on_poll=None):
        """unposed frames (all the same grid, e.g. 1280x720) -> PosedRGBD frames (camera + depth), in order."""
        fr = [{"imageAsset": {"base64": f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}"}}
              for p in images]
        body = {"frames": fr, "targetResolution": [1280, 720], "refinePoses": True,
                "canonicalizeToFirstFrame": False, "base64Outputs": False}
        self.save("images2PosedRGBD.request", dict(body, frames=[f"<{p.name}>" for p in images]))
        op = self.req("POST", "/api/v2/tasks:images2PosedRGBD", body, query={"wait": "true"},
                      headers={"Idempotency-Key": key or f"djirl-i2p-{uuid.uuid4().hex[:12]}"},
                      tag="images2PosedRGBD.submit", timeout=300, retries=0)
        done = op if op.get("done") else self.wait(op["id"], "images2PosedRGBD", on_poll=on_poll)
        self.save("images2PosedRGBD.result", done)
        if done.get("error"):
            raise RuntimeError(f"images2PosedRGBD failed: {json.dumps(done['error'])[:500]}")
        return done["response"]["frames"]

    def splats(self, posed_frames, prompt: str, refine=True, key=None, on_poll=None):
        aid = lambda a: {"assetId": a["assetId"]}
        posed = [{"imageAsset": aid(f["imageAsset"]), "camera": f["camera"],
                  "depth": {"depthAsset": aid(f["depth"]["depthAsset"]),
                            **({"confidenceAsset": aid(f["depth"]["confidenceAsset"])}
                               if f["depth"].get("confidenceAsset") else {})}} for f in posed_frames]
        body = {"posedFrames": posed, "refine": refine, "prompt": prompt, "enhancePrompt": True}
        tag = "posedImages2Splats"
        self.save(tag + ".request", body)
        op = self.req("POST", "/api/v2/tasks:posedImages2Splats", body,
                      headers={"Idempotency-Key": key or f"djirl-p2s-{uuid.uuid4().hex[:12]}"},
                      tag=tag + ".submit", timeout=120, retries=0)
        done = op if op.get("done") else self.wait(op["id"], tag, poll=30, on_poll=on_poll)
        if done.get("error"):
            raise RuntimeError(f"posedImages2Splats failed: {json.dumps(done['error'])[:500]}")
        resp = done["response"]
        assets = list(resp.get("refinedSplats", {}).items()) or ([("draft", resp["draftSplat"])] if resp.get("draftSplat") else [])
        if not assets:
            raise RuntimeError(f"posedImages2Splats returned no splats: {list(resp)}")
        return assets  # [(name, TaskAsset with signed url)]

    def download(self, task_asset, dest: Path):
        url = task_asset.get("url")
        if not url and task_asset.get("assetId"):
            url = self.req("POST", f"/api/v2/assets/{task_asset['assetId']}:createReadUrl", {})["readUrl"]
        r = requests.get(url, timeout=900)  # signed URL: no API key
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(r.content)
        return dest
