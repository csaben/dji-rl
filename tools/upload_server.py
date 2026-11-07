#!/usr/bin/env python3
"""Tailnet-only upload page for phone footage -> data/raw/<date>/<source>/.

Streams each file straight to disk (no multipart parsing, no size limit beyond
free disk), writes to <name>.part and renames when complete. Binds to the
tailscale IP only, so it is not reachable from the LAN or internet.

Run: python3 tools/upload_server.py --host 100.80.98.4 --port 8966
"""
import argparse
import datetime as dt
import html
import json
import os
import re
import shutil
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent / "data" / "raw"
SOURCES = ("phone", "drone", "logs", "telemetry")
MIN_FREE = 5 * 1024**3  # refuse uploads that would leave < 5 GB free
CHUNK = 4 * 1024 * 1024
SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

PAGE = """<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>dji-rl upload</title>
<style>
:root{--bg:#fff;--fg:#111;--mut:#666;--line:#ddd;--acc:#2563eb;--ok:#16a34a;--bad:#dc2626}
@media (prefers-color-scheme:dark){:root{--bg:#111;--fg:#eee;--mut:#999;--line:#333;--acc:#60a5fa;--ok:#4ade80;--bad:#f87171}}
body{background:var(--bg);color:var(--fg);font:16px system-ui,sans-serif;margin:0;padding:16px;max-width:640px}
h1{font-size:20px;margin:0 0 4px}p{color:var(--mut);margin:0 0 16px}
label{display:block;margin:12px 0 4px;font-weight:600}
input,select,button{font:inherit;width:100%;box-sizing:border-box;padding:12px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--fg)}
button{background:var(--acc);color:#fff;border:0;margin-top:16px;font-weight:600}
button:disabled{opacity:.5}
.f{border:1px solid var(--line);border-radius:8px;padding:10px;margin-top:8px}
.n{word-break:break-all;font-size:14px}.s{font-size:13px;color:var(--mut)}
progress{width:100%;height:10px}.ok{color:var(--ok)}.bad{color:var(--bad)}
</style></head><body>
<h1>Upload footage to cradle</h1>
<p>Files go to dji-rl/data/raw/&lt;date&gt;/&lt;source&gt;/. Free: __FREE__.</p>
<label for="d">Session date</label><input id="d" type="date" value="__TODAY__">
<label for="s">Source</label>
<select id="s"><option>phone</option><option>drone</option><option>logs</option><option>telemetry</option></select>
<label for="f">Files</label><input id="f" type="file" multiple accept="video/*,image/*,.srt,.SRT,.txt,.csv,.jsonl">
<button id="go">Upload</button>
<div id="list"></div>
<script>
const $=id=>document.getElementById(id);
function fmt(b){return b>1e9?(b/1e9).toFixed(2)+' GB':(b/1e6).toFixed(1)+' MB'}
function up(file,date,src,row){return new Promise(res=>{
  const x=new XMLHttpRequest(),bar=row.querySelector('progress'),st=row.querySelector('.s');
  x.open('PUT','/upload?date='+encodeURIComponent(date)+'&source='+src+'&name='+encodeURIComponent(file.name));
  x.upload.onprogress=e=>{if(e.lengthComputable){bar.value=e.loaded/e.total;st.textContent=fmt(e.loaded)+' / '+fmt(e.total)}};
  x.onload=()=>{let m=x.responseText;try{m=JSON.parse(m).path||JSON.parse(m).error}catch(e){}
    st.textContent=(x.status==200?'Done: ':'Failed: ')+m;st.className='s '+(x.status==200?'ok':'bad');res()};
  x.onerror=()=>{st.textContent='Failed: network error';st.className='s bad';res()};
  x.send(file)})}
$('go').onclick=async()=>{
  const files=[...$('f').files];if(!files.length)return;
  $('go').disabled=true;$('list').innerHTML='';
  const rows=files.map(f=>{const r=document.createElement('div');r.className='f';
    r.innerHTML='<div class="n"></div><progress value="0"></progress><div class="s">waiting</div>';
    r.querySelector('.n').textContent=f.name;$('list').append(r);return r});
  for(let i=0;i<files.length;i++) await up(files[i],$('d').value,$('s').value,rows[i]);
  $('go').disabled=false};
</script></body></html>"""


def free_bytes() -> int:
    return shutil.disk_usage(ROOT).free


class Handler(BaseHTTPRequestHandler):
    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if urlparse(self.path).path != "/":
            return self._json(404, {"error": "not found"})
        page = (PAGE.replace("__TODAY__", dt.date.today().isoformat())
                    .replace("__FREE__", html.escape(f"{free_bytes() / 1e9:.0f} GB")))
        body = page.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_PUT(self):
        u = urlparse(self.path)
        if u.path != "/upload":
            return self._json(404, {"error": "not found"})
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        date, source = q.get("date", ""), q.get("source", "phone")
        name = SAFE_NAME.sub("_", os.path.basename(q.get("name", "")))
        if not DATE_RE.match(date) or source not in SOURCES or not name.strip("._"):
            return self._json(400, {"error": "bad date, source or name"})
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            return self._json(411, {"error": "Content-Length required"})
        if free_bytes() - length < MIN_FREE:
            return self._json(507, {"error": f"not enough disk space ({free_bytes() / 1e9:.0f} GB free)"})

        dest_dir = ROOT / date / source
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / name
        if dest.exists():  # never overwrite raw data; keep both
            stem, suf, i = dest.stem, dest.suffix, 1
            while (dest_dir / f"{stem}_{i}{suf}").exists():
                i += 1
            dest = dest_dir / f"{stem}_{i}{suf}"
        part = dest.with_name(dest.name + ".part")

        remaining = length
        try:
            with open(part, "wb") as f:
                while remaining > 0:
                    chunk = self.rfile.read(min(CHUNK, remaining))
                    if not chunk:
                        raise ConnectionError("client disconnected")
                    f.write(chunk)
                    remaining -= len(chunk)
            part.rename(dest)
        except Exception as e:  # leave nothing half-written behind
            part.unlink(missing_ok=True)
            return self._json(500, {"error": str(e)})
        self.log_message("saved %s (%d bytes)", dest, length)
        return self._json(200, {"path": str(dest.relative_to(ROOT.parent.parent)), "bytes": length})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="100.80.98.4")
    ap.add_argument("--port", type=int, default=8966)
    a = ap.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    print(f"serving http://{a.host}:{a.port} -> {ROOT}", flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
