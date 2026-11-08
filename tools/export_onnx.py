"""Export the on-phone pilot stack (TAESD encoder + hazard net + policy actor) as one ONNX graph,
check it against PyTorch, and time it on CPU.

    .venv/bin/python tools/export_onnx.py --ckpt runs/route-wlref-2/policy.pt --hazard runs/hazard-wlref/hazard.pt \
        --out runs/onnx/pilot.onnx

Graph (batch 1, float32):
  inputs   rgb        [1,96,128,3]  0..1, RGB, the camera frame centre-cropped to 4:3 and resized to 128x96
           prev_lat   [1,4,12,16]   the previous frame's `lat` output (first frame: run once and feed lat back)
           vec        [1,13]        proprio/goal vector, built exactly as PilotCore.frame (src/djirl/pilot.py)
  outputs  lat        [1,4,12,16]   this frame's latent (feed back as prev_lat next frame)
           clear      [1,2]         predicted free metres along velocity, along camera forward
           p_brake    [1]           sigmoid(must_brake logit)
           action     [1,4]         policy mean, normalised sticks [fwd, right, up, yaw_right] in -1..1
The trigger / handback state machine and the stick scaling stay in app code (same rules as pilot.py).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


class PilotGraph(torch.nn.Module):
    def __init__(self, enc, haz, pol):
        super().__init__()
        self.enc, self.haz, self.pol = enc, haz, pol

    def forward(self, rgb, prev_lat, vec):
        lat = self.enc(rgb.permute(0, 3, 1, 2) * 2 - 1)
        stack = torch.stack([prev_lat, lat], 1)
        clear, logit = self.haz(stack, vec)
        mu = self.pol.actor(torch.cat([self.pol.a_trunk(stack), vec], -1))
        return lat, clear, torch.sigmoid(logit[:, -1]), mu.clamp(-1, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="runs/route-wlref-2/policy.pt")
    ap.add_argument("--hazard", default="runs/hazard-wlref/hazard.pt")
    ap.add_argument("--out", default="runs/onnx/pilot.onnx")
    a = ap.parse_args()
    from djirl.encoder import LatentEncoder
    from djirl.pilot import PilotCore
    core = PilotCore(a.ckpt, a.hazard, device="cpu")
    enc = LatentEncoder("cpu", dtype=torch.float32).enc
    g = PilotGraph(enc, core.haz, core.pol).eval()
    rgb, prev, vec = torch.rand(1, 96, 128, 3), torch.randn(1, 4, 12, 16), torch.randn(1, 13) * 0.3
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(g, (rgb, prev, vec), out, input_names=["rgb", "prev_lat", "vec"],
                      output_names=["lat", "clear", "p_brake", "action"], opset_version=17)
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 4
    s = ort.InferenceSession(str(out), so, providers=["CPUExecutionProvider"])
    feed = {"rgb": rgb.numpy(), "prev_lat": prev.numpy(), "vec": vec.numpy()}
    ref = [t.detach().numpy() for t in g(rgb, prev, vec)]
    got = s.run(None, feed)
    err = {n: float(np.abs(r - o).max()) for n, r, o in zip(["lat", "clear", "p_brake", "action"], ref, got)}
    for _ in range(5):
        s.run(None, feed)
    t0 = time.time()
    for _ in range(50):
        s.run(None, feed)
    ms = (time.time() - t0) / 50 * 1000
    params = sum(p.numel() for p in g.parameters())
    info = {"onnx": str(out), "bytes": out.stat().st_size, "params": params, "max_abs_err": err,
            "cpu_ms_4threads": round(ms, 2), "ckpt": a.ckpt, "hazard": a.hazard}
    out.with_suffix(".json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
