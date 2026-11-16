"""Serve a trained ACT run over openpi's websocket protocol (same client code as pi0.5).

    python act_serve.py RUN_DIR [port]

Observation keys sent by the eval clients:
  OCBench:   observation/front, observation/wrist (128x128 uint8), observation/state (19), prompt
Returns {"actions": (chunk, action_dim)} un-normalized.
"""

import os

import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, os.environ["LAIR_OPENPI"] + "/src")
from lerobot.common.policies.act.modeling_act import ACTPolicy  # noqa: E402
from openpi.serving import websocket_policy_server  # noqa: E402
import functools as _ft  # noqa: E402
import websockets.asyncio.server as _wss  # noqa: E402

_wss.serve = _ft.partial(_wss.serve, ping_interval=None, ping_timeout=None)  # no keepalive drop

CLIENT_KEYS = {
    "ocbs": {"observation.images.front": "observation/front", "observation.images.wrist": "observation/wrist"},
}


class ActServer:
    def __init__(self, run_dir: str):
        run = pathlib.Path(run_dir)
        self.meta = json.load(open(run / "meta.json"))
        self.policy = ACTPolicy.from_pretrained(run / "policy").cuda().eval()
        self.keys = CLIENT_KEYS[self.meta["bench"]]
        self.tasks = {t: i for i, t in enumerate(self.meta["tasks"])}
        self.metadata = {"policy": "act", "run": str(run)}

    @torch.no_grad()
    def infer(self, obs: dict) -> dict:
        p = self.policy
        b = {}
        for dst, src in self.keys.items():
            img = torch.from_numpy(np.ascontiguousarray(obs[src])).cuda().permute(2, 0, 1).float() / 255.0
            b[dst] = img[None]
        state = np.asarray(obs["observation/state"], np.float32)
        b["observation.state"] = torch.as_tensor(state).cuda()[None]
        if self.tasks:
            oh = torch.zeros(1, len(self.tasks), device="cuda")
            oh[0, self.tasks[str(obs["prompt"]).strip()]] = 1.0
            b["observation.environment_state"] = oh
        b = p.normalize_inputs(b)
        b["observation.images"] = [b[k] for k in p.config.image_features]
        actions = p.model(b)[0]
        actions = p.unnormalize_outputs({"action": actions})["action"]
        actions = actions[0].float().cpu().numpy()
        if self.meta.get("arm_delta"):  # trained on arm deltas from the chunk-start joints
            actions[:, :6] += state[:6]
        return {"actions": actions}

    def reset(self):
        pass


if __name__ == "__main__":
    srv = ActServer(sys.argv[1])
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    websocket_policy_server.WebsocketPolicyServer(srv, host="0.0.0.0", port=port, metadata=srv.metadata).serve_forever()
