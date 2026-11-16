"""pi0.5 train configs for the overnight run (LIBERO + OCBench), kept outside lair-openpi.

    python ovn_pi05.py train  <name> [--resume]      # name: pi05_l90 | pi05_ocbs
    python ovn_pi05.py serve  <name> <ckpt_dir> [port]  # websocket policy server (LIBERO eval)
"""

from __future__ import annotations

import dataclasses
import os
import sys

import numpy as np

sys.path.insert(0, "/scratch/aap9348/lair-openpi/scripts")
sys.path.insert(0, "/scratch/aap9348/lair-openpi/src")

import openpi.models.model as _model  # noqa: E402
import openpi.models.pi0_config as pi0_config  # noqa: E402
import openpi.training.config as C  # noqa: E402
import openpi.training.optimizer as _optimizer  # noqa: E402
import openpi.transforms as T  # noqa: E402
import functools as _ft  # noqa: E402
import websockets.asyncio.server as _wss  # noqa: E402

_wss.serve = _ft.partial(_wss.serve, ping_interval=None, ping_timeout=None)  # no keepalive drop

RUNS = "/scratch/aap9348/rlft-bench/runs/overnight"
STEPS = int(os.environ.get("OVN_STEPS", 15000))


@dataclasses.dataclass(frozen=True)
class OcbInputs(T.DataTransformFn):
    """OCBench stacking: front -> base view, wrist -> left wrist; right wrist padded + masked (as LIBERO does)."""

    def __call__(self, data: dict) -> dict:
        def img(x):
            x = np.asarray(x)
            if np.issubdtype(x.dtype, np.floating):
                x = (255 * x).astype(np.uint8)
            return np.moveaxis(x, 0, -1) if x.shape[0] == 3 else x

        base = img(data["observation/front"])
        out = {
            "state": np.asarray(data["observation/state"], np.float32),
            "image": {"base_0_rgb": base, "left_wrist_0_rgb": img(data["observation/wrist"]),
                      "right_wrist_0_rgb": np.zeros_like(base)},
            "image_mask": {"base_0_rgb": np.True_, "left_wrist_0_rgb": np.True_, "right_wrist_0_rgb": np.False_},
        }
        if "actions" in data:
            out["actions"] = np.asarray(data["actions"], np.float32)
        if "prompt" in data:
            out["prompt"] = data["prompt"]
        return out


@dataclasses.dataclass(frozen=True)
class OcbOutputs(T.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :7])}


@dataclasses.dataclass(frozen=True)
class OcbDataConfig(C.DataConfigFactory):
    def create(self, assets_dirs, model_config: _model.BaseModelConfig) -> C.DataConfig:
        repack = T.Group(inputs=[T.RepackTransform({
            "observation/front": "observation.images.front", "observation/wrist": "observation.images.wrist",
            "observation/state": "observation.state", "actions": "action", "prompt": "prompt"})])
        # arm targets as deltas from the chunk-start joints (state[:6]); gripper stays absolute (same as ACT)
        mask = T.make_bool_mask(6, -1)
        data_t = T.Group(inputs=[OcbInputs(), T.DeltaActions(mask)], outputs=[T.AbsoluteActions(mask), OcbOutputs()])
        model_t = C.ModelTransformFactory()(model_config)
        return dataclasses.replace(self.create_base_config(assets_dirs, model_config), repack_transforms=repack,
                                   data_transforms=data_t, model_transforms=model_t, action_sequence_keys=("action",))


@dataclasses.dataclass(frozen=True)
class L90DataConfig(C.LeRobotLiberoDataConfig):
    """openpi's LIBERO transforms, with key names from our converted LIBERO-90 dataset."""

    def create(self, assets_dirs, model_config: _model.BaseModelConfig) -> C.DataConfig:
        cfg = super().create(assets_dirs, model_config)
        repack = T.Group(inputs=[T.RepackTransform({
            "observation/image": "observation.images.image", "observation/wrist_image": "observation.images.wrist",
            "observation/state": "observation.state", "actions": "action", "prompt": "prompt"})])
        return dataclasses.replace(cfg, repack_transforms=repack, action_sequence_keys=("action",))


def _shared(cfg, name):
    """Shared overnight recipe (same as act_train.py): batch 32, STEPS steps, 500 warmup, cosine 5e-5 -> 5e-6,
    grad clip 1.0, no EMA, action horizon 10."""
    return dataclasses.replace(
        cfg, name=name, exp_name="run", batch_size=32, num_train_steps=STEPS,
        fsdp_devices=int(os.environ.get("OVN_FSDP", 2)), ema_decay=None,
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=500, peak_lr=5e-5, decay_steps=STEPS, decay_lr=5e-6),
        save_interval=1000, keep_period=5000, log_interval=50, wandb_enabled=False,
        checkpoint_base_dir=f"{RUNS}/checkpoints", assets_base_dir=f"{RUNS}/assets")


def get(name: str) -> C.TrainConfig:
    base = C.get_config("pi05_libero")  # pi05_base weights, pi0.5 model, LIBERO-style transforms
    if name == "pi05_l90":
        return _shared(dataclasses.replace(base, data=L90DataConfig(
            repo_id="local/libero90_stack", base_config=C.DataConfig(prompt_from_task=True),
            extra_delta_transform=False)), name)
    if name == "pi05_ocbs":
        return _shared(dataclasses.replace(base, data=OcbDataConfig(
            repo_id="local/ocb_stack", base_config=C.DataConfig(prompt_from_task=True))), name)
    raise KeyError(name)


if __name__ == "__main__":
    mode, name = sys.argv[1], sys.argv[2]
    cfg = get(name)
    if mode == "train":
        import train as _train

        resume = "--resume" in sys.argv
        _train.main(dataclasses.replace(cfg, resume=resume, overwrite=not resume))
    elif mode == "serve":
        from openpi.policies import policy_config
        from openpi.serving import websocket_policy_server

        policy = policy_config.create_trained_policy(cfg, sys.argv[3])
        _infer = policy.infer
        policy.infer = lambda obs: _infer({k: v for k, v in obs.items() if not k.endswith("_raw")})
        port = int(sys.argv[4]) if len(sys.argv) > 4 else 8000
        websocket_policy_server.WebsocketPolicyServer(policy, host="0.0.0.0", port=port).serve_forever()
