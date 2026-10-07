#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from lerobot.configs import PreTrainedConfig
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.storage import load_dataset_metadata
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.processor.rename_processor import rename_batch_keys, rename_stats
from lerobot.utils.constants import OBS_PREFIX
from lerobot.utils.constants import ACTION


@dataclass
class EvalAggregate:
    sum_sq: float = 0.0
    sum_abs: float = 0.0
    count: int = 0
    max_abs: float = 0.0

    def update(self, diff: torch.Tensor) -> None:
        abs_diff = diff.abs()
        self.sum_sq += float((diff * diff).sum().item())
        self.sum_abs += float(abs_diff.sum().item())
        self.count += int(diff.numel())
        local_max = float(abs_diff.max().item()) if diff.numel() > 0 else 0.0
        self.max_abs = max(self.max_abs, local_max)

    def metrics(self) -> dict[str, float]:
        if self.count == 0:
            return {"mse": 0.0, "rmse": 0.0, "mae": 0.0, "max_abs_err": self.max_abs}
        mse = self.sum_sq / self.count
        return {
            "mse": mse,
            "rmse": mse**0.5,
            "mae": self.sum_abs / self.count,
            "max_abs_err": self.max_abs,
        }


@dataclass
class JointTraceCollector:
    action_dim: int
    max_points: int
    time_s: list[float]
    episode_index: list[int]
    pred: list[list[float]]
    target: list[list[float]]
    cursor: int = 0

    @classmethod
    def create(cls, action_dim: int, max_points: int) -> "JointTraceCollector":
        return cls(
            action_dim=action_dim,
            max_points=max_points,
            time_s=[],
            episode_index=[],
            pred=[[] for _ in range(action_dim)],
            target=[[] for _ in range(action_dim)],
        )

    def update(
        self,
        pred_valid: torch.Tensor,
        target_valid: torch.Tensor,
        time_valid_s: torch.Tensor,
        episode_valid: torch.Tensor,
    ) -> None:
        if self.cursor >= self.max_points:
            return
        remaining = self.max_points - self.cursor
        n_take = min(remaining, pred_valid.shape[0])
        if n_take <= 0:
            return

        pred_np = pred_valid[:n_take].detach().cpu().to(torch.float32)
        target_np = target_valid[:n_take].detach().cpu().to(torch.float32)
        time_np = time_valid_s[:n_take].detach().cpu().to(torch.float64)
        ep_np = episode_valid[:n_take].detach().cpu().to(torch.int64)
        self.time_s.extend(time_np.tolist())
        self.episode_index.extend(ep_np.tolist())
        for j in range(self.action_dim):
            self.pred[j].extend(pred_np[:, j].tolist())
            self.target[j].extend(target_np[:, j].tolist())
        self.cursor += n_take


@dataclass
class TimingAggregate:
    """Accumulates wall-clock latency of each measured policy forward pass."""

    latencies_ms: list[float]

    @classmethod
    def create(cls) -> "TimingAggregate":
        return cls(latencies_ms=[])

    def update(self, latency_s: float) -> None:
        self.latencies_ms.append(latency_s * 1000.0)

    def metrics(
        self,
        *,
        chunk_size: int,
        batch_size: int,
        dataset_fps: int,
    ) -> dict[str, float | int]:
        n = len(self.latencies_ms)
        if n == 0:
            return {"measured_forward_passes": 0}
        arr = sorted(self.latencies_ms)

        def _pct(p: float) -> float:
            k = max(0, min(n - 1, int(round((n - 1) * p))))
            return arr[k]

        mean_ms = sum(self.latencies_ms) / n
        chunk_span_s = chunk_size / float(dataset_fps) if dataset_fps > 0 else 0.0
        real_time_factor = (chunk_span_s / (mean_ms / 1000.0)) if mean_ms > 0 else float("inf")
        return {
            "measured_forward_passes": n,
            "batch_size": batch_size,
            "chunk_size": chunk_size,
            "dataset_fps": dataset_fps,
            "mean_ms_per_forward": mean_ms,
            "p50_ms_per_forward": _pct(0.50),
            "p95_ms_per_forward": _pct(0.95),
            "min_ms_per_forward": arr[0],
            "max_ms_per_forward": arr[-1],
            "mean_ms_per_sample_batched": mean_ms / max(batch_size, 1),
            "chunk_horizon_covers_s": chunk_span_s,
            "chunk_real_time_factor": real_time_factor,
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open-loop action evaluation for LeRobot policies.")
    parser.add_argument("--dataset.repo_id", dest="dataset_repo_id", required=True)
    parser.add_argument("--dataset.root", dest="dataset_root", default=None)
    parser.add_argument("--dataset.revision", dest="dataset_revision", default=None)
    parser.add_argument("--dataset.video_backend", dest="dataset_video_backend", default="torchcodec")
    parser.add_argument("--episode", type=int, default=None, help="Evaluate a single episode id only.")
    parser.add_argument("--policy.path", dest="policy_paths", action="append", required=True)
    parser.add_argument("--policy.name", dest="policy_names", action="append", default=None)
    parser.add_argument(
        "--policy.type",
        dest="policy_types",
        action="append",
        default=None,
        help="Optional label per model: smolvla, pi05, or custom.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=0, help="0 means all batches.")
    parser.add_argument(
        "--warmup_batches",
        type=int,
        default=2,
        help="Batches to skip before recording forward-pass latency (avoids CUDA JIT / cudnn autotune spikes).",
    )
    parser.add_argument("--max_plot_points", type=int, default=2000)
    parser.add_argument(
        "--trace_horizon",
        type=int,
        default=0,
        help="Which action-chunk horizon index to use for plots/rerun traces (default: 0).",
    )
    parser.add_argument("--rename_map", default="{}", help='JSON string, e.g. {"a":"b"}')
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--save_plots_dir", default=None)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--rerun_session_name", default="open_loop_eval")
    parser.add_argument(
        "--rerun_spawn_viewer",
        action="store_true",
        help="Spawn rerun desktop viewer from this process (requires DISPLAY/WAYLAND).",
    )
    parser.add_argument(
        "--rerun_output_rrd",
        default=None,
        help="Optional path to save rerun recording (.rrd) for later viewing.",
    )
    return parser.parse_args()


def _preprocess_dataset_batch(
    batch: dict[str, Any],
    camera_keys: list[str],
    rename_map: dict[str, str],
    preprocessor: Any,
) -> Any:
    model_batch: dict[str, Any] = {}
    for key, value in batch.items():
        if key.startswith(OBS_PREFIX) or key in {"task", "task_index", "timestamp", "frame_index", "episode_index", "index"}:
            model_batch[key] = value

    for cam_key in camera_keys:
        if cam_key in model_batch and model_batch[cam_key].dtype == torch.uint8:
            model_batch[cam_key] = model_batch[cam_key].to(dtype=torch.float32) / 255.0
    model_batch = rename_batch_keys(model_batch, rename_map)
    return preprocessor(model_batch)


def _mask_and_align(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if target.device != pred.device:
        target = target.to(pred.device)

    horizon = min(pred.shape[1], target.shape[1])
    dim = min(pred.shape[2], target.shape[2])
    pred = pred[:, :horizon, :dim]
    target = target[:, :horizon, :dim]

    if mask is None:
        valid = torch.ones((pred.shape[0], pred.shape[1]), dtype=torch.bool, device=pred.device)
    else:
        valid = mask[:, :horizon].to(device=pred.device, dtype=torch.bool)
    return pred[valid], target[valid], valid


def _save_joint_plots(
    *,
    output_dir: Path,
    policy_name: str,
    joint_names: list[str],
    traces: JointTraceCollector,
) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required when --save_plots_dir is set. Install it in the LeRobot venv."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: list[str] = []
    if not traces.episode_index:
        return plot_paths

    episode_to_indices: dict[int, list[int]] = {}
    for idx, ep_idx in enumerate(traces.episode_index):
        episode_to_indices.setdefault(int(ep_idx), []).append(idx)

    sorted_episodes = sorted(episode_to_indices.keys())
    for joint_idx, joint_name in enumerate(joint_names):
        all_pred = traces.pred[joint_idx]
        all_target = traces.target[joint_idx]
        if not all_pred or not all_target:
            continue

        safe_joint = joint_name.replace("/", "_")
        for ep_idx in sorted_episodes:
            sel = episode_to_indices[ep_idx]
            if not sel:
                continue
            pred = [all_pred[i] for i in sel if i < len(all_pred)]
            target = [all_target[i] for i in sel if i < len(all_target)]
            x = [traces.time_s[i] for i in sel if i < len(traces.time_s)]
            n = min(len(pred), len(target), len(x))
            if n == 0:
                continue

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(x[:n], target[:n], label="target", linewidth=1.0)
            ax.plot(x[:n], pred[:n], label="pred", linewidth=1.0)
            ax.set_title(f"{policy_name} - episode {ep_idx:06d} - {joint_name}")
            ax.set_xlabel("time (s)")
            ax.set_ylabel("value")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")
            fig.tight_layout()

            ep_dir = output_dir / f"episode_{ep_idx:06d}"
            ep_dir.mkdir(parents=True, exist_ok=True)
            plot_path = ep_dir / f"{policy_name}_joint_{joint_idx:02d}_{safe_joint}.png"
            fig.savefig(plot_path, dpi=120)
            plt.close(fig)
            plot_paths.append(str(plot_path))
    return plot_paths


def _log_rerun_joint_series(
    *,
    session_name: str,
    policy_name: str,
    joint_names: list[str],
    traces: JointTraceCollector,
    spawn_viewer: bool,
    output_rrd: str | None,
) -> None:
    try:
        import rerun as rr
    except ImportError as exc:
        raise RuntimeError("rerun is required when --rerun is set. Install rerun-sdk in the LeRobot venv.") from exc

    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") or os.environ.get("WAYLAND_SOCKET"))
    effective_spawn = spawn_viewer and has_display
    if spawn_viewer and not has_display:
        print("Rerun viewer spawn disabled: no DISPLAY/WAYLAND found. Saving/logging without GUI spawn.")

    rr.init(f"{session_name}_{policy_name}", spawn=effective_spawn)
    if output_rrd:
        rrd_path = str(Path(output_rrd).expanduser().resolve())
        Path(rrd_path).parent.mkdir(parents=True, exist_ok=True)
        if not hasattr(rr, "save"):
            raise RuntimeError("Installed rerun-sdk does not support rr.save(). Please upgrade rerun-sdk.")
        rr.save(rrd_path)

    def _set_time(step: int, time_s: float | None) -> None:
        if hasattr(rr, "set_time_sequence"):
            rr.set_time_sequence("valid_action_index", step)
            if hasattr(rr, "set_time") and time_s is not None:
                rr.set_time("time_s", duration=float(time_s))
            return
        if hasattr(rr, "set_time"):
            rr.set_time("valid_action_index", sequence=step)
            if time_s is not None:
                rr.set_time("time_s", duration=float(time_s))
            return
        raise RuntimeError(
            "Installed rerun-sdk does not expose set_time_sequence or set_time. Please upgrade rerun-sdk."
        )

    def _scalar_component(value: float) -> Any:
        # rerun-sdk API changed across versions: some expose Scalar, older ones expose Scalars.
        if hasattr(rr, "Scalar"):
            return rr.Scalar(value)
        if hasattr(rr, "Scalars"):
            return rr.Scalars([value])
        raise RuntimeError("Installed rerun-sdk does not expose Scalar or Scalars archetypes.")

    rr.log("meta/policy_name", rr.TextDocument(policy_name))
    for joint_idx, joint_name in enumerate(joint_names):
        joint_path = joint_name.replace("/", "_")
        pred = traces.pred[joint_idx]
        target = traces.target[joint_idx]
        if not pred or not target:
            continue
        n = min(len(pred), len(target))
        for step_idx in range(n):
            t = traces.time_s[step_idx] if step_idx < len(traces.time_s) else None
            _set_time(step_idx, t)
            ep_idx = traces.episode_index[step_idx] if step_idx < len(traces.episode_index) else -1
            rr.log(
                f"episodes/episode_{ep_idx:06d}/joints/{joint_path}/target",
                _scalar_component(target[step_idx]),
            )
            rr.log(
                f"episodes/episode_{ep_idx:06d}/joints/{joint_path}/pred",
                _scalar_component(pred[step_idx]),
            )


def _decode_action_scale(postprocessor: Any, action_chunk: torch.Tensor) -> torch.Tensor:
    decoded = postprocessor(action_chunk)
    if not isinstance(decoded, torch.Tensor):
        raise RuntimeError("Policy postprocessor must return a torch.Tensor for action chunks.")
    return decoded


def _compute_valid_times_s(
    *,
    raw_batch: dict[str, Any],
    valid_mask_2d: torch.Tensor,
    horizon: int,
    fps: int,
) -> torch.Tensor:
    if "timestamp" in raw_batch:
        base = raw_batch["timestamp"]
        if not isinstance(base, torch.Tensor):
            base = torch.as_tensor(base, dtype=torch.float64)
        else:
            base = base.to(dtype=torch.float64)
        if base.ndim > 1:
            base = base.reshape(base.shape[0], -1)[:, 0]
    else:
        base = torch.arange(valid_mask_2d.shape[0], dtype=torch.float64)

    offsets = torch.arange(horizon, dtype=torch.float64) / float(fps)
    time_grid = base[:, None] + offsets[None, :]
    return time_grid[valid_mask_2d.detach().cpu()]


def _collect_trace_horizon(
    *,
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask_2d: torch.Tensor,
    raw_batch: dict[str, Any],
    fps: int,
    trace_horizon: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    horizon = min(pred.shape[1], target.shape[1])
    if horizon <= 0:
        return (
            torch.empty((0, pred.shape[-1]), dtype=pred.dtype, device=pred.device),
            torch.empty((0, target.shape[-1]), dtype=target.dtype, device=target.device),
            torch.empty((0,), dtype=torch.float64),
            torch.empty((0,), dtype=torch.int64),
        )

    h = max(0, min(trace_horizon, horizon - 1))
    pred_h = pred[:, h, :]
    target_h = target[:, h, :]
    valid_h = valid_mask_2d[:, h]

    if "timestamp" in raw_batch:
        base = raw_batch["timestamp"]
        if not isinstance(base, torch.Tensor):
            base = torch.as_tensor(base, dtype=torch.float64)
        else:
            base = base.to(dtype=torch.float64)
        if base.ndim > 1:
            base = base.reshape(base.shape[0], -1)[:, 0]
    else:
        base = torch.arange(valid_mask_2d.shape[0], dtype=torch.float64)

    t_h = (base + (float(h) / float(fps)))[valid_h.detach().cpu()]
    if "episode_index" in raw_batch:
        ep = raw_batch["episode_index"]
        if not isinstance(ep, torch.Tensor):
            ep = torch.as_tensor(ep, dtype=torch.int64)
        else:
            ep = ep.to(dtype=torch.int64)
        if ep.ndim > 1:
            ep = ep.reshape(ep.shape[0], -1)[:, 0]
    else:
        ep = torch.full((valid_mask_2d.shape[0],), -1, dtype=torch.int64)

    ep_h = ep[valid_h.detach().cpu()]
    return pred_h[valid_h], target_h[valid_h], t_h, ep_h


def evaluate_policy(
    *,
    policy_name: str,
    policy_path: str,
    dataset_repo_id: str,
    dataset_root: str | None,
    dataset_revision: str | None,
    dataset_video_backend: str,
    episode: int | None,
    device: str,
    batch_size: int,
    num_workers: int,
    max_batches: int,
    warmup_batches: int,
    max_plot_points: int,
    trace_horizon: int,
    rename_map: dict[str, str],
    save_plots_dir: str | None,
    rerun_enable: bool,
    rerun_session_name: str,
    rerun_spawn_viewer: bool,
    rerun_output_rrd: str | None,
) -> dict[str, Any]:
    model_path = Path(policy_path).expanduser().resolve()
    policy_cfg = PreTrainedConfig.from_pretrained(model_path)
    policy_cfg.device = device
    policy_cfg.pretrained_path = str(model_path)

    ds_meta = load_dataset_metadata(
        dataset_repo_id,
        root=dataset_root,
        revision=dataset_revision,
        repo_type="dataset",
    )
    delta_timestamps = resolve_delta_timestamps(policy_cfg, ds_meta, rename_map)
    dataset = LeRobotDataset(
        dataset_repo_id,
        root=dataset_root,
        episodes=None if episode is None else [episode],
        delta_timestamps=delta_timestamps,
        image_transforms=None,
        revision=dataset_revision,
        video_backend=dataset_video_backend,
        return_uint8=True,
        depth_output_unit="mm",
        tolerance_s=1e-4,
        repo_type="dataset",
    )

    policy = make_policy(policy_cfg, ds_meta=dataset.meta, rename_map=rename_map)
    policy.eval()

    processor_stats = rename_stats(dataset.meta.stats, rename_map)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(model_path),
        pretrained_revision=getattr(policy_cfg, "pretrained_revision", None),
        dataset_stats=processor_stats,
        dataset_meta=dataset.meta,
        preprocessor_overrides={
            "device_processor": {"device": device},
            "normalizer_processor": {
                "features": policy.config.input_features,
                "norm_map": policy.config.normalization_mapping,
                "stats": processor_stats,
            },
            "rename_observations_processor": {"rename_map": rename_map},
        },
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )

    agg = EvalAggregate()
    per_horizon: list[EvalAggregate] | None = None
    per_joint: list[EvalAggregate] | None = None
    traces: JointTraceCollector | None = None
    timing = TimingAggregate.create()
    total_batches = len(dataloader)
    batch_limit = total_batches if max_batches <= 0 else min(max_batches, total_batches)
    cuda_active = device.startswith("cuda") and torch.cuda.is_available()

    with torch.no_grad():
        for batch_idx, raw_batch in enumerate(dataloader):
            if batch_idx >= batch_limit:
                break

            batch = _preprocess_dataset_batch(raw_batch, dataset.meta.camera_keys, rename_map, preprocessor)

            if cuda_active:
                torch.cuda.synchronize()
            t_forward_start = time.perf_counter()
            raw_pred = policy.predict_action_chunk(batch)
            if cuda_active:
                torch.cuda.synchronize()
            forward_elapsed_s = time.perf_counter() - t_forward_start
            if batch_idx >= warmup_batches:
                timing.update(forward_elapsed_s)

            pred = _decode_action_scale(postprocessor, raw_pred)
            target = raw_batch[ACTION]
            action_is_pad = raw_batch.get("action_is_pad")
            valid_mask = None if action_is_pad is None else (~action_is_pad)

            flat_pred, flat_target, valid_mask_2d = _mask_and_align(pred, target, valid_mask)
            diff = flat_pred - flat_target
            agg.update(diff)
            if per_joint is None:
                per_joint = [EvalAggregate() for _ in range(flat_pred.shape[-1])]
                traces = JointTraceCollector.create(action_dim=flat_pred.shape[-1], max_points=max_plot_points)
            for j in range(flat_pred.shape[-1]):
                per_joint[j].update(diff[:, j])
            if traces is not None:
                pred_h, target_h, time_h_s, ep_h = _collect_trace_horizon(
                    pred=pred,
                    target=target,
                    valid_mask_2d=valid_mask_2d,
                    raw_batch=raw_batch,
                    fps=dataset.meta.fps,
                    trace_horizon=trace_horizon,
                )
                traces.update(pred_h, target_h, time_h_s, ep_h)

            horizon = min(pred.shape[1], target.shape[1])
            if per_horizon is None:
                per_horizon = [EvalAggregate() for _ in range(horizon)]
            for h in range(min(horizon, len(per_horizon))):
                if valid_mask is None:
                    horizon_mask = torch.ones((pred.shape[0],), dtype=torch.bool, device=pred.device)
                else:
                    horizon_mask = valid_mask_2d[:, h]
                if horizon_mask.any():
                    d = pred[horizon_mask, h, : flat_pred.shape[-1]] - target[horizon_mask, h, : flat_pred.shape[-1]]
                    per_horizon[h].update(d)

    action_feature = dataset.meta.features.get(ACTION, {})
    joint_names = list(action_feature.get("names", []))
    if not joint_names or len(joint_names) != (len(per_joint) if per_joint is not None else 0):
        n_joints = len(per_joint) if per_joint is not None else 0
        joint_names = [f"joint_{i}" for i in range(n_joints)]

    plot_files: list[str] = []
    if save_plots_dir and traces is not None:
        plot_files = _save_joint_plots(
            output_dir=Path(save_plots_dir).expanduser().resolve(),
            policy_name=policy_name,
            joint_names=joint_names,
            traces=traces,
        )
    if rerun_enable and traces is not None:
        _log_rerun_joint_series(
            session_name=rerun_session_name,
            policy_name=policy_name,
            joint_names=joint_names,
            traces=traces,
            spawn_viewer=rerun_spawn_viewer,
            output_rrd=rerun_output_rrd,
        )

    chunk_size = len(per_horizon) if per_horizon is not None else 0
    result = {
        "policy_name": policy_name,
        "policy_path": str(model_path),
        "dataset_repo_id": dataset_repo_id,
        "dataset_root": dataset_root,
        "episode": episode,
        "device": device,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "evaluated_batches": batch_limit,
        "evaluated_action_values": agg.count,
        "warmup_batches": warmup_batches,
        "overall": agg.metrics(),
        "per_joint": [
            {"joint": joint_names[i], **joint_agg.metrics()} for i, joint_agg in enumerate(per_joint or [])
        ],
        "per_horizon": [h.metrics() for h in (per_horizon or [])],
        "timing": timing.metrics(
            chunk_size=chunk_size,
            batch_size=batch_size,
            dataset_fps=int(dataset.meta.fps),
        ),
        "plot_files": plot_files,
        "rerun_output_rrd": rerun_output_rrd,
        "trace_horizon": trace_horizon,
    }
    return result


def main() -> None:
    args = _parse_args()
    rename_map = json.loads(args.rename_map)
    if args.policy_names and len(args.policy_names) != len(args.policy_paths):
        raise ValueError("--policy.name must be provided once per --policy.path.")
    if args.policy_types and len(args.policy_types) != len(args.policy_paths):
        raise ValueError("--policy.type must be provided once per --policy.path.")

    policy_names = args.policy_names or [f"model_{i+1}" for i in range(len(args.policy_paths))]
    policy_types = args.policy_types or ["custom" for _ in range(len(args.policy_paths))]
    all_results = []
    for idx, policy_path in enumerate(args.policy_paths):
        policy_type = policy_types[idx].lower()
        policy_rename_map = dict(rename_map)
        if policy_type == "smolvla":
            policy_rename_map = {
                "observation.images.top": "observation.images.camera1",
                "observation.images.side": "observation.images.camera2",
                **policy_rename_map,
            }
        res = evaluate_policy(
            policy_name=policy_names[idx],
            policy_path=policy_path,
            dataset_repo_id=args.dataset_repo_id,
            dataset_root=args.dataset_root,
            dataset_revision=args.dataset_revision,
            dataset_video_backend=args.dataset_video_backend,
            episode=args.episode,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_batches=args.max_batches,
            warmup_batches=args.warmup_batches,
            max_plot_points=args.max_plot_points,
            trace_horizon=args.trace_horizon,
            rename_map=policy_rename_map,
            save_plots_dir=args.save_plots_dir,
            rerun_enable=args.rerun,
            rerun_session_name=args.rerun_session_name,
            rerun_spawn_viewer=args.rerun_spawn_viewer,
            rerun_output_rrd=args.rerun_output_rrd,
        )
        all_results.append(res)
        print(json.dumps(res, indent=2))

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({"results": all_results}, indent=2), encoding="utf-8")
        print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()
