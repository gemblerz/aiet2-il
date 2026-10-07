# Open-loop evaluation (SmolVLA + Pi0.5)

## 1) Activate LeRobot environment

```bash
cd <repo-root>
source ../lerobot/.venv/bin/activate
```

## 1.1) Optional visualization packages

```bash
pip install matplotlib rerun-sdk
```

## 2) Run SmolVLA open-loop eval

```bash
python scripts/open_loop_eval.py \
  --dataset.repo_id=dominicdx/so101_overhead \
  --dataset.root=./data \
  --policy.path=./artifacts/train/smolvla_dominicdx_so101_overhead_e8/checkpoints/008594/pretrained_model \
  --policy.name=smolvla \
  --policy.type=smolvla \
  --device=cuda \
  --batch_size=4 \
  --num_workers=0 \
  --max_batches=200 \
  --trace_horizon=0 \
  --save_plots_dir=./artifacts/eval/plots/smolvla \
  --max_plot_points=2000 \
  --output_json=./artifacts/eval/smolvla_open_loop.json
```

## 3) Run Pi0.5 open-loop eval

```bash
python scripts/open_loop_eval.py \
  --dataset.repo_id=dominicdx/so101_overhead \
  --dataset.root=./data \
  --policy.path=./artifacts/train/pi05_dominicdx_so101_overhead_e8/checkpoints/<step>/pretrained_model \
  --policy.name=pi05 \
  --policy.type=pi05 \
  --device=cuda \
  --batch_size=4 \
  --num_workers=0 \
  --max_batches=200 \
  --trace_horizon=0 \
  --save_plots_dir=./artifacts/eval/plots/pi05 \
  --max_plot_points=2000 \
  --output_json=./artifacts/eval/pi05_open_loop.json
```

## 4) Run both models in one command

```bash
python scripts/open_loop_eval.py \
  --dataset.repo_id=dominicdx/so101_overhead \
  --dataset.root=./data \
  --policy.path=./artifacts/train/smolvla_dominicdx_so101_overhead_e8/checkpoints/008594/pretrained_model \
  --policy.name=smolvla \
  --policy.type=smolvla \
  --policy.path=./artifacts/train/pi05_dominicdx_so101_overhead_e8/checkpoints/<step>/pretrained_model \
  --policy.name=pi05 \
  --policy.type=pi05 \
  --device=cuda \
  --batch_size=4 \
  --num_workers=0 \
  --max_batches=200 \
  --trace_horizon=0 \
  --save_plots_dir=./artifacts/eval/plots/both \
  --max_plot_points=2000 \
  --output_json=./artifacts/eval/open_loop_both.json
```

## 5) Optional: custom key remapping

```bash
--rename_map='{"observation.images.top":"observation.images.camera1","observation.images.side":"observation.images.camera2"}'
```

## 6) Optional: evaluate one episode only

```bash
--episode=0
```

Example:

```bash
python scripts/open_loop_eval.py \
  --dataset.repo_id=dominicdx/so101_overhead \
  --dataset.root=./data \
  --episode=0 \
  --policy.path=./artifacts/train/pi05_dominicdx_so101_overhead_e8/checkpoints/<step>/pretrained_model \
  --policy.name=pi05 \
  --policy.type=pi05 \
  --device=cuda \
  --batch_size=4 \
  --num_workers=0 \
  --save_plots_dir=./artifacts/eval/plots/pi05_ep0 \
  --output_json=./artifacts/eval/pi05_ep0_open_loop.json
```

## 7) Optional: open rerun visualizer

```bash
python scripts/open_loop_eval.py \
  --dataset.repo_id=dominicdx/so101_overhead \
  --dataset.root=./data \
  --policy.path=./artifacts/train/smolvla_dominicdx_so101_overhead_e8/checkpoints/008594/pretrained_model \
  --policy.name=smolvla \
  --policy.type=smolvla \
  --device=cuda \
  --batch_size=4 \
  --num_workers=0 \
  --max_batches=100 \
  --trace_horizon=0 \
  --rerun \
  --rerun_spawn_viewer \
  --rerun_session_name=smolvla_open_loop
```

## 8) Headless server rerun (no DISPLAY/WAYLAND)

```bash
python scripts/open_loop_eval.py \
  --dataset.repo_id=dominicdx/so101_overhead \
  --dataset.root=./data \
  --policy.path=./artifacts/train/smolvla_dominicdx_so101_overhead_e8/checkpoints/008594/pretrained_model \
  --policy.name=smolvla \
  --policy.type=smolvla \
  --device=cuda \
  --episode=0 \
  --batch_size=4 \
  --num_workers=0 \
  --max_batches=100 \
  --trace_horizon=0 \
  --rerun \
  --rerun_output_rrd=./artifacts/eval/rerun/smolvla_ep0.rrd \
  --output_json=./artifacts/eval/smolvla_ep0_open_loop.json
```

Open later on a machine with GUI:

```bash
rerun ./artifacts/eval/rerun/smolvla_ep0.rrd
```

## Output metrics

- `overall.mse`
- `overall.rmse`
- `overall.mae`
- `overall.max_abs_err`
- `per_joint[]` (per-joint mse/rmse/mae/max_abs_err)
- `per_horizon[]` (same metrics at each predicted action timestep)
- `timing.*` (wall-clock forward-pass latency — see below)
- `plot_files[]` (PNG file paths when `--save_plots_dir` is set; grouped under `episode_XXXXXX/`)

## Throughput / horizon feasibility

The `timing` block reports wall-clock latency of `policy.predict_action_chunk` (CUDA-synchronized on both sides). Skipped batches: `--warmup_batches` (default 2) to avoid the CUDA JIT / cuDNN autotune spike on the first pass.

Fields:

- `mean_ms_per_forward`, `p50_ms_per_forward`, `p95_ms_per_forward`, `min/max` — per-forward-pass latency.
- `mean_ms_per_sample_batched` — `mean_ms_per_forward / batch_size`. Optimistic single-sample latency assuming perfect batch amortization (usually only relevant if you serve many robots on one GPU).
- `chunk_size` — actions predicted per forward pass (matches the length of `per_horizon`).
- `chunk_horizon_covers_s` — `chunk_size / dataset_fps`. How much wall-clock robot time one chunk buys you.
- `chunk_real_time_factor` — `chunk_horizon_covers_s / (mean_ms_per_forward / 1000)`. **> 1 means the policy can compute a fresh chunk faster than the robot consumes the previous one**, i.e. closed-loop replanning at chunk boundaries is feasible. Under 1 means you'd fall behind and need to execute more of the chunk before replanning (which loads onto the increasing tail of `per_horizon`, degrading accuracy).

## Plot x-axis

- PNG plots use dataset `timestamp` in seconds on x-axis.
- If timestamp is unavailable, index fallback is used.
- `--trace_horizon=0` plots the first action in each chunk (recommended for episode trajectory view).
- Rerun logs are grouped by episode path: `episodes/episode_XXXXXX/joints/...`.
