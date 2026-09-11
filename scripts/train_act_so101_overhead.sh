#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LEROBOT_DIR="${LEROBOT_DIR:-$HOME/repo/lerobot}"
DATASET_REPO_ID="${DATASET_REPO_ID:-dominicdx/so101_overhead}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/artifacts/train/act_so101_overhead}"
JOB_NAME="${JOB_NAME:-act_so101_overhead}"
DEVICE="${DEVICE:-cuda}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"
POLICY_REPO_ID="${POLICY_REPO_ID:-}"

if [ -x "$LEROBOT_DIR/.venv/bin/lerobot-train" ]; then
  export PATH="$LEROBOT_DIR/.venv/bin:$PATH"
fi

if ! command -v lerobot-train >/dev/null 2>&1; then
  echo "lerobot-train command not found." >&2
  echo "Install LeRobot first: ./scripts/install_lerobot.sh" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

cmd=(
  lerobot-train
  "--dataset.repo_id=$DATASET_REPO_ID"
  "--policy.type=act"
  "--output_dir=$OUTPUT_DIR"
  "--job_name=$JOB_NAME"
  "--policy.device=$DEVICE"
  "--wandb.enable=$WANDB_ENABLE"
)

if [ -n "$POLICY_REPO_ID" ]; then
  cmd+=("--policy.repo_id=$POLICY_REPO_ID")
fi

if [ "$#" -gt 0 ]; then
  cmd+=("$@")
fi

printf 'Running command:\n%s\n' "${cmd[*]}"
"${cmd[@]}"
