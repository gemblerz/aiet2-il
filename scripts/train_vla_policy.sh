#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

POLICY="${1:-}"
if [ -z "$POLICY" ]; then
  echo "Usage: $0 <smolvla|pi05|groot> [extra lerobot-train args...]" >&2
  exit 1
fi
shift || true

LEROBOT_DIR="${LEROBOT_DIR:-$HOME/repo/lerobot}"
DATASET_REPO_ID="${DATASET_REPO_ID:-dominicdx/so101_overhead}"
EPOCHS="${EPOCHS:-8}"
BATCH_SIZE="${BATCH_SIZE:-8}"
DEVICE="${DEVICE:-cuda}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOG_FREQ="${LOG_FREQ:-1}"
SAVE_FREQ="${SAVE_FREQ:-1000}"
WANDB_PROJECT="${WANDB_PROJECT:-lerobot-vla-so101}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"
WANDB_MODE="${WANDB_MODE:-offline}"
DATASET_ROOT="${DATASET_ROOT:-$REPO_ROOT/artifacts/datasets/${DATASET_REPO_ID}}"

if [ -x "$LEROBOT_DIR/.venv/bin/lerobot-train" ]; then
  export PATH="$LEROBOT_DIR/.venv/bin:$PATH"
fi

if ! command -v lerobot-train >/dev/null 2>&1; then
  echo "lerobot-train command not found." >&2
  echo "Install LeRobot first. For VLA/GR00T: INSTALL_EXTRAS=training,smolvla,pi,groot ./scripts/install_lerobot.sh" >&2
  exit 1
fi
if ! command -v accelerate >/dev/null 2>&1; then
  echo "accelerate command not found." >&2
  echo "Install LeRobot training extras first: INSTALL_EXTRAS=training,smolvla,pi,groot ./scripts/install_lerobot.sh" >&2
  exit 1
fi

if ! command -v curl >/dev/null 2>&1 || ! command -v jq >/dev/null 2>&1; then
  echo "curl and jq are required." >&2
  exit 1
fi
if ! command -v hf >/dev/null 2>&1; then
  echo "hf CLI is required." >&2
  exit 1
fi

if [ "$POLICY" != "smolvla" ] && [ "$POLICY" != "pi05" ] && [ "$POLICY" != "groot" ]; then
  echo "Unsupported policy '$POLICY'. Use smolvla, pi05, or groot." >&2
  exit 1
fi

total_frames="$(
  curl -sSL "https://huggingface.co/datasets/${DATASET_REPO_ID}/resolve/main/meta/info.json" \
    | jq -r '.total_frames'
)"
if ! [[ "$total_frames" =~ ^[0-9]+$ ]]; then
  echo "Failed to read total_frames from dataset info." >&2
  exit 1
fi

if ! [[ "$NUM_PROCESSES" =~ ^[0-9]+$ ]] || [ "$NUM_PROCESSES" -lt 1 ]; then
  echo "NUM_PROCESSES must be a positive integer." >&2
  exit 1
fi

steps="$(( (total_frames * EPOCHS + (BATCH_SIZE * NUM_PROCESSES) - 1) / (BATCH_SIZE * NUM_PROCESSES) ))"
if [ "$steps" -lt 1 ]; then
  echo "Computed steps is < 1. Check EPOCHS and BATCH_SIZE." >&2
  exit 1
fi

if [ ! -f "$DATASET_ROOT/meta/info.json" ]; then
  mkdir -p "$DATASET_ROOT"
  echo "Downloading dataset to local root: $DATASET_ROOT"
  hf download "$DATASET_REPO_ID" \
    --repo-type dataset \
    --local-dir "$DATASET_ROOT"
fi

OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/artifacts/train/${POLICY}_${DATASET_REPO_ID//\//_}_e${EPOCHS}}"
JOB_NAME="${JOB_NAME:-${POLICY}_${DATASET_REPO_ID##*/}_e${EPOCHS}}"
RUN_SUFFIX="${RUN_SUFFIX:-}"
if [ -z "$RUN_SUFFIX" ] && [ -d "$OUTPUT_DIR" ] && [ "${RESUME:-false}" != "true" ]; then
  RUN_SUFFIX="$(date +%Y%m%d_%H%M%S)"
fi
if [ -n "$RUN_SUFFIX" ]; then
  OUTPUT_DIR="${OUTPUT_DIR}_${RUN_SUFFIX}"
  JOB_NAME="${JOB_NAME}_${RUN_SUFFIX}"
fi
LOG_DIR="${LOG_DIR:-$REPO_ROOT/artifacts/logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/${JOB_NAME}.log}"
mkdir -p "$(dirname "$OUTPUT_DIR")" "$LOG_DIR"

base_cmd=(
  accelerate launch "--num_processes=$NUM_PROCESSES" "$(which lerobot-train)"
  "--dataset.repo_id=$DATASET_REPO_ID"
  "--dataset.root=$DATASET_ROOT"
  "--output_dir=$OUTPUT_DIR"
  "--job_name=$JOB_NAME"
  "--batch_size=$BATCH_SIZE"
  "--num_workers=$NUM_WORKERS"
  "--steps=$steps"
  "--policy.device=$DEVICE"
  "--policy.push_to_hub=false"
  "--log_freq=$LOG_FREQ"
  "--save_freq=$SAVE_FREQ"
)

if [ "$WANDB_ENABLE" = "true" ]; then
  base_cmd+=(
    "--wandb.enable=true"
    "--wandb.mode=$WANDB_MODE"
    "--wandb.project=$WANDB_PROJECT"
  )
else
  base_cmd+=("--wandb.enable=false")
fi

if [ "$POLICY" = "smolvla" ]; then
  pretrained_path="${PRETRAINED_PATH:-lerobot/smolvla_base}"
  policy_cmd=(
    "--policy.path=$pretrained_path"
    "--policy.empty_cameras=1"
    "--rename_map={\"observation.images.top\":\"observation.images.camera1\",\"observation.images.side\":\"observation.images.camera2\"}"
  )
elif [ "$POLICY" = "pi05" ]; then
  pretrained_path="${PRETRAINED_PATH:-lerobot/pi05_base}"
  policy_cmd=(
    "--policy.type=pi05"
    "--policy.pretrained_path=$pretrained_path"
    "--policy.normalization_mapping={\"ACTION\":\"MEAN_STD\",\"STATE\":\"MEAN_STD\",\"VISUAL\":\"IDENTITY\"}"
  )
else
  base_model_path="${BASE_MODEL_PATH:-nvidia/GR00T-N1.7-3B}"
  embodiment_tag="${EMBODIMENT_TAG:-}"
  if [ -z "$embodiment_tag" ]; then
    if [ -t 0 ]; then
      read -r -p "Enter GR00T embodiment tag (e.g. new_embodiment, libero_sim): " embodiment_tag
    else
      echo "EMBODIMENT_TAG is required for groot in non-interactive mode." >&2
      echo "Example: EMBODIMENT_TAG=new_embodiment ./scripts/train_vla_policy.sh groot" >&2
      exit 1
    fi
  fi
  if [ -z "$embodiment_tag" ]; then
    echo "EMBODIMENT_TAG cannot be empty for groot." >&2
    exit 1
  fi

  policy_cmd=(
    "--policy.type=groot"
    "--policy.base_model_path=$base_model_path"
    "--policy.embodiment_tag=$embodiment_tag"
    "--policy.use_bf16=true"
    "--dataset.image_transforms.enable=true"
  )
fi

cmd=( "${base_cmd[@]}" "${policy_cmd[@]}" "$@" )

echo "Dataset total_frames: $total_frames"
echo "Target epochs: $EPOCHS"
echo "Num processes (GPUs): $NUM_PROCESSES"
echo "Computed steps: $steps"
echo "Output dir: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo "Running:"
printf '%s ' "${cmd[@]}"
printf '\n'

set -o pipefail
"${cmd[@]}" 2>&1 | tee "$LOG_FILE"
