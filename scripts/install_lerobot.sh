#!/usr/bin/env bash
set -euo pipefail

LEROBOT_DIR="${1:-$HOME/repo/lerobot}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BRANCH="${BRANCH:-main}"
UPDATE_EXISTING="${UPDATE_EXISTING:-0}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple}"
PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL:-}"
INSTALL_EXTRAS="${INSTALL_EXTRAS:-training}"

if ! command -v git >/dev/null 2>&1; then
  echo "git is required but not found in PATH." >&2
  exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable '$PYTHON_BIN' was not found." >&2
  exit 1
fi

if [ -d "$LEROBOT_DIR/.git" ]; then
  if [ "$UPDATE_EXISTING" = "1" ]; then
    git -C "$LEROBOT_DIR" fetch --all --tags
    git -C "$LEROBOT_DIR" checkout "$BRANCH"
    git -C "$LEROBOT_DIR" pull --ff-only origin "$BRANCH"
  fi
else
  mkdir -p "$(dirname "$LEROBOT_DIR")"
  git clone --branch "$BRANCH" https://github.com/huggingface/lerobot.git "$LEROBOT_DIR"
fi

"$PYTHON_BIN" -m venv "$LEROBOT_DIR/.venv"
# shellcheck disable=SC1091
source "$LEROBOT_DIR/.venv/bin/activate"

python -m pip install --upgrade pip setuptools wheel
pip_args=(--index-url "$PIP_INDEX_URL")
if [ -n "$PIP_EXTRA_INDEX_URL" ]; then
  pip_args+=(--extra-index-url "$PIP_EXTRA_INDEX_URL")
fi

python -m pip install "${pip_args[@]}" -e "$LEROBOT_DIR[$INSTALL_EXTRAS]"

echo "LeRobot install complete."
echo "Activate with: source $LEROBOT_DIR/.venv/bin/activate"
echo "Verify with: lerobot-info"
