#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d_%H%M%S)}"

RUN_SUFFIX="$RUN_SUFFIX" "$SCRIPT_DIR/train_vla_policy.sh" smolvla "$@"
RUN_SUFFIX="$RUN_SUFFIX" "$SCRIPT_DIR/train_vla_policy.sh" pi05 "$@"
