# ACT training: SO101 (`dominicdx/so101_overhead`)

## 0) Prerequisites

```bash
python3 --version
git --version
nvidia-smi
```

## 1) Install LeRobot outside this repository

```bash
cd <repo-root>
chmod +x scripts/install_lerobot.sh scripts/train_act_so101_overhead.sh
./scripts/install_lerobot.sh
source ../lerobot/.venv/bin/activate
lerobot-info
```

If your machine injects a custom pip index, use:

```bash
PIP_INDEX_URL=https://pypi.org/simple PIP_EXTRA_INDEX_URL= ./scripts/install_lerobot.sh
```

## 2) Optional logins

```bash
hf auth login
wandb login
```

## 3) Start ACT training (SO101 overhead dataset)

```bash
cd <repo-root>
source ../lerobot/.venv/bin/activate
./scripts/train_act_so101_overhead.sh
```

## 4) Training with explicit options

```bash
cd <repo-root>
source ../lerobot/.venv/bin/activate
DEVICE=cuda \
WANDB_ENABLE=true \
POLICY_REPO_ID=<your-hf-username>/act_so101_overhead \
./scripts/train_act_so101_overhead.sh
```

## 5) Resume from a checkpoint

```bash
cd <repo-root>
source ../lerobot/.venv/bin/activate
./scripts/train_act_so101_overhead.sh --resume=true
```
or:
```bash
lerobot-train --config_path=<your-hf-username>/act_so101_overhead --resume=true
```

## 6) Quick smoke run (short training)

```bash
cd <repo-root>
source ../lerobot/.venv/bin/activate
./scripts/train_act_so101_overhead.sh --steps=200 --job_name=act_so101_overhead_smoke
```

## 7) Output locations

```text
Local training outputs:
./artifacts/train/act_so101_overhead

LeRobot install:
../lerobot
```

## 8) Dataset confirmation

```bash
curl -sS https://huggingface.co/api/datasets/dominicdx/so101_overhead | jq '.id, .sha'
```
