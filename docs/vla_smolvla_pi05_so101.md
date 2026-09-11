# VLA training: SmolVLA + Pi0.5 (8 epochs, no Hub push)

## 0) Prerequisites

```bash
python3 --version
git --version
nvidia-smi
```

## 1) Install LeRobot with VLA extras (outside this repo)

```bash
cd <repo-root>
chmod +x scripts/install_lerobot.sh scripts/train_vla_policy.sh scripts/train_vla_duo_8epochs.sh
PIP_INDEX_URL=https://pypi.org/simple \
PIP_EXTRA_INDEX_URL= \
INSTALL_EXTRAS=training,smolvla,pi \
./scripts/install_lerobot.sh
source ../lerobot/.venv/bin/activate
lerobot-info
```

## 2) Optional login (for gated base models)

```bash
hf auth login
```

## 3) Train SmolVLA for 8 epochs (2 GPUs)

```bash
cd <repo-root>
source ../lerobot/.venv/bin/activate
CUDA_VISIBLE_DEVICES=0,1 NUM_PROCESSES=2 EPOCHS=8 DEVICE=cuda ./scripts/train_vla_policy.sh smolvla
```

## 4) Train Pi0.5 for 8 epochs (2 GPUs)

```bash
cd <repo-root>
source ../lerobot/.venv/bin/activate
CUDA_VISIBLE_DEVICES=0,1 NUM_PROCESSES=2 EPOCHS=8 DEVICE=cuda ./scripts/train_vla_policy.sh pi05
```

## 5) Train both sequentially

```bash
cd <repo-root>
source ../lerobot/.venv/bin/activate
CUDA_VISIBLE_DEVICES=0,1 NUM_PROCESSES=2 EPOCHS=8 DEVICE=cuda ./scripts/train_vla_duo_8epochs.sh
```

## 6) Confirm no Hub push

Training command includes:

```text
--policy.push_to_hub=false
```

Distributed launcher used:

```text
accelerate launch --num_processes=2
```

## 7) Logs and artifacts

```text
Full logs:
./artifacts/logs/

Checkpoints and outputs:
./artifacts/train/
```

## 8) Monitoring (equivalent to TensorBoard)

```text
W&B is disabled by default in scripts:
--wandb.enable=false
```

```bash
# Enable W&B when needed
WANDB_ENABLE=true WANDB_MODE=offline ./scripts/train_vla_policy.sh smolvla
```

## 9) Dataset

```text
dominicdx/so101_overhead
```

## 10) GR00T N1.7

Use guide: [docs/groot_so101.md](./groot_so101.md)
