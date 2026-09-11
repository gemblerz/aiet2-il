# GR00T N1.7 training: SO101 (`dominicdx/so101_overhead`)

## 0) Install LeRobot with GR00T support

```bash
cd <repo-root>
chmod +x scripts/install_lerobot.sh scripts/train_vla_policy.sh
PIP_INDEX_URL=https://pypi.org/simple \
PIP_EXTRA_INDEX_URL= \
INSTALL_EXTRAS=training,groot \
./scripts/install_lerobot.sh
source ../lerobot/.venv/bin/activate
lerobot-info
```

## 1) Login (required for gated base model access)

```bash
hf auth login
```

## 2) Train GR00T N1.7 (embodiment via prompt)

```bash
cd <repo-root>
source ../lerobot/.venv/bin/activate
CUDA_VISIBLE_DEVICES=0,1 NUM_PROCESSES=2 EPOCHS=8 DEVICE=cuda ./scripts/train_vla_policy.sh groot
```

When prompted, enter embodiment tag:

```text
new_embodiment
```

## 3) Train GR00T N1.7 (non-interactive mode)

```bash
cd <repo-root>
source ../lerobot/.venv/bin/activate
CUDA_VISIBLE_DEVICES=0,1 \
NUM_PROCESSES=2 \
EPOCHS=8 \
DEVICE=cuda \
EMBODIMENT_TAG=new_embodiment \
./scripts/train_vla_policy.sh groot
```

## 4) Defaults applied by script

```text
--policy.type=groot
--policy.base_model_path=nvidia/GR00T-N1.7-3B
--policy.embodiment_tag=<your input>
--policy.use_bf16=true
--dataset.image_transforms.enable=true
--policy.push_to_hub=false
```

## 5) Output paths

```text
Logs:
./artifacts/logs/

Checkpoints:
./artifacts/train/
```
