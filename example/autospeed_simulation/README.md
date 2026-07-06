# AutoSpeed Simulation Implement Guidance

## Preparation

Install the dependencies required by the target simulator before running its scripts. CleanDiffuser is used by the action heads:

```bash
cd example/autospeed_simulation
mkdir -p repos
git clone https://github.com/CleanDiffuserTeam/CleanDiffuser.git repos/CleanDiffuser
```

Download the DINOv2 code and weights for config `encoder_type: dino`:

```text
weights/dinov2/
weights/dinov2_weight/dinov2_vitb14_reg4_pretrain.pth
```

Download the language encoder from [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) when task embeddings need to be generated.

Set the data roots as needed:

```bash
export AUTOSPEED_ROOT=$(pwd)
export ALOHA_SIM_DATA_DIR=data/alohasim
export LIBERO_DATA_DIR=data/libero
export METAWORLD_DATA_DIR=data/metaworld
```

## Training

```bash
python scripts/train_alohasim.py
python scripts/train_libero.py
python scripts/train_metaworld.py
```

## Evaluation

```bash
python scripts/eval_alohasim.py --ckpt-path checkpoints/<run>/snapshot/<step>.pt
python scripts/eval_libero.py --ckpt-path checkpoints/<run>/snapshot/<step>.pt
python scripts/eval_metaworld.py --ckpt-path checkpoints/<run>/snapshot/<step>.pt
```

### AlohaSim high-gain controller

`scripts/eval_alohasim.py` enables the AlohaSim speedup setting by default:

```bash
EVAL_SPEEDUP=true python scripts/eval_alohasim.py --ckpt-path checkpoints/<run>/snapshot/<step>.pt
```

Set `EVAL_SPEEDUP=false` to evaluate with the normal controller XML:

```bash
EVAL_SPEEDUP=false python scripts/eval_alohasim.py --ckpt-path checkpoints/<run>/snapshot/<step>.pt
```

`EVAL_SPEEDUP` is passed to `make_sim_env(task_name, speedup)`. When it is enabled, the simulator loads the high-gain MuJoCo XML files instead of the normal XML files under `/suite/act/assets/`. This increases the gripper actuator gain so that open/close commands respond faster and with stronger tracking during evaluation.


### Pretrained Checkpoints
We have prepared some pretrained checkpoints in the AlohaSim publicly available for the community to use.
You can download it here. [[Pretrained Checkpoints]](https://huggingface.co/Telon1/autospeed_alohasim_ckpt)