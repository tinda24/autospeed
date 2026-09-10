# AutoSpeed Simulation Implementation Guide

## Preparation

Install the dependencies required by the target simulator before running its
scripts.

```bash
cd example/autospeed_simulation
python -m pip install torch-dct huggingface_hub
```

CleanDiffuser is used by the diffusion and flow action heads:

```bash
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

## Pretrained ALOHA Sim Checkpoints

The public checkpoints are available from
[Telon1/autospeed_alohasim_ckpt](https://huggingface.co/Telon1/autospeed_alohasim_ckpt).

Keep each snapshot with its configuration and normalization statistics:

```text
checkpoints/
|-- transfer/
|   |-- agent_config.yaml
|   |-- full_config.yaml
|   |-- stats.hdf5
|   `-- snapshot/
|       `-- 80000.pt
`-- insertion/
    |-- agent_config.yaml
    |-- full_config.yaml
    |-- stats.hdf5
    `-- snapshot/
        `-- 80000.pt
```

## ALOHA Sim Evaluation

The following settings are recommended to reproduce the evaluations. EGL is used
for headless MuJoCo rendering:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EVAL_SPEEDUP=true
export EVAL_USE_NTA=false
export EVAL_TEMPORAL_AGG=false
export EVAL_NUM_ROLLOUTS=50
export EVAL_MAX_TIMESTEPS=400
export EVAL_SAVE_VIDEO=false
export EVAL_SAVE_PLOTS=false
```

Evaluate Transfer Cube:

```bash
python scripts/eval_alohasim.py \
  --ckpt-path checkpoints/transfer/snapshot/80000.pt
```

Evaluate Insertion:

```bash
python scripts/eval_alohasim.py \
  --ckpt-path checkpoints/insertion/snapshot/80000.pt
```

Set `EVAL_TEMPORAL_AGG=true` to evaluate the same checkpoint with temporal
aggregation. The evaluator uses the high-gain ALOHA Sim controller when
`EVAL_SPEEDUP=true`; set it to `false` to use the normal controller XML.

## Training

The simulation training entry points are:

```bash
python scripts/train_alohasim.py
python scripts/train_libero.py
python scripts/train_metaworld.py
```

The default generic Actor configuration is an integration example. Reproducing
the ALOHA Sim results requires the WiseActor-ACT configuration used by the
released checkpoint.

For the LIBERO experiments, we train the policy to predict absolute actions, so
the original delta-action trajectories must be converted to absolute actions
during data preprocessing.

## Other Evaluation Entry Points

```bash
python scripts/eval_libero.py --ckpt-path checkpoints/<run>/snapshot/<step>.pt
python scripts/eval_metaworld.py --ckpt-path checkpoints/<run>/snapshot/<step>.pt
```
