## Autospeed Codes for cobot magic platform Usage

#### Preparation
```python
cd example/autospeed_cobot_magic/repos
git clone https://github.com/CleanDiffuserTeam/CleanDiffuser.git
```

The language encoder is available at [[all-MiniLM-L6-v2]](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).

#### Training
```python
python example/autospeed_cobot_magic/scripts/train.py
```

#### Real Robot Inference
```python
python example/autospeed_cobot_magic/scripts/inference.py
```

#### Pretrained Checkpoints
We have made the real-robot checkpoints trained on the cobot magic dual-arm platform publicly available for the community to use.
You can download it here. [[Pretrained Checkpoints]](https://huggingface.co/Telon1/autospeed_cobot_magic_ckpt)