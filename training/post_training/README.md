Post-Training: Generate Synthethic Data → Rollout via VJEPA-AC → DPO
===================================

This folder contains utilities for post-training processing: generate synthetic trajectories, peform rollout using vjepa2-ac , and run Direct DPO on the labelled set.

Overview
--------
- Step 1 — Generate data: run `generate_data.py` to produce synthetic trajectory datasets.
- Step 2 — Label data: run `label_generated_data.py` to convert raw generated data into a labeled dataset suitable for training.
- Step 3 — DPO training: run `train_dpo_nora.py` (or `modelling_expert_with_dpo.py`).

Prerequisites
-------------
- `requirements.txt` in the top-level `nora-1.5` folder.

Typical file locations
----------------------
- Generator script: training/post_training/generate_data.py
- Labeller script: training/post_training/label_generated_data.py
- DPO training: training/post_training/train_dpo_nora.py
- Script for training VJEPA2-AC: training/post_training/vjepa2_train.py

  
How to train VJEPA2-AC
------------------
To train VJEPA2-AC, first clone the vjepa2 repository in this directory

```bash
git clone https://github.com/facebookresearch/vjepa2.git
```

Create a new conda env for vjepa2.

```bash
conda create -n vjepa2-312 python=3.12
conda activate vjepa2-312
cd vjepa2-312
pip install .  # or `pip install -e .` for development mode
```

Modify the training hyperparameter in vjepa2_train.py. Launch training!
```
accelerate launch vjepa2_train.py ## Set your own accelerator config.
```

Quick run examples
------------------

Our dataset structure uses the same RLDS format used by OpenVLA training. You can check more details in the [OpenVLA](https://github.com/openvla/openvla) repository.
You will have to download the RLDS dataset for the dataset you are interested in generating (Eg LIBERO,Fractal). Pass in the RLDS data dir in ```data_root_dir``` argument.

This script will sample a random data point (instruction,image) from the dataset and generate 5 actions by the policy and save the goal image (last frame).

1) Generate data (adjust args for your robot/task):

```python
python training/post_training/generate_data.py \
  --output_dir ./data_generated \
  --n_samples 20000 \
  --data_root_dir /your_rlds_data_dir
```


This script performs rollout using VJEPA2-AC that compute the future frame given the current frame and action chunk. It will then compute the L1 distance betwen the goal image and save them in a new column(goal energy).

2) Label generated data (point this at the directory produced above):

```bash
python label_generated_data.py \
  --model_path vjepa_ac_predictor.safetensors \
  --dataset_dir ./data_generated
```

3) Run DPO training on labeled data (example):
Modify the training config in train_dpo_nora.py. Launch training

```python
accelerate launch train_dpo_nora.py
```


