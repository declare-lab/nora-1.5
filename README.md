# NORA-1.5: A Vision-Language-Action Model Trained using World Model- and Action-based Preference Rewards

[![Project Website](https://img.shields.io/badge/Project-Website-blue.svg)](https://declare-lab.github.io/nora-1.5)
[![Model](https://img.shields.io/badge/Model-NORA--1.5-brightgreen)](https://huggingface.co/declare-lab/nora-1.5)
[![arXiv](https://img.shields.io/badge/arXiv-2511.14659-b31b1b.svg)](https://arxiv.org/abs/2511.14659)
![Status](https://img.shields.io/badge/Status-Active-orange)

🔥 Project NORA is supported by Gemini and Lambda Labs! We are thankful to them.

NORA-1.5 is a **Vision-Language-Action (VLA)** model that improves generalization and real-world decision making through **post-training with world-model-based and action-based preference rewards**.  
The model builds upon the NORA foundation to achieve stronger **instruction following**, **closed-loop control**, and **real-robot success**, demonstrating reliability across **LIBERO** and **SimplerEnv** environments.

This repository consolidates the full open-source release of **model checkpoints**, **inference code**, **training code**, and **evaluation tools**, along with documentation and examples.

<p align="center">
  <img src="https://declare-lab.github.io/assets/images/nora-1.5-arxiv-teaser.png" width="100%">
</p>


---
## Setup guide
First, prepare a conda environment.
```
conda create -n nora1_5 python=3.10 -y
conda activate nora1_5
```
Clone repository
``` 
git clone https://github.com/declare-lab/nora-1.5.git
```
Install requirements
```
pip install -r requirements.txt
```
## 🌐 Project Website

🔗 **https://declare-lab.github.io/nora-1.5**
 
---

## 🚀 Key Features

- **Vision-Language-Action architecture** with enhanced **task completion rate** and **distraction rate**
- **Action-based preference optimization** using expert preference rewards  
- **World-model-based preference learning** for improved planning and consistency  
- Strong **closed-loop control**, enabling deployment in real robot settings  
- Supports **multi-task**, **long-horizon**, and **few-shot generalization**  
- Compatible with **LeRobot**, **LIBERO**, **SimplerEnv**, and custom environments  

---




## 📆 TODO <a name="todos"></a>  ~
- [x] Release the inference code of Nora-1.5
- [x] Release all relevant model checkpoints(Pretrained, libero, SimplerEnv etc)
- [x] Release the training/fine-tuning code of Nora-1.5 with LeRobot Dataset
- [x] Release SimplerEnv evaluation code 

## Minimal Inference Sample (Will update)
```python
from inference.modelling_expert import VLAWithExpert

model = VLAWithExpert.from_pretrained("declare-lab/nora-1.5") 
outputs = model.sample_actions(PIL IMAGE,instruction,num_steps=10) ## Outputs 7 Dof action of normalized action
```
## How to train/finetune on your own Lerobot dataset.
To train/finetune NORA-1.5 on your own Lerobot dataset, there are 2 main steps that is required. 
1: Compute normalization statistic of your Lerobot dataset. Note that NORA-1.5 is pretrained in delta action space, hence we will need to compute the normalization statistic for delta action. 
Run the script 
```python python utils/compute_norm_stats.py --dataset_path='YOUR LEROBOT DATASET' --delta_transform```
This will create a norm_stats.json in your lerobot dataset local directory, or remote directory (base on whether your dataset is local on remote).


If your dataset is in delta action space and you have already computed the normalization statistic, you may skip this step.

2: 
Modify the REMAP_KEY for mapping dictionary key name in your lerobot dataset.
https://github.com/declare-lab/nora-1.5/blob/be1376679daad51601e96889efaded00d7243d62/training/lerobot/train_lerobot.py#L37-L42
Set up training hyperparameter, dataset_dir, output_dir in training/lerobot/train_lerobot.py

We use huggingface's accelerator for training. Set up your accelerate config via ```python accelerate config ```


Run training with accelerate launch --config_file='config.yaml' training/lerobot/train_lerobot.py!!!


We jointly optimize cross entropy loss(on FAST token) and flow matching loss on action expert, hence we can use sample discrete action via FAST tokenizer, or continous action via flow matching (action expert). Base on our experiment on Galaxea A1, we found that discrete action performs better than continous action. However, in simulation such as SimplerEnv and LIBERO, continous action outpeform discrete action. Feel free to try both action sampling method.

## 🤗 Model Zoo

<table>
  <tr>
    <th>Model Name</th>
    <th>Backbone</th>
    <th>Note</th>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/declare-lab/nora-1.5">declare-lab/nora-1.5</a></td>
    <td>declare-lab/nora-1.5</td>
     <td>Pretrained on OXE. Jointly optimize cross entropy loss and flow matching loss</a></td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/declare-lab/nora-1.5-fractal-dpo">declare-lab/nora-1.5-fractal-dpo</a></td>
    <td>declare-lab/nora-1.5</td>
    <td>Finetuned on fractal and perform DPO via the method detailed in the paper</a></td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/declare-lab/nora-1.5-libero">declare-lab/nora-1.5-libero</a></td>
    <td>declare-lab/nora-1.5</td>
    <td>Finetuned on 4 LIBERO subset mixed</a></td>
  </tr>
</table>


## SimplerEnv evaluation
Navigate to  https://github.com/hungchiayu1/SimplerEnv-OpenVLA

## LIBERO Evaluation
We used [OpenVLA's](https://github.com/openvla/openvla) code base to perform LIBERO evaluation. To perform LIBERO evaluation, follow the instruction in OpenVLA and set up the evaluation accordingly. 

Copy the inference folder to openvla/experiments/robot/libero and modify the inference function of run_libero_eval.py in OpenVLA's codebase.

```python
from inference.modelling_expert import VLAWithExpert

model = VLAWithExpert.from_pretrained("declare-lab/nora-1.5-libero") 
outputs = model.sample_actions(PIL IMAGE,instruction,num_steps=10) ## Outputs 7 Dof action of normalized action
```

## Acknowledgement
This repository is built based on [OpenVLA](https://github.com/openvla/openvla), [Open X-Embodiment](https://github.com/google-deepmind/open_x_embodiment?tab=readme-ov-file),[transformers](https://github.com/huggingface/transformers), [accelerate](https://github.com/huggingface/accelerate), [Qwen2.5 VL](https://github.com/QwenLM/Qwen2.5-VL), [Lerobot](https://github.com/huggingface/lerobot), [SpatialVLA](https://github.com/SpatialVLA/SpatialVLA).  Thanks!
. Thanks for their contribution!

## Citation

```bibtex
@article{hung2025nora15,
  title={NORA-1.5: A Vision-Language-Action Model Trained using World Model- and Action-Based Preference Rewards},
  author={Hung, Chia-Yu and Majumder, Navonil and Deng, Haoyuan, Liu Renhang, Yankang Ang, Amir Zadeh, Chuan Li, Dorien Herremans, Ziwei Wang, and Soujanya Poria},
  journal={arXiv preprint},
  year={2025}
}
```
