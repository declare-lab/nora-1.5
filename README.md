# NORA-1.5: A Vision–Language–Action Model Post-Trained With World-Model and Action-Based Preference Rewards

[![Project Website](https://img.shields.io/badge/Project-Website-blue.svg)](https://declare-lab.github.io/nora-1.5)
[![Model](https://img.shields.io/badge/Model-NORA--1.5-brightgreen)](https://huggingface.co/declare-lab/nora-1.5)
[![arXiv](https://img.shields.io/badge/arXiv-2511.14659-b31b1b.svg)](https://arxiv.org/abs/2511.14659)
![Status](https://img.shields.io/badge/Status-Active-orange)

NORA-1.5 is a **Vision-Language-Action (VLA)** model that improves generalization and real-world decision making through **post-training with world-model-based and action-based preference rewards**.  
The model builds upon the NORA foundation to achieve stronger **instruction following**, **closed-loop control**, and **real-robot success**, demonstrating reliability across **LeRobot**, **LIBERO**, and **SimplerEnv** environments.

This repository consolidates the full open-source release of **model checkpoints**, **inference code**, **training code**, and **evaluation tools**, along with documentation and examples.

<p align="center">
  <img src="https://declare-lab.github.io/assets/images/nora-1.5-arxiv-teaser.png" width="100%">
</p>


---

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

## 📦 Repository Structure (will update)



## 📆 TODO <a name="todos"></a>  ~ 1 week
- [ ] Release the inference code of Nora-1.5
- [ ] Release all relevant model checkpoints(Pretrained, libero, SimplerEnv etc)
- [ ] Release the training/fine-tuning code of Nora-1.5 with LeRobot Dataset
- [ ] Release SimplerEnv evaluation code 

## Minimal Inference Sample (Will update)
```python
from inference.modelling_expert import VLAWithExpert

model = VLAWithExpert() 
model.to('cuda')
outputs = model.sample_actions(PIL IMAGE,instruction,num_steps=10) ## Outputs 7 Dof action of normalized and unnormalized action
```

## Citation

```bibtex
@article{hung2025nora15,
  title={NORA-1.5: A Vision-Language-Action Model Trained using World Model- and Action-Based Preference Rewards},
  author={Hung, Chia-Yu and Majumder, Navonil and Deng, Haoyuan, Liu Renhang, Yankang Ang, Amir Zadeh, Chuan Li, Dorien Herremans, Ziwei Wang, and Soujanya Poria},
  journal={arXiv preprint},
  year={2025}
}
```
