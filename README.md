# Nora-1.5

## 📆 TODO <a name="todos"></a>
- [ ] Release the inference code of Nora-1.5
- [ ] Release all relevant model checkpoints(Pretrained, libero, SimplerEnv etc)
- [ ] Release the training/fine-tuning code of Nora-1.5 with LeRobot Dataset
- [ ] Release SimplerEnv evaluation code



## Minimal Inference Sample (Will update)
```
from inference.modelling_expert import VLAWithExpert

model = VLAWithExpert() 
model.to('cuda')
outputs = model.sample_actions(PIL IMAGE,instruction,num_steps=10) ## Outputs 7 Dof action of normalized and unnormalized action
```
