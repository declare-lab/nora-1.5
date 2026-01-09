import os
import json
import logging
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import copy

from accelerate import Accelerator
from accelerate.logging import get_logger
from transformers import get_scheduler

import sys
sys.path.insert(0, "vjepa2")

from app.vjepa_droid.utils import init_video_model, load_pretrained
from app.vjepa_droid.transforms import make_transforms

logger = get_logger(__name__)


# --- 1. Configuration ---
class VJEPATrainingConfig:
    """Configuration class for V-JEPA training, adapted for the accelerate framework."""
    """Modifed from https://github.com/facebookresearch/vjepa2 """
    def __init__(self, **kwargs):
        
        self.output_dir: str = '/your_output'
       
        self.pretrain_checkpoint: Optional[str] = None
        self.seed: int = 42
        self.wandb_project_name: str = "vjepa2_ac"
        self.rlds_data_root_dir: str = '/path/to/rlds/data'

        # Model
        self.model_name: str = 'vit_giant_xformers'
        self.pred_depth: int = 24
        self.pred_embed_dim: int = 1024
        self.pred_is_frame_causal = True
        self.patch_size: int = 16
        self.tubelet_size: int = 2
        self.pred_num_heads = 16

        # Data
        self.per_device_batch_size: int = 8
        self.max_num_frames: int = 8
        self.crop_size: int = 256
        

        # Optimization
        self.learning_rate: float = 0.00002
        self.weight_decay: float = 0.05
        self.num_warmup_steps: int = 5000
        self.max_train_steps: int = 200000
        self.gradient_accumulation_steps: int = 1
        self.gradient_clipping: Optional[float] = 1.0

        # Loss
        self.loss_exp: float = 1.0
        self.normalize_reps: bool = True
        self.auto_steps: int = 2 # Num autoregressive steps

        # Logging & Checkpointing
        self.logging_frequency: int = 100
        self.checkpoint_save_frequency: int = 10000

        # Update with any kwargs provided
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)



       
        
    


# --- 3. Training Loop ---
def train(config: VJEPATrainingConfig):
    """Main training loop using accelerate."""
    from accelerate import DistributedDataParallelKwargs
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="wandb",kwargs_handlers=[kwargs]
    )
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    accelerator.init_trackers(
        project_name=config.wandb_project_name,
        config=config.__dict__
    )

    

    target_encoder ,predictor = torch.hub.load("facebookresearch/vjepa2","vjepa2_ac_vit_giant")
    
    for param in target_encoder.parameters():
        param.requires_grad = False

    from pathlib import Path
    from rlds_datasets import  RLDSBatchTransform, RLDSDataset
    crop_size = 256
    tokens_per_frame = int((crop_size // target_encoder.patch_size) ** 2)
    transform = make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1., 1.),
        random_resize_scale=(1., 1.),
        reprob=0.,
        auto_augment=False,
        motion_shift=False,
        crop_size=crop_size,
    )
    
    train_dataset = RLDSDataset(
                data_root_dir=Path(config.),
                data_mix="libero_combined_no_noops",
                batch_transform=RLDSBatchTransform(),
                resize_resolution=(256, 256),
                shuffle_buffer_size=256_000,
                train=True,
                image_aug=True
            )
   
    def collate_fn(batch):
        clip = []
        for item in batch:
            clip.append(transform(item['image']))
        
        actions =  torch.stack([torch.tensor(item['action']) for item in batch],dim=0)

        return {  'clips': torch.stack(clip,dim=0),
            'actions':actions}
    
    batch_size = config.per_device_batch_size
    normalize_reps = config.normalize_reps 
    auto_steps = config.auto_steps
    loss_exp = config.loss_exp
    max_num_frames = config.max_num_frames
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        
    )
    
    

    optimizer = torch.optim.AdamW(
        predictor.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=config.num_warmup_steps*accelerator.num_processes,
        num_training_steps=config.max_train_steps*config.gradient_accumulation_steps*accelerator.num_processes,
    )


    


    predictor, target_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
         predictor, target_encoder, optimizer, train_dataloader, lr_scheduler
    )

    
    progress_bar = tqdm(range(config.max_train_steps), disable=not accelerator.is_local_main_process)
    
    tokens_per_frame = int((config.crop_size // config.patch_size) ** 2)
    

    ## Copied from vjepa2/app/vjepa_droid/train.py
    ## We removed extrinsics as we don't have them in our dataset
    def forward_target(c):
        with torch.no_grad():
            c = c.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
            h = target_encoder(c)
            h = h.view(batch_size, max_num_frames, -1, h.size(-1)).flatten(1, 2)
            if normalize_reps:
                h = F.layer_norm(h, (h.size(-1),))
            return h

    def forward_predictions(z,actions,states):

        def _step_predictor(_z, _a, _s):
            _z = predictor(_z, _a, _s)
            if normalize_reps:
                _z = F.layer_norm(_z, (_z.size(-1),))
            return _z

        # -- one step of predictor with teacher forcing
        _z, _a, _s = z[:, :-tokens_per_frame], actions, states[:, :-1]#, extrinsics[:, :-1]
        z_tf = _step_predictor(_z, _a, _s)

        # -- full auto-regressive rollouts of predictor
        _z = torch.cat([z[:, : tokens_per_frame], z_tf[:, : tokens_per_frame]], dim=1)
        for n in range(1, auto_steps):
            _a, _s = actions[:, : n + 1], states[:, : n + 1]#, extrinsics[:, : n + 1]
            _z_nxt = _step_predictor(_z, _a, _s)[:, -tokens_per_frame:]
            _z = torch.cat([_z, _z_nxt], dim=1)
        z_ar = _z[:, tokens_per_frame:]

        return z_tf, z_ar

    def loss_fn(z, h):
        _h = h[:, tokens_per_frame : z.size(1) + tokens_per_frame]
        return torch.mean(torch.abs(z - _h) ** loss_exp) / loss_exp
    completed_steps = 0
    while completed_steps < config.max_train_steps:
        predictor.train()
        target_encoder.eval()
        
        for batch in train_dataloader:
            clips, actions = batch['clips'],batch['actions'][:,:-1,:]
            #accelerator.log("CLIPS SHAPE",clips.shape,actions.shape,states.shape)
            with accelerator.accumulate(predictor):
                

                h = forward_target(clips)
                # The predictor uses the target encoder's representations as input context
                z_tf, z_ar = forward_predictions(h,actions,torch.zeros_like(batch['actions'],device=batch['actions'].device)) 
                
                # Step 2. Loss Calculation
                jloss = loss_fn(z_tf, h)
                sloss = loss_fn(z_ar, h)
                loss = jloss + sloss
                
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    

                    accelerator.clip_grad_norm_(predictor.parameters(), max_norm=1.0)

                    optimizer.step()
                    lr_scheduler.step()
                
            

                    if completed_steps % config.logging_frequency == 0:
                        loss_log = {"jloss": jloss.detach().item(), "sloss": sloss.detach().item(), "total_loss": loss.detach().item()}
                        lr = lr_scheduler.get_last_lr()[0]
                        accelerator.log({"learning_rate": lr, **loss_log}, step=completed_steps)
                        progress_bar.set_postfix(loss=loss.item(), lr=lr)

                    if completed_steps % config.checkpoint_save_frequency == 0 and completed_steps > 0:
                        output_dir = Path(config.output_dir) / f"checkpoint_{completed_steps}"
                        accelerator.save_state(output_dir)
                        accelerator.print(f"Checkpoint saved at step {completed_steps} to {output_dir}")


                    progress_bar.update(1)
                    completed_steps += 1

            if completed_steps >= config.max_train_steps:
                break
    
    
    


if __name__ == "__main__":

    config = VJEPATrainingConfig(
        output_dir=f'./vjepa2_libero_checkpoints',
        rlds_data_root_dir='path',
        per_device_batch_size=8,
        gradient_accumulation_steps=1,
        max_train_steps=200000, 
        crop_size=256,
        logging_frequency=100,
        checkpoint_save_frequency=20000
    )
    train(config)

    