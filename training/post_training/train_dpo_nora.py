import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional

import torch
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf
import dlimp as dl
from PIL import Image
import dlimp as dl
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import AutoProcessor, PreTrainedTokenizerBase, Qwen2_5_VLForConditionalGeneration
from transformers import SchedulerType, get_scheduler

from qwen_vl_utils import process_vision_info

import numpy as np
from tqdm import tqdm
import wandb
from modelling_expert_with_dpo import VLAWithExpert
logger = get_logger(__name__)
import os
from datasets import load_from_disk


class TrainingConfig:
    def __init__(
        self,
        per_device_batch_size: int = 16,
        learning_rate: float = 5e-7,
        gradient_accumulation_steps: int = 1,
        num_warmup_steps: int = 150,
        max_train_steps: int = 8000,
        output_dir: str = './output_dir',
        resume_from_checkpoint: str = '',
        wandb_project_name: str = "Nora-1.5 DPO ",
        data_root_dir: str = '/dataset_root_dir',
        checkpoint_save_frequency: int = 2000,
        logging_frequency: int = 50,
        gradient_clipping: Optional[float] = None, # Add gradient clipping option
    ):
        self.per_device_batch_size = per_device_batch_size
        self.learning_rate = learning_rate
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_warmup_steps = num_warmup_steps
        self.max_train_steps = max_train_steps
        self.output_dir = output_dir
        self.resume_from_checkpoint = resume_from_checkpoint ## This is used to continue a training by loadinng the optimizer states, model weights etc ... 
       
        self.data_root_dir = data_root_dir
        self.wandb_project_name = wandb_project_name
        self.checkpoint_save_frequency = checkpoint_save_frequency
        self.logging_frequency = logging_frequency
        self.gradient_clipping = gradient_clipping

# --- 2. Data Loading and Preprocessing ---
def load_and_prepare_dataset(config: TrainingConfig): 
    """Loads preference dataset dataset."""
    from datasets import load_from_disk
    
    dataset = load_from_disk(config.data_root_dir)
    
    return dataset

def map_fast_token_to_vlm_action(tokens: List[str]) -> str:
    """Maps fast action tokens to the VLM action format.
    Action token 0 is mapped to the string <robot_action_0>  ... and so on 
    """
    return ''.join([f"<robot_action_{token}>" for token in tokens])

def process_example(example: Dict[str, Any], fast_tokenizer: AutoProcessor) -> Dict[str, Any]:
    """Processes a single example from the dataset."""
    pixel_values = example['image']
    action = example['chosen_action'].unsqueeze(0)
    instruction = example['instruction']
    fast_tokens = fast_tokenizer(action)
    vlm_action = map_fast_token_to_vlm_action(fast_tokens[0])

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pixel_values},
                {"type": "text", "text": instruction},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": vlm_action},
            ],
        },
    ]
    return messages

def resize_image(image1):


    #image1 = tf.cast(image1*255, dtype=tf.uint8)
    image1 = np.array(image1)
    image1 = dl.transforms.resize_image(image1, size=(224,224))

    image1 = Image.fromarray(image1.numpy())
    return image1

def collate_fn(examples,processor,fast_tokenizer):
    messages = []

    chosen_actions = []
    rejected_actions = []
    for example in examples:
        example['image'] = resize_image(example['current_frame'])

        example['generated_action'] = torch.tensor(example['generated_action'])
        example['ground_truth_action'] = torch.tensor(example['ground_truth_action'])

        reward = torch.tensor(example['goal_energy'])+0.1*torch.mean(torch.abs(example['generated_action']- example['ground_truth_action'][:5,:]),axis=(1,2))
        
       

        idx_min = reward.argmin()

        idx_max = reward.argmax()

        chosen_action = example['generated_action'][idx_min,:,:] ## We choose the action with minimum energy and closest to GT action

        
        reject_action = example['generated_action'][idx_max,:,:]
        
        

        example['chosen_action'] = chosen_action

        chosen_actions.append(chosen_action)
        rejected_actions.append(reject_action)
        message = process_example(example,fast_tokenizer)
        messages.append(message)
        
    chosen_actions = torch.stack(chosen_actions,dim=0)
    rejected_actions = torch.stack(rejected_actions,dim=0)
    text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    image_inputs, video_inputs = process_vision_info(messages)
    batch_input = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    action_token_min = 151665
    action_token_max = 153712
    labels = batch_input['input_ids'].clone()
    # For each sequence in the batch, find the first occurrence of an action token.
    expert_attention_mask = batch_input['attention_mask'].clone()
    for i in range(labels.size(0)):
        seq = labels[i]
        # Create a mask for tokens within the action token range.
        mask_seq = (seq >= action_token_min) & (seq <= action_token_max)
        
        nonzero_indices = torch.nonzero(mask_seq, as_tuple=False)

        if nonzero_indices.numel() > 0:
            first_action_index = nonzero_indices[0].item()
            # Mask out all tokens before the first action token.
            seq[:first_action_index] = -100

            expert_attention_mask[i, first_action_index:] = 0

        else:
            # If no action token is found, mask the entire sequence.
            seq[:] = -100
    
        
                


    labels[labels == processor.tokenizer.pad_token_id] = -100 ## mask out pad tokens as well
    batch_input['labels'] = labels
    batch_input['expert_attention'] = expert_attention_mask
    batch_input['actions'] = torch.cat([chosen_actions,rejected_actions])
    return batch_input

# --- 3. Model Initialization ---
def load_model_and_processor(accelerator: Accelerator):
    """Loads the model and processor."""
    processor = AutoProcessor.from_pretrained('declare-lab/nora')
    processor.tokenizer.padding_side = 'left'
    
    model = VLAWithExpert.from_pretrained('declare-lab/nora-1.5-libero')
    fast_tokenizer = AutoProcessor.from_pretrained(
            "physical-intelligence/fast", trust_remote_code=True
        )
   
    accelerator.print("Pretrained weights loaded from libero")

    return model, processor, fast_tokenizer

# --- 4. Training Loop ---
def train(config: TrainingConfig):
    """Main training loop."""
    from accelerate import DistributedDataParallelKwargs

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    
    accelerator = Accelerator(gradient_accumulation_steps=config.gradient_accumulation_steps,kwargs_handlers=[ddp_kwargs])
    accelerator.dataloader_config.dispatch_batches =  False
    logger.info(accelerator.state, main_process_only=False)

    # Initialize Weights and Biases
    if accelerator.is_main_process:
        wandb.init(project=config.wandb_project_name)

    # Load model and processor
    model, processor, fast_tokenizer  = load_model_and_processor(accelerator)

    # Initialize reference model for DPO.
    import copy
    model.ref_action_expert = copy.deepcopy(model.action_expert)
    model.ref_action_out_proj = copy.deepcopy(model.action_out_proj)
    model.ref_action_expert.eval()
    model.ref_action_out_proj.eval()
    model.ref_action_expert.requires_grad_(False)
    model.ref_action_out_proj.requires_grad_(False)
    
    for param in model.ref_action_expert.parameters():
        param.requires_grad = False
    for param in model.ref_action_out_proj.parameters():
        param.requires_grad = False
    
    with accelerator.main_process_first():
        train_dataset = load_and_prepare_dataset(config)
        

    # Create DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.per_device_batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        collate_fn=lambda examples: collate_fn(examples, processor,fast_tokenizer)
    )

    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=1e-8,
        eps=1e-8,
    )

    # Initialize learning rate scheduler
    max_train_steps = config.max_train_steps
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=config.num_warmup_steps,
        num_training_steps=max_train_steps
    )

    # Prepare everything with Accelerator
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    # Resume from checkpoint if provided
    if config.resume_from_checkpoint:
        accelerator.load_state(config.resume_from_checkpoint)
        accelerator.print(f"Resumed from local checkpoint: {config.resume_from_checkpoint}")

    # Training loop
    # Right now we assume single node training. I did not test on multi node training.
    total_batch_size = config.per_device_batch_size * accelerator.num_processes * config.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num steps = {config.max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {config.per_device_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    completed_steps = 0
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    

    while completed_steps < max_train_steps:
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                optimizer.zero_grad()

                actions = batch.pop('actions')
                #actions,_ = actions.chunk(2,dim=0)
                #actions_w,actions_l = actions.chunk(2)
                output = model(batch,actions,do_dpo=True)
    
                loss = output['combined_loss']

                
                accelerator.backward(loss)

                

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    if config.gradient_clipping is not None:
                        accelerator.clip_grad_norm_(model.parameters(), config.gradient_clipping)

                    optimizer.step()
                    lr_scheduler.step()

                # Logging
                    completed_steps += 1
                    if completed_steps % config.logging_frequency == 0:
                        
                        
                        if accelerator.is_main_process:
                            
                            total_norm = 0.0
                            for p in model.parameters():
                                if p.grad is not None:
                                    total_norm += p.grad.data.norm(2).item() ** 2
        
                            total_norm = total_norm**0.5
                            lr = lr_scheduler.get_last_lr()[0]
                        
                            logger.info(f"Step {completed_steps}, Loss: {loss.item()}, Grad Norm: {total_norm},DPO Loss:{output['dpo_loss'].item()}, VLM Loss: {output['vlm_loss'].item()}")
                            
                            lr = lr_scheduler.get_last_lr()[0]
                            result = {
                                "train_combined_loss": loss.item(),
                                "implicit_acc":output['implicit_acc'].item(),
                                "dpo_loss": output['dpo_loss'].item(),
                                "vlm_loss": output['vlm_loss'].item(),
                                "grad_norm": total_norm,
                                "learning_rate": lr,
                            }
                            wandb.log(result, step=completed_steps)

                            
                

            # Checkpointing
            if completed_steps% config.checkpoint_save_frequency == 0 and completed_steps > 0:
                accelerator.save_state(os.path.join(config.output_dir, f"steps_{completed_steps}"))
            


    # Save final checkpoint
    accelerator.save_state(os.path.join(config.output_dir, f"steps_{completed_steps}"))
    
    wandb.finish()

def main():
    # Initialize training configuration
    config = TrainingConfig()

    # Set up basic logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Run the training
    train(config)

if __name__ == "__main__":
    main()