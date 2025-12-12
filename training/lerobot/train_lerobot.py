import os
import json
import logging
from pathlib import Path

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from typing import List, Dict, Any, Callable, Optional

import torch
from torch.utils.data import Dataset, DataLoader

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import AutoProcessor, PreTrainedTokenizerBase, Qwen2_5_VLForConditionalGeneration
from transformers import SchedulerType, get_scheduler
from qwen_vl_utils import process_vision_info
import numpy as np
from tqdm import tqdm
import wandb
from inference.modelling_expert import VLAWithExpert
logger = get_logger(__name__)
import os
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.configs.types import  NormalizationMode, PolicyFeature
from lerobot.policies.normalize import (
    Normalize,
   
)
from typing import Sequence,runtime_checkable,Protocol
import dataclasses

set_seed(42)
## MODIFY THESE BASE ON YOUR LEROBOT DATASET
REMAP_KEY = {
    "state": "observation_state",
    "image": "observation.images.scene",
    "task":"task",

}


'''
Copied from openpi repository. Transforms abosolute to delta


'''
class DataTransformFn(Protocol):
    def __call__(self, data ):
        """Apply transformation to the data.

        Args:
            data: The data to apply the transform to. This is a possibly nested dictionary that contains
                unbatched data elements. Each leaf is expected to be a numpy array. Using JAX arrays is allowed
                but not recommended since it may result in extra GPU memory usage inside data loader worker
                processes.

        Returns:
            The transformed data. Could be the input `data` that was modified in place, or a new data structure.
        """
@dataclasses.dataclass(frozen=True)
class DeltaActions(DataTransformFn):
    """Repacks absolute actions into delta action space."""

    # Boolean mask for the action dimensions to be repacked into delta action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data) :
        if "action" not in data or self.mask is None:
            return data

        state, actions = data[REMAP_KEY['state']], data['action'] ## You might need modify key here depending on your state/action 
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        #print(dims)
        actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["action"] = actions

        return data
    
transform = DeltaActions(mask=[True,True,True,True,True,True,False])





# --- 1. Configuration ---
class TrainingConfig:
    def __init__(
        self,
        per_device_batch_size: int = 24,
        learning_rate: float = 8e-5,
        gradient_accumulation_steps: int = 1,
        num_warmup_steps: int = 1000,
        max_train_steps: int = 10000,
        output_dir: str = './checkpoints',
        resume_from_checkpoint: str = '',
        load_model_weights: Optional[str] = None,
        data_root_dir: str = 'hungchiayu/lerobot_multi_task_1104',
        wandb_project_name: str = "Nora-1.5 Lerobot",
        checkpoint_save_frequency: int = 2000,
        logging_frequency: int = 50,
        delta_transform: bool = True,
        gradient_clipping: Optional[float] = 1.0, # Add gradient clipping option
    ):
        self.per_device_batch_size = per_device_batch_size
        self.learning_rate = learning_rate
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_warmup_steps = num_warmup_steps
        self.max_train_steps = max_train_steps
        self.output_dir = output_dir
        self.resume_from_checkpoint = resume_from_checkpoint ## This is used to continue a training by loadinng the optimizer states, model weights etc ... 
        self.load_model_weights = load_model_weights ## This is the path to a pretrained model weights if you want to finetune the model.
        self.data_root_dir = data_root_dir
        self.delta_transform = delta_transform ## This is extremely important. If set as true, it will transform your actions to delta action. NORA-1.5 is pretrained predicting delta action . 
        self.wandb_project_name = wandb_project_name
        self.checkpoint_save_frequency = checkpoint_save_frequency
        self.logging_frequency = logging_frequency
        self.gradient_clipping = gradient_clipping

# --- 2. Data Loading and Preprocessing ---
def load_and_prepare_dataset(config: TrainingConfig, processor: AutoProcessor): 
    """Loads and prepares the Lerobot dataset."""
    
    
    metadata = LeRobotDatasetMetadata(config.data_root_dir)
    
    path = config.data_root_dir
    delta_timestamps = {
        "action": [t / metadata.fps for t in range(5)], ## action horizon of 5
    }
    ds = LeRobotDataset(path,delta_timestamps=delta_timestamps)
    return ds,metadata

def map_fast_token_to_vlm_action(tokens: List[str]) -> str:
    """Maps fast action tokens to the VLM action format.
    Action token 0 is mapped to the string <robot_action_0>  ... and so on 
    """
    return ''.join([f"<robot_action_{token}>" for token in tokens])

def normalize_action(example,normalizer):
    #normalized_action = torch.cat((normalizer(example)['action'][:,:-1],1-example['action'][:,-1:]),dim=1)
    normalized_action = torch.cat((normalizer(example)['action'][:,:-1],example['action'][:,-1:]),dim=1) ## normalize all action except last dim (grippler dim)
    example['action'] = normalized_action
    return example

   
import tensorflow as tf
from PIL import Image
import dlimp as dl


## IMPORTANT. THIS IS THE IMAGE RESIZING FUNCTION FOR PRETRAINING.
def resize_image(image1):


    
    image1 = tf.cast(image1*255, dtype=tf.uint8)
    image1 = image1.numpy().transpose(1,2,0)
    image1 = dl.transforms.resize_image(image1, size=(224,224))

    image1 = Image.fromarray(image1.numpy())
    return image1

def process_example(example: Dict[str, Any], fast_tokenizer: AutoProcessor) -> Dict[str, Any]:
    """Processes a single example from the dataset."""
    #print("raw action",example['action'])
    
    normalized_action = example['action']

    
    image1 = resize_image(example[REMAP_KEY['image']])
    
    
    
    task =  example[REMAP_KEY['task']]
    
    
    fast_tokens = fast_tokenizer(normalized_action)
    
    vlm_action = map_fast_token_to_vlm_action(fast_tokens[0])
    ## If you want to finetune to take in multiple image, you will have to modify the message templlate below.

    ## Eg  "content": [
   #             {
    #                "type": "image", "image": image1,
     #               "resized_height": 224,
     #               "resized_width": 224,
     #           },
     #             {
    #                "type": "image", "image": image2,
     #               "resized_height": 224,
     #               "resized_width": 224,
     #           }],
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image", "image": image1,
                    "resized_height": 224,
                    "resized_width": 224,
                },
                {"type": "text", "text": task},

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

def collate_fn(examples,processor,fast_tokenizer,normalizer):
        for example in examples:
            example = transform(example)
            example = normalize_action(example,normalizer)

        messages = [process_example(example,fast_tokenizer) for example in examples]

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
               

                ## A hacky way to construct action expert attention mask. We can treat fast tokens as padded tokens (0) when constructing attention mask for action expert via the make_attn_2d function. 
                # This way the action expert cant attend to fast action tokens
                expert_attention_mask[i, first_action_index:] = 0

            else:
                # If no action token is found, mask the entire sequence.
                seq[:] = -100
        
        labels[labels == processor.tokenizer.pad_token_id] = -100 ## mask out pad tokens as well
        batch_input['labels'] = labels
        batch_input['expert_attention'] = expert_attention_mask
        batch_input['actions'] = torch.tensor(np.array([example['action'] for example in examples]))
        #batch_input['states'] = torch.tensor(np.array([example['observation_state'] for example in examples])) probably dont need
        
        return batch_input

# --- 3. Model Initialization ---
def load_model_and_processor( accelerator: Accelerator):
    """Loads the model and processor."""
    processor = AutoProcessor.from_pretrained('declare-lab/nora')
    processor.tokenizer.padding_side = 'left'
    
    model = VLAWithExpert.from_pretrained('declare-lab/nora-1.5')
    fast_tokenizer = AutoProcessor.from_pretrained(
            "physical-intelligence/fast", trust_remote_code=True
        )
    accelerator.print("Pretrained weights loaded.")



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

    # Load and prepare dataset
    with accelerator.main_process_first():
        train_dataset,metadata = load_and_prepare_dataset(config, processor)
        
        stats = metadata.stats
       
        
        features = {
                    'action': PolicyFeature(shape=stats['action']['mean'].shape, type='action'),
                }
        norm_map = {
            'action': NormalizationMode.MIN_MAX,
        }

        ## If you do not want to train with delta action but with absolute end effector pose, please comment out  line 317 to 334
        try:
            if os.path.exists(config.data_root_dir):
                ## try to load from local dir
                with open(os.path.join(config.data_root_dir, 'norm_stats.json'),'r') as f:
                    new_stats = json.load(f)['norm_stats']
            else:
                ## load from hf hub 
                from huggingface_hub import hf_hub_download
                stats_file_path = hf_hub_download(repo_id=config.data_root_dir, filename='norm_stats.json', repo_type='dataset')
                with open(stats_file_path,'r') as f:
                    new_stats = json.load(f)['norm_stats']
                
        except:
            raise ValueError("Normalization stats norm_stats.json not found. Please run compute_norm_stats.py first.")
            
        stats['action']['min'] = np.array(new_stats['action']['q01'])
        stats['action']['max'] = np.array(new_stats['action']['q99'])
        normalizer = Normalize(features, norm_map, stats)

    # Create DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.per_device_batch_size,
        collate_fn=lambda examples: collate_fn(examples, processor,fast_tokenizer,normalizer),
        pin_memory=True,
        num_workers=32,
        shuffle=True,
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
    total_loss = 0.0
   
    while completed_steps < max_train_steps:
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                optimizer.zero_grad()

                actions = batch.pop('actions')
                
                
                   
                output = model(batch,actions,alpha=10.0) ## alpha is the scale between cross entropy loss and flow matching loss
               
    
                loss = output['combined_loss']
                accelerator.backward(loss)

                

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1
                    if config.gradient_clipping is not None:
                        accelerator.clip_grad_norm_(model.parameters(), config.gradient_clipping)

                    optimizer.step()
                    lr_scheduler.step()

                
                if completed_steps % config.logging_frequency == 0:
                    
                    
                    if accelerator.is_main_process:
                        
                        total_norm = 0.0
                        for p in model.parameters():
                            if p.grad is not None:
                                total_norm += p.grad.data.norm(2).item() ** 2
    
                        total_norm = total_norm**0.5
                        lr = lr_scheduler.get_last_lr()[0]
                       
                        logger.info(f"Step {completed_steps}, Loss: {loss.item()}, Grad Norm: {total_norm},Expert Loss:{output['expert_loss'].item()}, VLM Loss: {output['vlm_loss'].item()}")
                        
                        lr = lr_scheduler.get_last_lr()[0]
                        result = {
                            "train_combined_loss": loss.item(),
                            "expert_loss": output['expert_loss'].item(),
                            "vlm_loss": output['vlm_loss'].item(),
                            "grad_norm": total_norm,
                            "learning_rate": lr,
                        }
                        wandb.log(result, step=completed_steps)
                

            
            if completed_steps% config.checkpoint_save_frequency == 0 and completed_steps > 0:
                accelerator.save_state(os.path.join(config.output_dir, f"steps_{completed_steps}"))
            
            if completed_steps >= max_train_steps:
                break


    # Save final checkpoint

    accelerator.save_state(os.path.join(config.output_dir, f"steps_{completed_steps}"))
    
    wandb.finish()

def main():
    # Initialize training configuration
    config = TrainingConfig()

    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Run the training
    train(config)

if __name__ == "__main__":
    main()