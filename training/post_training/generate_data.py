import PIL.Image
from rlds_datasets import EpisodicRLDSDataset, RLDSBatchTransform, RLDSDataset
from pathlib import Path

from modelling_expert_with_dpo import VLAWithExpert
import torch
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Generate samples with VLA for")
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='./data', 
        help='Path to save the generated data'
    )
    parser.add_argument(
        '--data_root_dir', 
        type=str, 
        default='', 
        help='Path to the root directory of the RLDS dataset.'
    )
    parser.add_argument(
        '--n_samples', 
        type=int, 
        default=10000, 
        help='Number of samples to generate'
    )
    return parser.parse_args()

def main():
    args = parse_args()
    model = VLAWithExpert.from_pretrained("declare-lab/nora-1.5-libero") 
    model.to(torch.bfloat16)
    model.to('cuda')

    train_dataset = RLDSDataset(
            data_root_dir=Path(args.data_root_dir),
            data_mix="libero_combined_no_noops",
            batch_transform=RLDSBatchTransform(),
            resize_resolution=(256, 256),
            shuffle_buffer_size=256_000,
            train=True,
            image_aug=False
        )

    n_samples = args.n_samples


    current_frame = []
    goal_frame = []
    generated_action = []
    ground_truth_action = []
    instruction = []

    idx = 0 

    for data in train_dataset:

        
        current_frame.append(PIL.Image.fromarray(data['image']))
        goal_frame.append(PIL.Image.fromarray(data['goal_image']))
        instruction.append(data['lang'])
        ground_truth_action.append(data['action'])
        actions = []
        for _ in range(5): ## 5 samples per data point
            actions.append(model.sample_actions(data['image'],data['lang'])[0])

        generated_action.append(actions)
        idx+=1
        if idx>=n_samples:
            break
        print(f"Processed {idx} samples")

    from datasets import Dataset
    data_dict = {
        "current_frame": current_frame,
        "goal_frame": goal_frame, 
        "ground_truth_action": ground_truth_action, # Convert to list for the arrow writer
        "generated_action": generated_action,
        "instruction": instruction,
        
    }

    ds = Dataset.from_dict(data_dict)
    ds.save_to_disk(args.output_dir)
if __name__ == "__main__":
    main()
