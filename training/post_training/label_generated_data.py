import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from tqdm import tqdm

sys.path.insert(0, "vjepa2")

# Attempt imports, handle errors if local modules are missing
try:
    from app.vjepa_droid.utils import init_video_model, load_pretrained
    from app.vjepa_droid.transforms import make_transforms
except ImportError as e:
    print(f"Error importing local modules: {e}")
    print("Ensure you are running this from the root of your project or 'app.vjepa_droid' is accessible.")
    sys.exit(1)

# --- Constants ---
CROP_SIZE = 256
STATE_DIM = 7  # Dimension of the state vector (s)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokens_per_frame = 256
def parse_args():
    parser = argparse.ArgumentParser(description="Run V-JEPA rollout and energy calculation.")
    parser.add_argument(
        '--model_path', 
        type=str, 
        required=True, 
        help='Path to the .safetensors model checkpoint.'
    )
    parser.add_argument(
        '--device', 
        type=str, 
        default='cuda', 
        help='Device to run on (cuda or cpu).'
    )
    parser.add_argument(
        '--dataset_dir', 
        type=str, 
        default='data', 
        help='Path to the generated dataset'
    )

    return parser.parse_args()

def load_vjepa_model(checkpoint_path, dataset_dir,device):
   
    encoder, predictor = torch.hub.load("facebookresearch/vjepa2", "vjepa2_ac_vit_giant")
    
    print(f"Loading custom weights from {checkpoint_path}...")
    tensors = {}
    with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    
    # Load state dict (strict=False to allow partial loading if needed)
    missing, unexpected = predictor.load_state_dict(tensors, strict=False)
    if missing:
        print(f"Warning: Missing keys during load: {len(missing)}")
    
    encoder.to(device).eval()
    predictor.to(device).eval()
    
    return encoder, predictor

def get_transforms():
    """Returns the standard V-JEPA transforms."""
    return make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1., 1.),
        random_resize_scale=(1., 1.),
        reprob=0.,
        auto_augment=False,
        motion_shift=False,
        crop_size=CROP_SIZE,
    )
@torch.no_grad()
def forward_target(c,encoder, normalize_reps=True):
    B, C, T, H, W = c.size()
    c = c.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
    h = encoder(c)
    h = h.view(B, T, -1, h.size(-1)).flatten(1, 2)
    if normalize_reps:
        h = F.layer_norm(h, (h.size(-1),))
    return h



@torch.no_grad()
def rollout(predictor,z_hat, a, s_hat,steps=4,normalize_reps=True,device='cuda'):
    z_hat = z_hat.to(device)
    a = a.to(device)
    s_hat = s_hat.to(device)
    def _step_predictor(_z, _a, _s):
            _z = predictor(_z, _a, _s)
            if normalize_reps:
                _z = F.layer_norm(_z, (_z.size(-1),))
            return _z,torch.zeros(_z.shape[0],1,7,device=_z.device)

    #_z, _s = step_predictor(h2[:,:256,:],actions_tensor[:1,:1,:],torch.zeros(1,1,7))
    z_hat = z_hat.repeat(a.shape[0],1,1)
    s_hat = s_hat.repeat(a.shape[0],1,1)
    a_hat = a[:,:1,:] ## start from first step
    for i in range(1,steps+1):
        _z,_s = _step_predictor(z_hat, a_hat, s_hat)
    # _z_gt,_s = _step_predictor(z_hat_gt, a_hat_gt, s_hat)
    # z_hat_gt = torch.cat([z_hat_gt, _z_gt[:,-tokens_per_frame:]], dim=1)
        z_hat = torch.cat([z_hat, _z[:,-tokens_per_frame:]], dim=1)
        s_hat = torch.cat([s_hat, _s], dim=1)
        a_hat = torch.cat([a_hat, a[:,i:i+1,:]], dim=1)
    
    return z_hat

def main():
    args = parse_args()
    global DEVICE
    DEVICE = torch.device(args.device)

    # 1. Load Model
    encoder, predictor = load_vjepa_model(args.model_path, DEVICE)
    tokens_per_frame = int((CROP_SIZE // encoder.patch_size) ** 2)

    
    from datasets import load_from_disk
    dataset = load_from_disk(args.dataset_dir)
   
    transform = get_transforms()

    # 3. Process Specific Sample (e.g., index 40)
    
    energy_list = []
    for i in tqdm(range(len(dataset))):
        data_sample = dataset[i]
        
       
        curr_np = np.expand_dims(np.asarray(data_sample['current_frame']), 0) 
        goal_np = np.expand_dims(np.asarray(data_sample['goal_frame']), 0)
        
        # Apply transform and unsqueeze for Time dimension
        current_image = transform(curr_np).unsqueeze(0).to(DEVICE)
        goal_image = transform(goal_np).unsqueeze(0).to(DEVICE)

        # Get Latent Representations
        print("Encoding frames...")
        curr_latent = forward_target(current_image, encoder)
        goal_latent = forward_target(goal_image, encoder)


    
        actions_to_eval = torch.tensor(dataset[i]['generated_action']).to(torch.float32)
        
        
        s_init = torch.zeros(1, 1, STATE_DIM)

        # Run Rollout
        z_hat = rollout(
            predictor,
            curr_latent, 
            actions_to_eval, 
            s_init
        )
    
        # Calculate Energy
        # Calculate L1 distance between last predicted frame and goal latent
        # Expanding goal_latent to match batch size of z_hat
        B_act = z_hat.shape[0]
        goal_expanded = goal_latent.repeat(B_act, 1, 1)
        
        energy = torch.mean(
            torch.abs(z_hat[:, -tokens_per_frame:, :] - goal_expanded),
            dim=(1, 2)
        ).detach().cpu().numpy()
        energy_list.append(energy)

    
    dataset = dataset.add_column('goal_energy',energy_list)
    dataset.save_to_disk(args.dataset_dir)
        
        

if __name__ == "__main__":
    main()
