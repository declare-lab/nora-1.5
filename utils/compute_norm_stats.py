"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import os
import numpy as np
import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata 
import normalize as normalize
from torch.utils.data import DataLoader
from huggingface_hub import HfApi
from typing import Sequence,runtime_checkable,Protocol
import dataclasses


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

        state, actions = data[REMAP_KEY['state']], data["action"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        #print(dims)
        actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["action"] = actions

        return data
    
transform = DeltaActions(mask=[True,True,True,True,True,True,False])

def main(lerobot_dataset_path, delta_transform=True):
    metadata = LeRobotDatasetMetadata(lerobot_dataset_path)
    
    #path = config.data_root_dir
    delta_timestamps = {
        "action": [t / metadata.fps for t in range(5)],
    }

    ds = LeRobotDataset(lerobot_dataset_path,delta_timestamps=delta_timestamps)
    keys = [ "action"]
    stats = {key: normalize.RunningStats() for key in keys}
    data_loader = DataLoader(
        ds,
        batch_size=32,
        num_workers=32,
        shuffle=True,
    )
    #num_batches = len(data_loader)//32
    
    for batch in tqdm.tqdm(data_loader, total=len(data_loader), desc="Computing stats"):
        
        for key in keys:
            if delta_transform:
                batch = transform(batch)

            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    #os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(lerobot_dataset_path):
        print(f"Writing stats to: {lerobot_dataset_path}")
        normalize.save(lerobot_dataset_path, norm_stats)
    else:
        normalize.save('./', norm_stats) ## save to current dir
        api = HfApi()
        #norm_stats = normalize.serialize_json(norm_stats)
        api.upload_file(
            path_or_fileobj='./norm_stats.json',
            path_in_repo='norm_stats.json', # This is the target file path in the repo
            repo_id=lerobot_dataset_path,
            repo_type="dataset", # Specify 'dataset' repo type
           # commit_message=f"Compute and upload normalization statistics for {stats_filename}",
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute normalization statistics for a LeRobot dataset.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the LeRobot dataset directory.")
    
    parser.add_argument("--delta_transform", action="store_true", help="Apply delta transformation to actions.")
    args = parser.parse_args()
    
    main(args.dataset_path, delta_transform=args.delta_transform)