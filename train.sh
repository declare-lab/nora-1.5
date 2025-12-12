#python utils/compute_norm_stats.py --dataset_path='hungchiayu/lerobot_multi_task_1104' --delta_transform
export WANDB_API_KEY='fa31aa7f2f594f38ca93dd9ed601470fc508202d'

accelerate launch --config_file='config.yaml' training/lerobot/train_lerobot.py