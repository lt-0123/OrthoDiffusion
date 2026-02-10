'''
torchrun --nproc_per_node=1 train.py \
    --pose_id 1 \
    --results_folder "results_pose_1" > log.log
'''
from torchvision.transforms import RandomCrop, Compose, ToPILImage, Resize, ToTensor, Lambda
from diffusion_model.trainer import GaussianDiffusion, Trainer
from diffusion_model.unet import create_model
from dataset import NiftiImageGenerator, NiftiPairImageGenerator, NiftiImagePoseGenerator
import argparse
import torch
import os 
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--targetfolder', type=str, default="dataset/knee")
parser.add_argument('--input_size', type=int, default=256)
parser.add_argument('--depth_size', type=int, default=16)
parser.add_argument('--num_channels', type=int, default=64)
parser.add_argument('--num_res_blocks', type=int, default=1)
parser.add_argument('--num_class_labels', type=int, default=3)
parser.add_argument('--train_lr', type=float, default=1e-5)
parser.add_argument('--batchsize', type=int, default=10)
parser.add_argument('--epochs', type=int, default=21500) # epochs parameter specifies the number of training iterations
parser.add_argument('--timesteps', type=int, default=1000)
parser.add_argument('--save_and_sample_every', type=int, default=500)
parser.add_argument('-r', '--resume_weight', type=str, default=None)
parser.add_argument('--pose_csv_path', type=str, default="dataset/pretrain_pose_1.csv")
parser.add_argument('--pose_id', type=int, default=1)
parser.add_argument('--results_folder', type=str, default="results")

args = parser.parse_args()
local_rank = 0
if "LOCAL_RANK" in os.environ:
    local_rank = setup_distributed()

targetfolder = args.targetfolder
pose_csv_path = args.pose_csv_path
input_size = args.input_size
depth_size = args.depth_size
num_channels = args.num_channels
num_res_blocks = args.num_res_blocks
num_class_labels = args.num_class_labels
save_and_sample_every = args.save_and_sample_every
resume_weight = args.resume_weight
train_lr = args.train_lr
pose_id = args.pose_id
batchsize = args.batchsize
# b, c, d, h, w
# value range: [-1, 1]

transform = Compose([
    Lambda(lambda t: torch.tensor(t).float()),                            # (H, W, D)
    Lambda(lambda t: (t - t.min()) / (t.max() - t.min() + 1e-8)),         # normalize
    Lambda(lambda t: (t * 2) - 1),                                         # scale to [-1, 1]
    Lambda(lambda t: t.permute(2, 0, 1)),                                 # (D, H, W)
    Lambda(lambda t: t.unsqueeze(0))                                      # (1, D, H, W)
])


dataset = NiftiImagePoseGenerator(
    targetfolder,
    pose_csv_path,
    input_size=input_size,
    depth_size=depth_size,
    transform=transform
)

sampler = DistributedSampler(dataset) if dist.is_initialized() else None
dataloader = DataLoader(
    dataset,
    batch_size=batchsize,
    shuffle=(sampler is None),
    num_workers=4,
    pin_memory=True,
    sampler=sampler
)
print(f"[Rank {local_rank}] Dataset size: {len(dataset)}")

in_channels = num_class_labels if with_condition else 1
out_channels = 1


model = create_model(input_size, num_channels, num_res_blocks, in_channels=in_channels, out_channels=out_channels).to(local_rank)

diffusion = GaussianDiffusion(
    model,
    image_size = input_size,
    depth_size = depth_size,
    timesteps = args.timesteps,   # number of steps
    loss_type = 'l2',    # L1 or L2
    with_condition=with_condition,
    channels=out_channels
).to(local_rank)

if dist.is_initialized():
    diffusion = DDP(diffusion, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

world_size = dist.get_world_size() if dist.is_initialized() else 1

scaled_lr = train_lr * world_size * batchsize
print(f"[Rank {local_rank}] Base LR={args.train_lr}, World Size={world_size}, Final LR={scaled_lr}")


trainer = Trainer(
    diffusion,
    dataloader,
    image_size = input_size,
    depth_size = depth_size,
    train_batch_size = batchsize,
    train_lr = scaled_lr,
    train_num_steps = args.epochs,         # total training steps
    gradient_accumulate_every = 1,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    fp16 = False,#True,                       # turn on mixed precision training with apex
    save_and_sample_every = save_and_sample_every,
    results_folder = args.results_folder
)

if resume_weight is not None:
    trainer.load(resume_weight)
    print(f"Resumed from {resume_weight} at step {trainer.step}")


trainer.train()


