import os
import torch
import numpy as np

from envs import make_vec_envs
from arguments import get_args
from model import Semantic_Mapping_BA

os.environ["OMP_NUM_THREADS"] = "1"

args = get_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

def main():
    num_episodes = int(args.num_eval_episodes)
    args.device = torch.device("cuda:0" if args.cuda else "cpu")
    device = args.device
    num_scenes = args.num_processes
    ngc = 8 + args.num_sem_categories
    nc = args.num_sem_categories + 4  # num channels
    map_size = args.map_size_cm // args.map_resolution
    full_w, full_h = map_size, map_size # 480, 480
    local_w = int(full_w / args.global_downscaling) # 240
    local_h = int(full_h / args.global_downscaling) # 240
    local_pose = torch.zeros(num_scenes, 3).float().to(device)
    local_map = torch.zeros(num_scenes, nc, local_w, local_h).float().to(device)
    planner_pose_inputs = np.zeros((num_scenes, 7))
    full_map = torch.zeros(num_scenes, nc, full_w, full_h).float().to(device)
    full_pose = torch.zeros(num_scenes, 3).float().to(device)
    origins = np.zeros((num_scenes, 3))
    lmb = np.zeros((num_scenes, 4)).astype(int)
    intrinsic_rews = torch.zeros(num_scenes).to(device)
    torch.set_num_threads(1)
    envs = make_vec_envs(args)
    obs, infos = envs.reset()
    
    poses = torch.from_numpy(np.asarray([infos[env_idx]['sensor_pose'] for env_idx in range(num_scenes)])).float().to(device)
    sem_map_module = Semantic_Mapping_BA(args).to(device)
    sem_map_module.eval()
    _, local_map, _, local_pose = sem_map_module(obs, poses, local_map, local_pose)
    print(poses)
    print(obs.shape)
    print(local_map.shape)
    print(local_pose.shape)
    
    locs = local_pose.cpu().numpy()
    global_input = torch.zeros(num_scenes, ngc, local_w, local_h)
    global_orientation = torch.zeros(num_scenes, 1).long()
    
    for ep_num in range(num_episodes):
        for step in range(args.max_episode_length):
            action = torch.randint(0, 3, (args.num_processes,))
            obs, _, done, infos = envs.step(action)
            ## Test Logic
            _, c, h, w = obs.shape
            if not (c > 4 or h <= 120 or w <= 160):
                continue
            # poses = torch.from_numpy(np.asarray([infos[env_idx]['sensor_pose'] for env_idx in range(num_scenes)])).float().to(device)
            # print(poses)
            print(obs.shape)
            depth = obs[0, 3, :, :]
            rgb = obs[0, :3, :, :]
            print(rgb.shape)
            depth_img = depth.cpu().numpy()
            rgb_img = rgb.permute(1, 2, 0).cpu().numpy()
            print(rgb_img.shape)
            import cv2
            cv2.imwrite("./depth_ex.png", depth_img)
            cv2.imwrite("./rgb_ex.png", rgb_img)
            
            if done:
                break

    print("Test successfully completed")

if __name__ == "__main__":
    main()
