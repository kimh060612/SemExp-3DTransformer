CUDA_VISIBLE_DEVICES=0 python train.py -n 1 --auto_gpu_config=0 --num_global_steps=5 --num_mini_batch=3
CUDA_VISIBLE_DEVICES=0 python3 main.py --split val --eval 1 --load pretrained_models/sem_exp.pth --total_num_scenes 2
CUDA_VISIBLE_DEVICES=0 python3 main.py --total_num_scenes 2
CUDA_VISIBLE_DEVICES=0,1 python3 main.py --total_num_scenes=2 --num_training_frames=10000 --num_global_steps=25 --exp_name=exp2