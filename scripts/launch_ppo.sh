method="ppo"
env_name=${1:-"5RoomLargeExtInt-v0"}
run=${2:-"run_0"}

python3 launch.py \
  -alg ${method} \
  -curiosity_alg none \
  -env ${env_name} \
  -lr 0.0001 \
  -entropy_loss_coeff 0.001 \
  -minibatches 8 \
  -curiosity_step_minibatches 4 \
  -feature_encoding none \
  -iterations 99968000 \
  -lstm \
  -num_envs 64 \
  -sample_mode gpu \
  -num_gpus 1 \
  -num_cpus 8 \
  -gpu_per_run 1 \
  -eval_envs 0 \
  -eval_max_steps 51000 \
  -eval_max_traj 50 \
  -timestep_limit 500 \
  -log_interval 10000 \
  -record_freq 25 \
  -pretrain None \
  -log_dir ./results/${method}_${env_name}/${run} \
  -discount 0.99 \
  -v_loss_coeff 1.0 \
  -grad_norm_bound 1.0 \
  -gae_lambda 0.95 \
  -epochs 3 \
  -ratio_clip 0.1 \
  -launch_tmux no \
  -log_heatmaps \
  -normalize_reward \
  -normalize_advantage \
  -normalize_obs \
  -obs_type rgb_full

#-grayscale
#-no_extrinsic