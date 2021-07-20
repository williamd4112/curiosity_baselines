method="ppo"
env_name=${1:-"5RoomLargeExtInt-v0"}
run=${2:-"run_0"}

python3 launch.py \
  -alg ${method} \
  -curiosity_alg icm \
  -env ${env_name} \
  -lr 0.0001 \
  -entropy_loss_coeff 0.001 \
  -minibatches 8 \
  -curiosity_step_minibatches 8 \
  -feature_encoding idf_burda \
  -iterations 99968000 \
  -lstm \
  -num_envs 64 \
  -sample_mode gpu \
  -num_gpus 1 \
  -num_cpus 32 \
  -gpu_per_run 1 \
  -eval_envs 0 \
  -eval_max_steps 51000 \
  -eval_max_traj 50 \
  -timestep_limit 500 \
  -log_interval 10000 \
  -record_freq 25 \
  -pretrain None \
  -log_dir ./results/${method}_icm_${env_name}/${run} \
  -discount 0.99 \
  -v_loss_coeff 1.0 \
  -grad_norm_bound 1.0 \
  -gae_lambda 0.95 \
  -epochs 3 \
  -ratio_clip 0.1 \
  -launch_tmux no \
  -forward_loss_wt -1.0 \
  -log_heatmaps \
  -forward_model og \
  -batch_norm \
  -normalize_reward \
  -normalize_advantage \
  -normalize_obs \
  -feature_space inverse \
  -prediction_beta 1.0 \
  -obs_type rgb_full

#-grayscale
#-no_extrinsic