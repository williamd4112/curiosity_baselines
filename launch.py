import os
import sys
import json
import argparse
from six.moves import shlex_quote
import torch

# Runners
from rlpyt.runners.minibatch_rl import MinibatchRl, MinibatchRlEval

# Policies
from rlpyt.agents.pg.atari import AtariFfAgent, AtariLstmAgent
from rlpyt.agents.pg.mujoco import MujocoFfAgent, MujocoLstmAgent

# Samplers
from rlpyt.samplers.parallel.cpu.collectors import CpuResetCollector, CpuWaitResetCollector, CpuEvalCollector
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler

# Environments
from rlpyt.envs.atari.atari_env import AtariEnv, AtariTrajInfo
from rlpyt.envs.gym import make as gym_make
from rlpyt.envs.gym import mario_make

# Learning Algorithms
from rlpyt.algos.pg.ppo import PPO
from rlpyt.algos.pg.a2c import A2C

# Utils
from rlpyt.utils.logging.context import logger_context
from rlpyt.utils.launching.affinity import encode_affinity, affinity_from_code
from rlpyt.utils.launching.arguments import get_args
from rlpyt.utils.misc import wrap_print


with open('./global.json') as global_params:
    params = json.load(global_params)
    _WORK_DIR = params['container_workdir']
    _RESULTS_DIR = params['container_resultsdir']
    _TB_PORT = params['tb_port']
    _ATARI_ENVS = params['envs']['atari_envs']
    _MUJOCO_ENVS = params['envs']['mujoco_envs']


def launch_tmux(args):

    name = '_'.join([args.alg, args.env])

    if args.pretrain is not None:
        log_dir = os.path.join(_RESULTS_DIR, args.pretrain)
    else:
        if os.path.isdir(f'{_RESULTS_DIR}/{name}/run_0'):
            runs = os.listdir(f'{_RESULTS_DIR}/{name}')
            try:
                runs.remove('.DS_Store')
            except ValueError:
                pass
            sorted_runs = sorted(runs, key=lambda run: int(run.split('_')[-1]))
            run_id = int(sorted_runs[-1].split('_')[-1]) + 1
        else:
            run_id = 0
        log_dir = os.path.join(_RESULTS_DIR, name, f'run_{run_id}')

    # fill in arguments for launch
    args_string = ''
    for arg, value in vars(args).items():
        if arg == 'launch_tmux':
            args_string += '-launch_tmux no '
        elif value is None and arg == 'log_dir':
            args_string += f'-log_dir {log_dir} '
        elif value is True:
            args_string += f'-{arg} '
        elif value is False:
            pass
        else:
            args_string += f'-{arg} {value} '

    # check whether to run
    print('\n')
    print('#'*50)
    print('Generated command:')
    print('-'*50)
    print(f'python3 launch.py {args_string}')
    print('#'*50)
    print('\n')

    wrap_print('Run this command? (y/n)')
    answer = input()

    # run commands if decided
    if answer == 'y':
        commands = [['htop', 'htop'],
                    ['runner', f'python3 launch.py {args_string}'],
                    ['tb', f'tensorboard --logdir {log_dir} --port {_TB_PORT} --bind_all']]
        os.system(f'kill -9 $( lsof -i:{_TB_PORT} -t ) > /dev/null 2>&1')
        os.system('tmux kill-session -t experiment')
        os.system('tmux new-session -s experiment -n htop -d bash')
        for i, cmd in enumerate(commands):
            if i != 0:
                os.system(f'tmux new-window -t experiment:{i+1} -n {cmd[0]} bash')
            os.system(f'tmux send-keys -t experiment:{cmd[0]} {shlex_quote(cmd[1])} Enter')
    else:
        print('Not running commands, exiting.')


def start_experiment(args):

    config = dict(env_id=args.env)
    affinity = {'master_cpus': [0], 'workers_cpus': list(range(1, args.num_cpus)), 'master_torch_threads': 1, 'worker_torch_threads': 1, 'alternating': False, 'set_affinity': True}

    # potentially reload models
    initial_optim_state_dict = None
    initial_model_state_dict = None
    if args.pretrain != 'None':
        os.system(f"find {args.log_dir} -name '*.json' -delete") # clean up json files for video recorder
        checkpoint = torch.load(os.path.join(_RESULTS_DIR, args.pretrain, 'params.pkl'))
        initial_optim_state_dict = checkpoint['optimizer_state_dict']
        initial_model_state_dict = checkpoint['agent_state_dict']

    # ----------------------------------------------------- LEARNING ALG ----------------------------------------------------- #
    if args.alg == 'ppo':
        algo = PPO(
                discount=args.discount,
                learning_rate=args.lr,
                value_loss_coeff=args.v_loss_coeff,
                entropy_loss_coeff=args.entropy_loss_coeff,
                OptimCls=torch.optim.Adam,
                optim_kwargs=None,
                clip_grad_norm=args.grad_norm_bound,
                initial_optim_state_dict=initial_optim_state_dict, # is None is not reloading a checkpoint
                gae_lambda=args.gae_lambda,
                minibatches=args.minibatches, # if recurrent: batch_B needs to be at least equal, if not recurrent: batch_B*batch_T needs to be at least equal to this
                epochs=args.epochs,
                ratio_clip=args.ratio_clip,
                linear_lr_schedule=args.linear_lr,
                normalize_advantage=args.normalize_advantage
                )
    elif args.alg == 'a2c':
        algo = A2C(
                discount=args.discount,
                learning_rate=args.lr,
                value_loss_coeff=args.v_loss_coeff,
                entropy_loss_coeff=args.entropy_loss_coeff,
                OptimCls=torch.optim.Adam,
                optim_kwargs=None,
                clip_grad_norm=args.grad_norm_bound,
                initial_optim_state_dict=initial_optim_state_dict,
                gae_lambda=args.gae_lambda,
                normalize_advantage=args.normalize_advantage
                )

    # ----------------------------------------------------- POLICY ----------------------------------------------------- #
    model_args = dict(curiosity_kwargs=dict(curiosity_alg=args.curiosity_alg))
    if args.curiosity_alg =='icm':
        model_args['curiosity_kwargs']['feature_encoding'] = args.feature_encoding
        model_args['curiosity_kwargs']['batch_norm'] = args.batch_norm

    if args.env in _MUJOCO_ENVS:
        if args.lstm:
            agent = MujocoLstmAgent(initial_model_state_dict=initial_model_state_dict)
        else:
            agent = MujocoFfAgent(initial_model_state_dict=initial_model_state_dict)
    else:
        if args.lstm:
            agent = AtariLstmAgent(
                        initial_model_state_dict=initial_model_state_dict,
                        model_kwargs=model_args
                        )
        else:
            agent = AtariFfAgent(initial_model_state_dict=initial_model_state_dict)

    # ----------------------------------------------------- SAMPLER ----------------------------------------------------- #

    # environment setup
    if 'mario' in args.env.lower():
        env_cl = mario_make
        env_args = dict(
            game=args.env,  
            no_extrinsic=args.no_extrinsic,
            no_negative_reward=args.no_negative_reward
            )
    elif args.env in _MUJOCO_ENVS:
        env_cl = gym_make
        env_args = dict(
            id=args.env, 
            no_extrinsic=args.no_extrinsic,
            no_negative_reward=args.no_negative_reward
            )
    else:
        env_cl = AtariEnv
        env_args = dict(
            game=args.env, 
            no_extrinsic=args.no_extrinsic,
            no_negative_reward=args.no_negative_reward
            )

    if args.lstm:
        collector_class = CpuWaitResetCollector
    else:
        collector_class = CpuResetCollector
    sampler = CpuSampler(
        EnvCls=env_cl,
        env_kwargs=env_args,
        eval_env_kwargs=env_args,
        batch_T=args.timestep_limit, # timesteps in a trajectory episode
        batch_B=args.num_envs, # environments distributed across workers
        max_decorrelation_steps=0,
        eval_n_envs=args.eval_envs,
        eval_max_steps=args.eval_max_steps,
        eval_max_trajectories=args.eval_max_traj,
        record_freq=args.record_freq,
        log_dir=args.log_dir,
        CollectorCls=collector_class
        )

    # ----------------------------------------------------- RUNNER ----------------------------------------------------- #
    if args.eval_envs > 0:
        runner = MinibatchRlEval(
            algo=algo,
            agent=agent,
            sampler=sampler,
            n_steps=args.iterations,
            affinity=affinity,
            log_interval_steps=args.log_interval,
            log_dir=args.log_dir
            )
    else:
        runner = MinibatchRl(
            algo=algo,
            agent=agent,
            sampler=sampler,
            n_steps=args.iterations,
            affinity=affinity,
            log_interval_steps=args.log_interval,
            log_dir=args.log_dir
            )

    with logger_context(args.log_dir, config, snapshot_mode="last", use_summary_writer=True):
        runner.train()


if __name__ == "__main__":

    args = get_args()
    if args.launch_tmux == 'yes':
        launch_tmux(args) # launches tmux
    else:
        start_experiment(args) # launches the actual experiment inside of a tmux session








