
import numpy as np
import copy
import torch
from collections import namedtuple

from rlpyt.algos.base import RlAlgorithm
from rlpyt.algos.utils import discount_return, generalized_advantage_estimation, valid_from_done

# Convention: traj_info fields CamelCase, opt_info fields lowerCamelCase
OptInfo = namedtuple("OptInfo", ["return_",
                                 "intrinsic_rewards",
                                 "extint_ratio",
                                 "valpred",
                                 "advantage",
                                 "loss", 
                                 "pi_loss",
                                 "value_loss",
                                 "entropy_loss",
                                 "inv_loss", 
                                 "forward_loss",
                                 "reward_total_std", 
                                 "curiosity_loss",
                                 "entropy", 
                                 "perplexity"])
OptInfoTwin = namedtuple("OptInfoTwin", ["return_", "return_int_",
                                 "intrinsic_rewards",
                                 "extint_ratio",
                                 "valpred", "int_valpred",
                                 "advantage", "advantage_int",
                                 "loss", 
                                 "pi_loss", "int_pi_loss",
                                 "value_loss", "int_value_loss", 
                                 "entropy_loss", "int_entropy_loss",
                                 "inv_loss", 
                                 "forward_loss",
                                 "reward_total_std", "int_reward_total_std",
                                 "curiosity_loss",
                                 "entropy", "int_entropy",
                                 "perplexity", "int_perplexity"])
AgentTrain = namedtuple("AgentTrain", ["dist_info", "value"])


class PolicyGradientAlgo(RlAlgorithm):
    """
    Base policy gradient / actor-critic algorithm, which includes
    initialization procedure and processing of data samples to compute
    advantages.
    """

    bootstrap_value = True  # Tells the sampler it needs Value(State')
    opt_info_fields = tuple(f for f in OptInfo._fields)  # copy

    def initialize(self, agent, n_itr, batch_spec, mid_batch_reset=False,
            examples=None, world_size=1, rank=0):
        """
        Build the torch optimizer and store other input attributes. Params
        ``batch_spec`` and ``examples`` are unused.
        """
        self.optimizer = self.OptimCls(agent.parameters(),
            lr=self.learning_rate, **self.optim_kwargs)
        if self.initial_optim_state_dict is not None:
            self.optimizer.load_state_dict(self.initial_optim_state_dict)
        self.agent = agent
        self.n_itr = n_itr
        self.batch_spec = batch_spec
        self.mid_batch_reset = mid_batch_reset

    def process_returns(self, samples):
        """
        Compute bootstrapped returns and advantages from a minibatch of
        samples.  Uses either discounted returns (if ``self.gae_lambda==1``)
        or generalized advantage estimation.  Mask out invalid samples
        according to ``mid_batch_reset`` or for recurrent agent.  Optionally,
        normalize advantages.
        """        
        if self.agent.dual_model:
            reward, done, value, bv, int_value, int_bv = (samples.env.reward, samples.env.done, 
                                samples.agent.agent_info.value, samples.agent.bootstrap_value, 
                                samples.agent.agent_info.int_value, samples.agent.int_bootstrap_value)
        else:
            reward, done, value, bv = (samples.env.reward, samples.env.done, samples.agent.agent_info.value, samples.agent.bootstrap_value)        
        done = done.type(reward.dtype)

        if self.curiosity_type in {'icm', 'disagreement', 'micm'}:
            intrinsic_rewards, _ = self.agent.curiosity_step(self.curiosity_type, samples.env.observation.clone(), samples.env.next_observation.clone(), samples.agent.action.clone())
            intrinsic_rewards_logging = intrinsic_rewards.clone().data.numpy()
            self.intrinsic_rewards = intrinsic_rewards_logging
            self.extint_ratio = reward.clone().data.numpy()/(intrinsic_rewards_logging+1e-15)
            if self.agent.dual_model:
                int_reward = intrinsic_rewards
            else:
                reward += intrinsic_rewards
        elif self.curiosity_type == 'ndigo':
            intrinsic_rewards, _ = self.agent.curiosity_step(self.curiosity_type, samples.env.observation.clone(), samples.agent.prev_action.clone(), samples.agent.action.clone()) # no grad
            intrinsic_rewards_logging = intrinsic_rewards.clone().data.numpy()
            self.intrinsic_rewards = intrinsic_rewards_logging
            self.extint_ratio = reward.clone().data.numpy()/(intrinsic_rewards_logging+1e-15)
            if self.agent.dual_model:
                int_reward = intrinsic_rewards
            else:
                reward += intrinsic_rewards
        elif self.curiosity_type == 'rnd':
            intrinsic_rewards, _ = self.agent.curiosity_step(self.curiosity_type, samples.env.next_observation.clone(), done.clone())
            intrinsic_rewards_logging = intrinsic_rewards.clone().data.numpy()
            self.intrinsic_rewards = intrinsic_rewards_logging
            self.extint_ratio = reward.clone().data.numpy()/(intrinsic_rewards_logging+1e-15)
            if self.agent.dual_model:
                int_reward = intrinsic_rewards
            else:
                reward += intrinsic_rewards

        if self.normalize_reward:
            rews = np.array([])
            for rew in reward.clone().detach().data.numpy():
                rews = np.concatenate((rews, self.reward_ff.update(rew)))
            self.reward_rms.update_from_moments(np.mean(rews), np.var(rews), len(rews))
            reward = reward / np.sqrt(self.reward_rms.var)

            if self.agent.dual_model:
                int_rews = np.array([])
                for int_rew in int_reward.clone().detach().data.numpy():
                    int_rews = np.concatenate((int_rews, self.int_reward_ff.update(int_rew)))
                self.int_reward_rms.update_from_moments(np.mean(int_rews), np.var(int_rews), len(int_rews))
                int_reward = int_reward / np.sqrt(self.int_reward_rms.var)

        if self.gae_lambda == 1:  # GAE reduces to empirical discounted.
            return_ = discount_return(reward, done, bv, self.discount)
            advantage = return_ - value
            if self.agent.dual_model:
                int_return_ = discount_return(int_reward, done, bv, self.discount)
                int_advantage = int_return_ - value
        else:
            advantage, return_ = generalized_advantage_estimation(reward, value, done, bv, self.discount, self.gae_lambda)
            if self.agent.dual_model:
                int_advantage, int_return_ = generalized_advantage_estimation(int_reward, value, done, bv, self.discount, self.gae_lambda)
        
        if not self.mid_batch_reset or self.agent.recurrent:
            valid = valid_from_done(done)  # Recurrent: no reset during training.
        else:
            valid = None  # OR torch.ones_like(done)

        if self.normalize_advantage:
            if valid is not None:
                valid_mask = valid > 0
                adv_mean = advantage[valid_mask].mean()
                adv_std = advantage[valid_mask].std()
            else:
                adv_mean = advantage.mean()
                adv_std = advantage.std()
            advantage[:] = (advantage - adv_mean) / max(adv_std, 1e-6)

        if self.agent.dual_model:
            return return_, advantage, valid, int_return_, int_advantage
        else:
            return return_, advantage, valid
