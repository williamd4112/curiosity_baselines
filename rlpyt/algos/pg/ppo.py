
from rlpyt.distributions.base import DistInfo
import numpy as np
import torch

from rlpyt.algos.pg.base import PolicyGradientAlgo, OptInfo, OptInfoTwin
from rlpyt.agents.base import AgentInputs, AgentInputsRnn, IcmAgentCuriosityInputs, NdigoAgentCuriosityInputs, RndAgentCuriosityInputs
from rlpyt.utils.tensor import valid_mean
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.buffer import buffer_to, buffer_method
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.misc import iterate_mb_idxs
from rlpyt.utils.averages import RunningMeanStd, RewardForwardFilter
from rlpyt.utils.grad_utils import plot_grad_flow

LossInputs = namedarraytuple("LossInputs", ["agent_inputs", "agent_curiosity_inputs", "action", "return_", "advantage", "valid", "old_dist_info"])
LossInputsTwin = namedarraytuple("LossInputsTwin", ["agent_inputs", "agent_curiosity_inputs", "action", 
            "return_", "return_int_", 
            "advantage", "advantage_int", 
            "valid", 
            "old_dist_info",
            "old_dist_int_info"])

class PPO(PolicyGradientAlgo):
    """
    Proximal Policy Optimization algorithm.  Trains the agent by taking
    multiple epochs of gradient steps on minibatches of the training data at
    each iteration, with advantages computed by generalized advantage
    estimation.  Uses clipped likelihood ratios in the policy loss.
    """

    def __init__(
            self,
            discount=0.99,
            learning_rate=0.001,
            value_loss_coeff=1.,
            entropy_loss_coeff=0.01,
            OptimCls=torch.optim.Adam,
            optim_kwargs=None,
            clip_grad_norm=1.,
            initial_optim_state_dict=None,
            gae_lambda=1,
            minibatches=4,
            epochs=4,
            ratio_clip=0.1,
            linear_lr_schedule=True,
            normalize_advantage=False,
            normalize_reward=False,
            curiosity_type='none',
            policy_loss_type='normal',
            ):
        """Saves input settings."""
        if optim_kwargs is None:
            optim_kwargs = dict()
        save__init__args(locals())
        self.policy_loss_type = policy_loss_type
        if self.policy_loss_type == 'dual':
            self.opt_info_fields = tuple(f for f in OptInfoTwin._fields)  # copy
        if self.normalize_reward:
            self.reward_ff = RewardForwardFilter(discount)
            self.reward_rms = RunningMeanStd()
            if self.policy_loss_type == 'dual':
                self.int_reward_ff = RewardForwardFilter(discount)
                self.int_reward_rms = RunningMeanStd()
        self.intrinsic_rewards = None
        self.extint_ratio = None        
        
    def initialize(self, *args, **kwargs):
        """
        Extends base ``initialize()`` to initialize learning rate schedule, if
        applicable.
        """
        super().initialize(*args, **kwargs)
        self._batch_size = self.batch_spec.size // self.minibatches  # For logging.
        if self.linear_lr_schedule:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=self.optimizer,
                lr_lambda=lambda itr: (self.n_itr - itr) / self.n_itr)  # Step once per itr.
            self._ratio_clip = self.ratio_clip  # Save base value.

    def optimize_agent(self, itr, samples):
        """
        Train the agent, for multiple epochs over minibatches taken from the
        input samples.  Organizes agent inputs from the training data, and
        moves them to device (e.g. GPU) up front, so that minibatches are
        formed within device, without further data transfer.
        """
        recurrent = self.agent.recurrent
        agent_inputs = AgentInputs(  # Move inputs to device once, index there.
            observation=samples.env.observation,
            prev_action=samples.agent.prev_action,
            prev_reward=samples.env.prev_reward,
        )
        agent_inputs = buffer_to(agent_inputs, device=self.agent.device)

        if hasattr(self.agent, "update_obs_rms"):
            self.agent.update_obs_rms(agent_inputs.observation)
        
        if self.agent.dual_model:
            return_, advantage, valid, return_int_, advantage_int = self.process_returns(samples)
        else:
            return_, advantage, valid = self.process_returns(samples)

        if self.curiosity_type in {'icm', 'micm', 'disagreement'}:
            agent_curiosity_inputs = IcmAgentCuriosityInputs(
                observation=samples.env.observation.clone(),
                next_observation=samples.env.next_observation.clone(),
                action=samples.agent.action.clone(),
                valid=valid
            )
            agent_curiosity_inputs = buffer_to(agent_curiosity_inputs, device=self.agent.device)
        elif self.curiosity_type == 'ndigo':
            agent_curiosity_inputs = NdigoAgentCuriosityInputs(
                observation=samples.env.observation.clone(),
                prev_actions=samples.agent.prev_action.clone(),
                actions=samples.agent.action.clone(),
                valid=valid
            )
            agent_curiosity_inputs = buffer_to(agent_curiosity_inputs, device=self.agent.device)
        elif self.curiosity_type == 'rnd':
            agent_curiosity_inputs = RndAgentCuriosityInputs(
                next_observation=samples.env.next_observation.clone(),
                valid=valid
            )
            agent_curiosity_inputs = buffer_to(agent_curiosity_inputs, device=self.agent.device)
        elif self.curiosity_type == 'none':
            agent_curiosity_inputs = None

        if self.policy_loss_type == 'dual':
            loss_inputs = LossInputsTwin(  # So can slice all.
                agent_inputs=agent_inputs,
                agent_curiosity_inputs=agent_curiosity_inputs,
                action=samples.agent.action,
                return_=return_,                
                advantage=advantage,                
                valid=valid,
                old_dist_info=samples.agent.agent_info.dist_info,                
                return_int_=return_int_,
                advantage_int=advantage_int,
                old_dist_int_info=samples.agent.agent_info.dist_int_info,
            )
        else:
            loss_inputs = LossInputs(  # So can slice all.
                agent_inputs=agent_inputs,
                agent_curiosity_inputs=agent_curiosity_inputs,
                action=samples.agent.action,
                return_=return_,
                advantage=advantage,
                valid=valid,
                old_dist_info=samples.agent.agent_info.dist_info,
            )

        if recurrent:
            # Leave in [B,N,H] for slicing to minibatches.
            init_rnn_state = samples.agent.agent_info.prev_rnn_state[0]  # T=0.
            if self.agent.dual_model:
                init_int_rnn_state = samples.agent.agent_info.prev_int_rnn_state[0]  # T=0.

        T, B = samples.env.reward.shape[:2]

        if self.policy_loss_type == 'dual':
            opt_info = OptInfoTwin(*([] for _ in range(len(OptInfoTwin._fields))))
        else:
            opt_info = OptInfo(*([] for _ in range(len(OptInfo._fields))))

        # If recurrent, use whole trajectories, only shuffle B; else shuffle all.
        batch_size = B if self.agent.recurrent else T * B
        mb_size = batch_size // self.minibatches

        for _ in range(self.epochs):
            for idxs in iterate_mb_idxs(batch_size, mb_size, shuffle=True):
                T_idxs = slice(None) if recurrent else idxs % T
                B_idxs = idxs if recurrent else idxs // T
                self.optimizer.zero_grad()
                rnn_state = init_rnn_state[B_idxs] if recurrent else None

                # NOTE: if not recurrent, will lose leading T dim, should be OK.
                if self.policy_loss_type == 'dual':
                    int_rnn_state = init_int_rnn_state[B_idxs] if recurrent else None
                    loss_inputs_batch = loss_inputs[T_idxs, B_idxs]
                    loss, pi_loss, value_loss, entropy_loss, entropy, perplexity, \
                        int_pi_loss, int_value_loss, int_entropy_loss, int_entropy, int_perplexity, \
                         curiosity_losses = self.loss(
                                    agent_inputs=loss_inputs_batch.agent_inputs,
                                    agent_curiosity_inputs=loss_inputs_batch.agent_curiosity_inputs,
                                    action=loss_inputs_batch.action,
                                    return_=loss_inputs_batch.return_,                
                                    advantage=loss_inputs_batch.advantage,                
                                    valid=loss_inputs_batch.valid,
                                    old_dist_info=loss_inputs_batch.old_dist_info,                
                                    return_int_=loss_inputs_batch.return_int_,
                                    advantage_int=loss_inputs_batch.advantage_int,
                                    old_dist_int_info=loss_inputs_batch.old_dist_int_info,
                                    init_rnn_state=rnn_state, init_int_rnn_state=int_rnn_state)
                else:
                    loss, pi_loss, value_loss, entropy_loss, entropy, perplexity, curiosity_losses = self.loss(*loss_inputs[T_idxs, B_idxs], rnn_state)

                loss.backward()
                count = 0
                grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.clip_grad_norm)
                self.optimizer.step()
                
                # Tensorboard summaries
                opt_info.loss.append(loss.item())
                opt_info.pi_loss.append(pi_loss.item())
                opt_info.value_loss.append(value_loss.item())
                opt_info.entropy_loss.append(entropy_loss.item())
                
                if self.policy_loss_type == 'dual':
                    opt_info.int_pi_loss.append(int_pi_loss.item())
                    opt_info.int_value_loss.append(int_value_loss.item())
                    opt_info.int_entropy_loss.append(int_entropy_loss.item())

                if self.curiosity_type in {'icm', 'micm'}:
                    inv_loss, forward_loss = curiosity_losses
                    opt_info.inv_loss.append(inv_loss.item())
                    opt_info.forward_loss.append(forward_loss.item())
                    opt_info.intrinsic_rewards.append(np.mean(self.intrinsic_rewards))
                    opt_info.extint_ratio.append(np.mean(self.extint_ratio))
                elif self.curiosity_type == 'disagreement':
                    forward_loss = curiosity_losses
                    opt_info.forward_loss.append(forward_loss.item())
                    opt_info.intrinsic_rewards.append(np.mean(self.intrinsic_rewards))
                    opt_info.extint_ratio.append(np.mean(self.extint_ratio))
                elif self.curiosity_type == 'ndigo':
                    forward_loss = curiosity_losses
                    opt_info.forward_loss.append(forward_loss.item())
                    opt_info.intrinsic_rewards.append(np.mean(self.intrinsic_rewards))
                    opt_info.extint_ratio.append(np.mean(self.extint_ratio))
                elif self.curiosity_type == 'rnd':
                    forward_loss = curiosity_losses
                    opt_info.forward_loss.append(forward_loss.item())
                    opt_info.intrinsic_rewards.append(np.mean(self.intrinsic_rewards))
                    opt_info.extint_ratio.append(np.mean(self.extint_ratio))

                if self.normalize_reward:
                    opt_info.reward_total_std.append(self.reward_rms.var**0.5)
                    if self.policy_loss_type == 'dual':
                        opt_info.int_reward_total_std.append(self.int_reward_rms.var**0.5)

                opt_info.entropy.append(entropy.item())
                opt_info.perplexity.append(perplexity.item())

                if self.policy_loss_type == 'dual':
                    opt_info.int_entropy.append(int_entropy.item())
                    opt_info.int_perplexity.append(int_perplexity.item())
                self.update_counter += 1

        opt_info.return_.append(torch.mean(return_.detach()).detach().clone().item())
        opt_info.advantage.append(torch.mean(advantage.detach()).detach().clone().item())
        opt_info.valpred.append(torch.mean(samples.agent.agent_info.value.detach()).detach().clone().item())

        if self.policy_loss_type == 'dual':
            opt_info.return_int_.append(torch.mean(return_int_.detach()).detach().clone().item())
            opt_info.advantage_int.append(torch.mean(advantage_int.detach()).detach().clone().item())
            opt_info.int_valpred.append(torch.mean(samples.agent.agent_info.int_value.detach()).detach().clone().item())

        if self.linear_lr_schedule:
            self.lr_scheduler.step()
            self.ratio_clip = self._ratio_clip * (self.n_itr - itr) / self.n_itr

        layer_info = dict() # empty dict to store model layer weights for tensorboard visualizations
        
        return opt_info, layer_info

    def loss(self, agent_inputs, agent_curiosity_inputs, action, return_, advantage, valid, old_dist_info, init_rnn_state=None,
            return_int_=None, advantage_int=None, old_dist_int_info=None, init_int_rnn_state=None):
        """
        Compute the training loss: policy_loss + value_loss + entropy_loss
        Policy loss: min(likelhood-ratio * advantage, clip(likelihood_ratio, 1-eps, 1+eps) * advantage)
        Value loss:  0.5 * (estimated_value - return) ^ 2
        Calls the agent to compute forward pass on training data, and uses
        the ``agent.distribution`` to compute likelihoods and entropies.  Valid
        for feedforward or recurrent agents.
        """
        if init_rnn_state is not None:
            # [B,N,H] --> [N,B,H] (for cudnn).
            init_rnn_state = buffer_method(init_rnn_state, "transpose", 0, 1)
            init_rnn_state = buffer_method(init_rnn_state, "contiguous")
            dist_info, value, _rnn_state = self.agent(*agent_inputs, init_rnn_state)
            if self.agent.dual_model and self.policy_loss_type == 'dual':
                init_int_rnn_state = buffer_method(init_int_rnn_state, "transpose", 0, 1)
                init_int_rnn_state = buffer_method(init_int_rnn_state, "contiguous")
                dist_int_info, int_value, _int_rnn_state = self.agent(*agent_inputs, init_int_rnn_state)
        else:            
            dist_info, value = self.agent(*agent_inputs) # uses __call__ instead of step() because rnn state is included here
            if self.agent.dual_model and self.policy_loss_type == 'dual':
                dist_int_info, int_value = self.agent(*agent_inputs, dual=True)
        dist = self.agent.distribution

        if self.policy_loss_type == 'dual':
            assert self.agent.dual_model
            joint_old_dist_info = type(old_dist_info)(prob=old_dist_info.prob * old_dist_int_info.prob)
            ratio = dist.likelihood_ratio(action, old_dist_info=joint_old_dist_info, new_dist_info=dist_info)
            ratio_int = dist.likelihood_ratio(action, old_dist_info=joint_old_dist_info, new_dist_info=dist_int_info)

            surr_1 = ratio * advantage
            surr_int_1 = ratio_int * advantage_int

            clipped_ratio = torch.clamp(ratio, 1. - self.ratio_clip, 1. + self.ratio_clip)
            clipped_ratio_int = torch.clamp(ratio_int, 1. - self.ratio_clip, 1. + self.ratio_clip)

            surr_2 = clipped_ratio * advantage
            surr_int_2 = clipped_ratio_int * advantage_int

            surrogate = torch.min(surr_1, surr_2)
            surrogate_int = torch.min(surr_int_1, surr_int_2)

            pi_loss = - valid_mean(surrogate, valid)
            int_pi_loss = - valid_mean(surrogate_int, valid)

            value_error = 0.5 * (value - return_) ** 2
            value_loss = self.value_loss_coeff * valid_mean(value_error, valid)

            int_value_error = 0.5 * (int_value - return_int_) ** 2
            int_value_loss = self.value_loss_coeff * valid_mean(int_value_error, valid)

            entropy = dist.mean_entropy(dist_info, valid)
            int_entropy = dist.mean_entropy(dist_int_info, valid)
            
            perplexity = dist.mean_perplexity(dist_info, valid)
            int_perplexity = dist.mean_perplexity(dist_int_info, valid)

            entropy_loss = - self.entropy_loss_coeff * entropy
            int_entropy_loss = - self.entropy_loss_coeff * int_entropy
            
            loss = (pi_loss + value_loss + entropy_loss) + (int_pi_loss + int_value_loss + int_entropy_loss)
        else:
            ratio = dist.likelihood_ratio(action, old_dist_info=old_dist_info, new_dist_info=dist_info)

            surr_1 = ratio * advantage
            clipped_ratio = torch.clamp(ratio, 1. - self.ratio_clip, 1. + self.ratio_clip)
            surr_2 = clipped_ratio * advantage
            surrogate = torch.min(surr_1, surr_2)
            pi_loss = - valid_mean(surrogate, valid)
            value_error = 0.5 * (value - return_) ** 2
            value_loss = self.value_loss_coeff * valid_mean(value_error, valid)

            entropy = dist.mean_entropy(dist_info, valid)
            perplexity = dist.mean_perplexity(dist_info, valid)
            entropy_loss = - self.entropy_loss_coeff * entropy

            loss = pi_loss + value_loss + entropy_loss

        if self.curiosity_type in {'icm', 'micm'}: 
            inv_loss, forward_loss = self.agent.curiosity_loss(self.curiosity_type, *agent_curiosity_inputs)           
            loss += inv_loss
            loss += forward_loss
            curiosity_losses = (inv_loss, forward_loss)
        elif self.curiosity_type == 'disagreement':
            forward_loss = self.agent.curiosity_loss(self.curiosity_type, *agent_curiosity_inputs)
            loss += forward_loss
            curiosity_losses = (forward_loss)
        elif self.curiosity_type == 'ndigo':
            forward_loss = self.agent.curiosity_loss(self.curiosity_type, *agent_curiosity_inputs)
            loss += forward_loss
            curiosity_losses = (forward_loss)
        elif self.curiosity_type == 'rnd':
            forward_loss = self.agent.curiosity_loss(self.curiosity_type, *agent_curiosity_inputs)
            loss += forward_loss
            curiosity_losses = (forward_loss)
        else:
            curiosity_losses = None

        if self.policy_loss_type == 'dual':
            return loss, pi_loss, value_loss, entropy_loss, entropy, perplexity, \
                int_pi_loss, int_value_loss, int_entropy_loss, int_entropy, int_perplexity, \
                curiosity_losses
        else:
            return loss, pi_loss, value_loss, entropy_loss, entropy, perplexity, curiosity_losses
