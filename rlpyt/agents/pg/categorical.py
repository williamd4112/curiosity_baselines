
import torch

from rlpyt.agents.base import AgentStep, AgentCuriosityStep, BaseAgent, RecurrentAgentMixin, AlternatingRecurrentAgentMixin
from rlpyt.agents.pg.base import AgentInfo, AgentInfoTwin, NdigoInfo, IcmInfo, RndInfo, AgentInfoRnn, AgentInfoRnnTwin
from rlpyt.agents.base import AgentInputs, AgentInputsRnn, IcmAgentCuriosityStepInputs, NdigoAgentCuriosityStepInputs, RndAgentCuriosityStepInputs
from rlpyt.distributions.categorical import Categorical, DistInfo
from rlpyt.utils.buffer import buffer_to, buffer_func, buffer_method
from rlpyt.utils.misc import iterate_mb_idxs

class CategoricalPgAgent(BaseAgent):
    """
    Agent for policy gradient algorithm using categorical action distribution.
    Same as ``GaussianPgAgent`` and related classes, except uses
    ``Categorical`` distribution, and has a different interface to the model
    (model here outputs discrete probabilities in place of means and log_stds,
    while both output the value estimate).
    """

    def __call__(self, observation, prev_action, prev_reward, dual=False):
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        pi, value = (self.model_int if dual else self.model)(*model_inputs)
        return buffer_to((DistInfo(prob=pi), value), device="cpu")

    def initialize(self, env_spaces, share_memory=False,
            global_B=1, env_ranks=None, **kwargs):
        super().initialize(env_spaces, share_memory,
            global_B=global_B, env_ranks=env_ranks, **kwargs)
        self.distribution = Categorical(dim=env_spaces.action.n)

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward), device=self.device)

         # TODO: need to decide which action to take
        pi, value = self.model(*model_inputs)
        int_pi, int_value = self.model_int(*model_inputs)

        dist_info = DistInfo(prob=pi)

        if self.dual_model:
            pi_int, pi_int = self.model_int(*model_inputs)
            dist_int_info = DistInfo(prob=pi_int)
            if self._mode == "eval":
                action = self.distribution.sample(dist_info)
            else:
                action = self.distribution.sample(dist_int_info)
        else:
            action = self.distribution.sample(dist_info)
        
        if self.dual_model:
            agent_info = AgentInfoTwin(dist_info=dist_info, value=value, 
                                dist_int_info=dist_int_info, int_value=int_value)
        else:
            agent_info = AgentInfo(dist_info=dist_info, value=value)

        action, agent_info = buffer_to((action, agent_info), device="cpu")
        return AgentStep(action=action, agent_info=agent_info)

    @torch.no_grad()
    def value(self, observation, prev_action, prev_reward, ret_int=False):
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        if ret_int:
            assert self.dual_model
            _pi, value = self.model_int(*model_inputs)
        else:
            _pi, value = self.model(*model_inputs)
        return value.to("cpu")


class RecurrentCategoricalPgAgentBase(BaseAgent):

    def __call__(self, observation, prev_action, prev_reward, init_rnn_state):
        # Assume init_rnn_state already shaped: [N,B,H]
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward, init_rnn_state), device=self.device)
        pi, value, next_rnn_state = self.model(*model_inputs)
        dist_info, value = buffer_to((DistInfo(prob=pi), value), device="cpu")
        return dist_info, value, next_rnn_state  # Leave rnn_state on device.

    def initialize(self, env_spaces, share_memory=False, global_B=1, obs_stats=None, env_ranks=None):
        super().initialize(env_spaces, share_memory, global_B=global_B, obs_stats=obs_stats, env_ranks=env_ranks)
        self.distribution = Categorical(dim=env_spaces.action.n)

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        prev_action = self.distribution.to_onehot(prev_action)
        agent_inputs = buffer_to((observation, prev_action, prev_reward), device=self.device)

        pi, value, rnn_state = self.model(*agent_inputs, self.prev_rnn_state)        
        dist_info = DistInfo(prob=pi)

        if self.dual_model:            
            int_pi, int_value, int_rnn_state = self.model_int(*agent_inputs, self.prev_int_rnn_state)
            dist_int_info = DistInfo(prob=int_pi)
            if self._mode == "eval":
                action = self.distribution.sample(dist_info)
            else:
                action = self.distribution.sample(dist_int_info)
        else:
            action = self.distribution.sample(dist_info)

        # Model handles None, but Buffer does not, make zeros if needed:
        prev_rnn_state = self.prev_rnn_state or buffer_func(rnn_state, torch.zeros_like)
        # Transpose the rnn_state from [N,B,H] --> [B,N,H] for storage.
        # (Special case: model should always leave B dimension in.)
        prev_rnn_state = buffer_method(prev_rnn_state, "transpose", 0, 1)

        if self.dual_model:                    
            prev_int_rnn_state = self.prev_int_rnn_state or buffer_func(int_rnn_state, torch.zeros_like)
            prev_int_rnn_state = buffer_method(prev_int_rnn_state, "transpose", 0, 1)            
            agent_info = AgentInfoRnnTwin(dist_info=dist_info, 
                                    value=value, 
                                    prev_rnn_state=prev_rnn_state,
                                    dist_int_info=dist_int_info, 
                                    int_value=int_value, 
                                    prev_int_rnn_state=prev_int_rnn_state)                                        
        else:
            agent_info = AgentInfoRnn(dist_info=dist_info, value=value, prev_rnn_state=prev_rnn_state)
        action, agent_info = buffer_to((action, agent_info), device="cpu")
        self.advance_rnn_state(rnn_state)  # Keep on device.
        if self.dual_model:
            self.advance_int_rnn_state(int_rnn_state)
        return AgentStep(action=action, agent_info=agent_info)

    @torch.no_grad()
    def curiosity_step(self, curiosity_type, *args):
        curiosity_model = self.model.module.curiosity_model if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model.curiosity_model
        curiosity_step_minibatches = self.model_kwargs['curiosity_step_kwargs']['curiosity_step_minibatches']
        T, B = args[0].shape[:2] # either observation or next_observation
        batch_size = B
        mb_size = batch_size // curiosity_step_minibatches

        if curiosity_type in {'icm', 'micm', 'disagreement'}:
            observation, next_observation, actions = args
            actions = self.distribution.to_onehot(actions)
            curiosity_agent_inputs = IcmAgentCuriosityStepInputs(
                observation=observation,
                next_observation=next_observation,
                actions=actions
            )
            curiosity_agent_inputs = buffer_to(curiosity_agent_inputs, device=self.device)
            agent_curiosity_info = IcmInfo()                
        elif curiosity_type == 'ndigo':
            observation, prev_actions, actions = args
            actions = self.distribution.to_onehot(actions)
            prev_actions = self.distribution.to_onehot(prev_actions)
            curiosity_agent_inputs = NdigoAgentCuriosityStepInputs(
                observations=observation,
                prev_actions=prev_actions,
                actions=actions
            )
            curiosity_agent_inputs = buffer_to(curiosity_agent_inputs, device=self.device)
            agent_curiosity_info = NdigoInfo(prev_gru_state=None)
        elif curiosity_type == 'rnd':
            next_observation, done = args
            curiosity_agent_inputs = RndAgentCuriosityStepInputs(
                next_observation=next_observation,
                done=done
            )
            curiosity_agent_inputs = buffer_to(curiosity_agent_inputs, device=self.device)
            agent_curiosity_info = RndInfo()

        # Need to split the intrinsic reward predictions to several minibatches -- otherwise, we will run out of GPU memory 
        r_ints = []
        for idxs in iterate_mb_idxs(batch_size, mb_size, shuffle=False):
            T_idxs = slice(None)
            B_idxs = idxs                    
            mb_r_int = curiosity_model.compute_bonus(*curiosity_agent_inputs[slice(None), B_idxs])
            r_ints.append(mb_r_int)
        r_int = torch.cat(r_ints, dim=1)

        r_int, agent_curiosity_info = buffer_to((r_int, agent_curiosity_info), device="cpu")

        return AgentCuriosityStep(r_int=r_int, agent_curiosity_info=agent_curiosity_info)

    def curiosity_loss(self, curiosity_type, *args):
        
        curiosity_model = self.model.module.curiosity_model if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model.curiosity_model
        if curiosity_type in {'icm', 'micm'}:
            observation, next_observation, actions, valid = args
            actions = self.distribution.to_onehot(actions)
            actions = actions.squeeze() # ([batch, 1, size]) -> ([batch, size])
            curiosity_agent_inputs = buffer_to((observation, next_observation, actions, valid), device=self.device)
            inv_loss, forward_loss = curiosity_model.compute_loss(*curiosity_agent_inputs)
            # inv_loss, forward_loss = curiosity_model.compute_loss(*args)
            losses = (inv_loss.to("cpu"), forward_loss.to("cpu"))
        elif curiosity_type == 'disagreement':
            observation, next_observation, actions, valid = args
            actions = self.distribution.to_onehot(actions)
            actions = actions.squeeze() # ([batch, 1, size]) -> ([batch, size])
            curiosity_agent_inputs = buffer_to((observation, next_observation, actions, valid), device=self.device)
            forward_loss = curiosity_model.compute_loss(*curiosity_agent_inputs)
            losses = (forward_loss.to("cpu"))
        elif curiosity_type == 'ndigo':
            observations, prev_actions, actions, valid = args
            actions = self.distribution.to_onehot(actions)
            prev_actions = self.distribution.to_onehot(prev_actions)
            actions = actions.squeeze() # ([batch, 1, size]) -> ([batch, size])
            prev_actions = prev_actions.squeeze() # ([batch, 1, size]) -> ([batch, size])
            curiosity_agent_inputs = buffer_to((observations, prev_actions, actions, valid), device=self.device)
            forward_loss = curiosity_model.compute_loss(*curiosity_agent_inputs)
            losses = (forward_loss.to("cpu"))
        elif curiosity_type == 'rnd':
            next_observation, valid = args
            curiosity_agent_inputs = buffer_to((next_observation, valid), device=self.device)
            forward_loss = curiosity_model.compute_loss(*curiosity_agent_inputs)
            losses = (forward_loss.to("cpu"))

        return losses

    @torch.no_grad()
    def value(self, observation, prev_action, prev_reward, ret_int=False):
        prev_action = self.distribution.to_onehot(prev_action)
        agent_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        # _pi, value, _rnn_state = self.model(*agent_inputs, self.prev_rnn_state)
        if ret_int:
            assert self.dual_model
            _pi, value, _rnn_state = self.model_int(*agent_inputs, self.prev_int_rnn_state)
        else:
            _pi, value, _rnn_state = self.model(*agent_inputs, self.prev_rnn_state)
        return value.to("cpu")


class RecurrentCategoricalPgAgent(RecurrentAgentMixin, RecurrentCategoricalPgAgentBase):
    pass


class AlternatingRecurrentCategoricalPgAgent(AlternatingRecurrentAgentMixin, RecurrentCategoricalPgAgentBase):
    pass


