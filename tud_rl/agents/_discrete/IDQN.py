import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import tud_rl.common.buffer as buffer
import tud_rl.common.nets as nets
from tud_rl.agents.base import BaseAgent
from tud_rl.agents._discrete.DQN import DQNAgent
from tud_rl.common.configparser import ConfigFile
from tud_rl.common.exploration import LinearDecayEpsilonGreedy


class IDQNAgent(BaseAgent):
    def __init__(self, c: ConfigFile, agent_name, init_DQN=True):
        super().__init__(c, agent_name)
        

        # attributes and hyperparameters
        self.N_agents         = getattr(c.Env, "env_kwargs")["N_TSs_max"] + 1
        self.lr         = c.lr
        self.dqn_weights    = c.dqn_weights
        self.is_multi         = True
        self.is_continuous    = False
        self.first_train      = True

        # checks
        assert not (self.mode == "test" and (self.dqn_weights is None)), "Need prior weights in test mode."

        if self.state_type == "image":
            raise NotImplementedError("Currently, image input is not supported for IDQN.")

        # linear epsilon schedule
        # self.exploration = LinearDecayEpsilonGreedy(eps_init        = self.eps_init, 
        #                                             eps_final       = self.eps_final,
        #                                             eps_decay_steps = self.eps_decay_steps)

    
        # init DQNs
        if init_DQN:
            if self.state_type == "feature":

                self.DQNs  = []
                for i in range(self.N_agents):
#                    c.dqn_weights = c.dqn_weights[:-13] + str(i) + c.dqn_weights[-12:]
                    self.DQNs.append(DQNAgent(c, agent_name))
                self.n_params = self.DQNs[0].n_params * self.N_agents
                
        

    @torch.no_grad()
    def select_action(self, s):
        """Selects action via actor network for a given state. Adds exploration bonus from noise and clips to action scale.
        Arg s:   np.array with shape (N_agents, state_shape)
        returns: np.array with shape (N_agents, num_actions)
        """        
        # if len(self.)
        action = np.zeros((self.N_agents,), dtype=np.int32)#[0] * self.N_agents
        for i in range(self.N_agents):
            a = self.DQNs[i].select_action(s[i])
            action[i] = a
        # import pdb
        # pdb.set_trace()
        # print(action)
        return action

    def _greedy_action(self, s):
        raise NotImplementedError

    def memorize(self, s, a, r, s2, d):
        """Stores current transition in replay buffer."""
        for idx in range(self.N_agents):
            self.DQNs[idx].memorize(s[idx], a[idx], r[idx], s2[idx], d)

    def _compute_target(self, r, s2, d, i):
        raise NotImplementedError

        with torch.no_grad():
            
            # Move tensors to the same device
            s2 = s2.to(self.device)

            # we need target actions from all agents
            target_a = torch.zeros((self.batch_size, self.N_agents, self.num_actions), dtype=torch.float32).to(self.device)
            for j in range(self.N_agents):
                s2_j = s2[:, j]

                if self.is_continuous:
                    target_a[:, j, :] = self.target_actor[j](s2_j)
                else:
                    target_a[:, j, :] = self._onehot(self.target_actor[j](s2_j))

            # next Q-estimate
            s2a2_for_Q = torch.cat([s2.reshape(self.batch_size, -1), target_a.reshape(self.batch_size, -1)], dim=1)
            Q_next = self.target_critic[i](s2a2_for_Q)

            # target
            y = r[:, i] + self.gamma * Q_next * (1 - d)
        return y

    def _compute_loss(self, Q, y, reduction="mean"):
        raise NotImplementedError
        for i in range(self.N_agents):
            self.DQNs[i]._compute_loss(Q[i], y[i], reduction=reduction)

    
    def train(self):
        """Samples from replay_buffer, updates actor, critic and their target networks."""
        if self.first_train:
            for i in range(self.N_agents):
                self.DQNs[i].logger = self.logger
            self.first_train = False
        for i in range(self.N_agents):
            self.DQNs[i].train()