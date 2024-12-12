# -*-coding:utf-8-*-
from builtins import print
import math
import numpy as np
import random
import scipy.signal
#from gym.spaces import Box, Discrete, Tuple, Dict
import torch
import Parameters
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])



def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):
    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError
    
    def forward(self, obs, req_list ,act=None):
        pi = self._distribution(obs,req_list)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MultiCategoricalActor(Actor):
    def __init__(self, observation, action_dim, hidden_sizes, activation):
        super().__init__()
        self.max_req = observation["ReqData"][0]
        #obs_dim = np.prod(observation["ReqData"])  #user_num * userinfo
        obs_dim = np.sum([np.prod(v) for k, v in observation.items()])
        out_dim = int(action_dim/Parameters.beam_open)
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [out_dim], activation).to(device)
        self.user_num = Parameters.user_number
        self.beam_open = Parameters.beam_open
    
    def _distribution(self, obs,req_list):
        """
        波束动作决策网络
        """
        # batch_size = 1 if len(obs.shape) == 1 else obs.shape[0]
        # mask = req_list==0
        # logits = self.logits_net(obs)
        # logits.reshape(mask.shape)
        # logits = logits.masked_fill_(mask, float('-inf'))
        # return logits
        batch_size = 1 if len(obs.shape) == 1 else obs.shape[0]
        inp = torch.cat((obs,req_list), 0 if len(obs.shape) == 1 else 1)
        logits = self.logits_net(inp)
        mask = ~(req_list.bool()).view(batch_size,self.user_num)
        mask_temp = torch.zeros((batch_size,1),device=device).bool()
        mask = torch.cat((mask_temp,mask),1)
        # print(f"logits  reshape之前：{logits}")
        logits = logits.reshape(batch_size,self.user_num+1)  # 
        logits = logits.masked_fill_(mask, -np.inf)

        # print(f"logits  reshape之后：{logits}")
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi,act):#logits, actions):
        """
        计算采样动作的对数概率
        """
        # probs = F.softmax(logits, dim=-1)  # [batch_size, num_beams]
        # log_probs = torch.log(probs + 1e-8)  # 添加一个小值防止 log(0)
        # # actions 的形状为 [batch_size, max_num_actions]
        # action_masks = actions != -1  # [batch_size, max_num_actions]

        # # 将填充值 -1 替换为 0，避免索引错误
        # actions = actions.clamp(min=0)
        # actions  = torch.tensor(actions,dtype=int)
        # # 取出选中动作的对数概率
        # selected_log_probs = torch.gather(log_probs, 1, actions)  # [batch_size, max_num_actions]

        # # 对无效的位置（填充值）置零
        # selected_log_probs = selected_log_probs * action_masks.float()

        # # 对每个样本的对数概率求和
        # logp_as = selected_log_probs.sum(dim=1)  # [batch_size]
        # return logp_as
        if len(act.shape) == 2:  # 两个维度，第一个维度为batch_size，第二个维度为每个动作的维数
            lp = pi.log_prob(act)
            # print('lp_forward',lp)
            # print('lp_forward.sum(lp, 1)',torch.sum(lp, 1))
            return torch.sum(lp, 1)  # 按照行为单位相加
        else:
            return torch.sum(pi.log_prob(act))
        # input()
        if len(act.shape) == 2:  # 两个维度，第一个维度为batch_size，第二个维度为每个动作的维数
            # lp = pi.log_prob(act)
            # return torch.sum(lp, 1)  # 按照行为单位相加
            logp = pi.log_prob(act.squeeze(-1))
            return logp
        else:
            return torch.sum(pi.log_prob(act))

    # def _log_prob_from_distribution(self, pi, act):
    #     if len(act.shape) == 2:  # 两个维度，第一个维度为batch_size，第二个维度为每个动作的维数
    #         logp = pi.log_prob(act)
    #         return torch.sum(logp, 1)  # 按照行为单位相加
    #     else:
    #         return torch.sum(pi.log_prob(act))


class MLPCritic(nn.Module):

    def __init__(self, observation_space, hidden_sizes, activation):
        super().__init__()
        obs_dim = np.sum([np.prod(v) for k, v in observation_space.items()])
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        v = self.v_net(obs.float())
        #print("状态的v值：", v)
        return torch.squeeze(v, -1)  # Critical to ensure v has right shape.


class RA_ActorCritic(nn.Module):
    def __init__(self, observation_space, action_space,
                 hidden_sizes=(256, 512,1024, 512, 256), activation=nn.Tanh, use_cuda=True):
        # print("action_space",action_space)

        super().__init__()
        action_dim = action_space["action_num"]
        self.pi = MultiCategoricalActor(observation_space, action_dim, hidden_sizes, activation)
        self.v = MLPCritic(observation_space, hidden_sizes, activation)
        self.use_cuda = use_cuda

        if use_cuda:
            self.pi = self.pi.cuda()
            self.v = self.v.cuda()

    def step(self, obs, req_list):
        if self.use_cuda:
            obs = obs.cuda()
            req_list = req_list.cuda()

        # 检查并调整 obs 和 req_list 的维度
        # if obs.dim() == 1:
        #     obs = obs.unsqueeze(0)  # [1, obs_dim]
        # if req_list.dim() == 1:
        #     req_list = req_list.unsqueeze(0)  # [1, num_beams]

        with torch.no_grad():
            pi = self.pi._distribution(obs, req_list)
            # print("概率值：",pi.probs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            # input()
            inp = torch.cat((obs, req_list), 0 if len(obs.shape) == 1 else 1)
            v = self.v(inp)

            

        if self.use_cuda:
            #a-1是为了将动作限制在【-1，User_num】之间
            return (a-1).cpu().flatten().numpy(), v.cpu().numpy(), logp_a.cpu().flatten().numpy()
        else:
            return (a-1).flatten().numpy(), v.numpy(), logp_a.flatten().numpy()
            

    def act(self, obs,req_list):
        return self.step(obs,req_list)[0]
