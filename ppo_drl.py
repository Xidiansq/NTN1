# -*-coding:utf-8-*-
import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import time
import ppo_core
import pandas as pd
import json
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
# 实时动态图
import matplotlib.pyplot as plt
import Satellite_run
from SINR_Calculate import *
import random
#from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

ax = []  # 定义一个 x 轴的空列表用来接收动态的数据
ay = []  # 定义一个 y 轴的空列表用来接收动态的数据
plt.ion()  # 开启一个画图的窗口


class PPOBuffer:
    def __init__(self, obs_dim, mask_dim,act_dim, size, gamma=0.99, lam=0.95):
        #self.max_req = max_req
        #每个进程中需要的交互次数
        self.obs_buf = np.zeros(ppo_core.combined_shape(size, obs_dim), dtype=np.float32)
        self.mask_buf = np.zeros(ppo_core.combined_shape(size, mask_dim), dtype=np.float32)
        self.act_buf = np.zeros(ppo_core.combined_shape(size, act_dim), dtype=np.int64)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.tti2ptr = {}
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store_pending(self, tti, obs, act, val, logp):
        self.tti2ptr[tti] = (obs, act, val, logp)

    def reward_throughput(self,txBytes):
        x1 = txBytes
        return x1
        

    # 设置奖励的公平性
    def reward_fairness(self, delay):
        total_delay = 0
        index =len(delay)
        index_delay = 0
        for i, value in enumerate(delay):
            total_delay+=value
            index_delay+= index*value
            index-=1
        if total_delay!=0:
            final_delay = index_delay/total_delay
        else:
            final_delay=0
        return final_delay

    # def get_reward(self, capacity, txBytes, delay):

    #     r_delay = self.reward_fairness(delay)#公平度
    #     #r2 = min(rbg_used, rbg_usable) / rbg_usable#已使用的资源块，如果等于1，代表资源利用率最大
    #     r = self.reward_throughput(capacity, txBytes)#一维数组
    #     return r, r_delay

    def get_reward(self, txBytes, delay):

        r_delay = self.reward_fairness(delay)#公平度
        #r2 = min(rbg_used, rbg_usable) / rbg_usable#已使用的资源块，如果等于1，代表资源利用率最大
        r = self.reward_throughput(txBytes)#一维数组
        return r, r_delay
    
    def execute_pop(self, tti, tti_reward):
        try:
            (obs, act, val, logp) = self.tti2ptr.pop(tti)  # 删除该tti，返回的是删除的tti的信息
            # ooo = np.array(obs['Requests'])
            # rbg_need = np.sum(ooo.reshape(self.max_req, -1)[:, 7])
            # if rbg_need == 0:
            #     return self.store(obs, act, val, logp, 0)
            # else:
            return self.store(obs, act, val, logp, tti_reward)
        except KeyError as e:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!KeyError", e)
            return False

    def store(self, obs, act, val, logp, reward):
        if self.ptr < self.max_size:  # buffer has to have room so you can store缓冲区未满
            self.obs_buf[self.ptr] = obs['ReqData']
            self.mask_buf[self.ptr]=obs['Req_list']
            # self.action_buf[self.ptr]=obs['action']
            # self.rbg_buf[self.ptr] = obs['RbgMap']
            # self.inv_buf[self.ptr] = obs['InvFlag']
            self.act_buf[self.ptr] = act
            self.val_buf[self.ptr] = val#价值函数
            self.logp_buf[self.ptr] = logp
            self.rew_buf[self.ptr] = reward
            self.ptr += 1#指针指向下一个地方
            #########################################
            # input()
            return True
        else:
            raise IndexError()
            return False

    #完成一个episode进行一次存储
    def finish_path(self, last_val=0, denorm=None):
        self.tti2ptr.clear()
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)#返回一个与 rews 数组等长的数组，每个元素是当前时刻到路径结束时的折扣奖励的累计值
        vals = np.append(self.val_buf[path_slice], last_val)
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = ppo_core.discount_cumsum(deltas, self.gamma * self.lam)
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = ppo_core.discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def __len__(self):
        return self.ptr

    def get(self):
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)#数据归一化
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, mask=self.mask_buf,
                    act=self.act_buf, ret=self.ret_buf, adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}

'''
ppo(statellite_run,
    actor_critic=core.RA_ActorCritic, ac_kwargs={"hidden_sizes": (256, 512, 1024, 512, 256)},
    steps_per_epoch=50, epochs=500, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
    vf_lr=1e-4, train_pi_iters=50, train_v_iters=50, lam=0.97, max_ep_len=50,
    logger_kwargs=logger_kwargs, use_cuda=True)
'''
#ac_kwargs 字典包含用于初始化 Actor-Critic 模型的参数设置
#steps_per_epoch  模型与环境交互的次数，更新次数
def ppo(env_fn, actor_critic=ppo_core.RA_ActorCritic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=50, epochs=10000, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=50,
        target_kl=0.01, logger_kwargs=dict(), save_freq=30, use_cuda=True):

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    seed += 10000 * random.randint(0,1000)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn.Env()
    obs_dim = np.prod(env.observation_space["ReqData"])
    mask_dim = np.prod(env.observation_space["Req_list"])
    #act_dim = env.action_space["beam_choice"][0]
    act_dim = env.beam_open


    # Create actor-critic module
    use_cuda = use_cuda and torch.cuda.is_available()
    ac_kwargs['use_cuda'] = use_cuda
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)

    # Sync params across processes
    sync_params(ac)

    # Count variables
    var_counts = tuple(ppo_core.count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())  #每个线程在一个训练周期需要执行的步骤   50除以
    buf = PPOBuffer(obs_dim, mask_dim, act_dim, local_steps_per_epoch, gamma, lam)

    def entropy(dist):
        #dist = torch.distributions.Categorical(logits=dist)
        min_real = torch.finfo(dist.logits.dtype).min
        logits = torch.clamp(dist.logits, min=min_real)
        p_log_p = logits * dist.probs
        return -p_log_p.sum(-1)
    
    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
        obs, deque, act, adv, logp_old = data['obs'], data['mask'], data['act'], data['adv'], data['logp']
        if use_cuda:
            obs, deque, act, adv, logp_old = obs.cuda(), deque.cuda(),  act.cuda(), adv.cuda(), logp_old.cuda()
        pi, logp = ac.pi(obs, deque,act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()
        approx_kl = (logp_old - logp).mean().item()
        ent = entropy(pi).mean().item()
        clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)
        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, deque, ret = data['obs'].float(), data['mask'].float(), data['ret'].float()
        if use_cuda:
            obs, deque, ret = obs.cuda(), deque.cuda(),ret.cuda()
        inp = torch.cat((obs, deque), dim=1)
        return ((ac.v(inp) - ret) ** 2).mean()

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    # Set up function for updating network parameters
    def update():
        data = buf.get()
        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = mpi_avg(pi_info['kl'])
            if kl > 50 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.' % i)
                break
            loss_pi.backward()
            mpi_avg_grads(ac.pi)
            pi_optimizer.step()

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(ac.v)
            vf_optimizer.step()
        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        logger.store(LossActor = pi_l_old, 
                     LossCritic = v_l_old, 
                     KL = kl, 
                     Entropy = ent, 
                     ClipFrac = cf,
                     DeltaLossA = (loss_pi.item() - pi_l_old), 
                     DeltaLossC = (loss_v.item() - v_l_old))

    #画图
    fig, ax = plt.subplots()
    x,y = [], []
    #初始化环境
    s0,o = env.reset()
    def get_env_info(extra):
        index = ['UserID', 'BsID', 'BsIfServ', 'Lat', 'Lon', 'Alt', 'Angle', 'Dis_Bs',
       'ReqID', 'QCI', 'NewData', 'Last_WaitData', 'Total_WaitData',
       'Finish_Data', 'Time_Delay', 'Down_TxData', 'Down_Throughput']
        data_sa = pd.DataFrame(extra["Sate_User"])
        data_bs = pd.DataFrame(extra["Bs_User"])
        columns_to_sum = ['NewData', 'Last_WaitData','Total_WaitData','Time_Delay','Down_TxData','Down_Throughput']  # 需要求和的列
        sum_result_sa = data_sa[columns_to_sum].sum()
        sum_result_bs = data_bs[columns_to_sum].sum()
        return sum_result_sa,sum_result_bs

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):#10000轮
        pbar = tqdm(total=local_steps_per_epoch)
        ep_tx, ep_capacity, ep_ret, ep_len, ep_newbytes ,ep_delay,ep_delay_normal,ep_throu= 0, 0, 0, 0, 0,0,0,0
        epoch_tx, epoch_waiting, epoch_reward, epoch_newbytes = 0, 0, 0, 0
        sum_tx = 0
        error = 0
        final_waiting = 0
        print(s0)
        #与环境进行交互
        while len(buf) < local_steps_per_epoch:
            #观测环境
            obs = torch.as_tensor(o["ReqData"], dtype=torch.float32)
            mask = torch.as_tensor(o["Req_list"])
            a, v, logp = ac.step(obs,mask)
            print("=======================动作a=================\n",a)
            print("action",a)
            o_info, next_o, extra,reward,done = env.step(a)
            print(o_info)
            #####################################获得日志信息######################################
            info_sa,info_bs=get_env_info(extra)
            # 当前的一个obs_tti
            #####################################奖励计算#########################################
             #只计算吞吐量的奖励
            tti_rdelay = 0 #! 暂时没有考虑时延
            threa_b = 1
            tti_reward = threa_b*reward + (1-threa_b)*tti_rdelay
            ###########################################################################################
            ep_tx += info_sa['Down_TxData']#每一个epoch的总传输量
            ep_newbytes += info_sa['NewData']
            final_waiting += info_sa['Total_WaitData']
            ep_ret += tti_reward
            ep_throu+=info_sa['Down_Throughput']
            ep_delay+=info_sa['Time_Delay']
            ep_delay_normal+=tti_rdelay
            ep_len += 1
            ##########################################################################################
            #a+1是为了将动作限制在【0，User_num+1】之间，为了能够让网络正常更新
            buf.store(o, a+1, v, logp, tti_reward)
            pbar.update(1)
            o = next_o
            timeout = ep_len == max_ep_len  # 一个episode
            terminal = timeout
            epoch_ended = len(buf) == local_steps_per_epoch  # 一个epoch
            # 缓存满了，触发更新
            if terminal or epoch_ended:
                if epoch_ended and not (terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)
                #if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    obs = torch.as_tensor(o["ReqData"], dtype=torch.float32)
                    req_list = torch.as_tensor(o["Req_list"], dtype=torch.int32)
                    _, v, _ = ac.step(obs,req_list)
                else:
                    v = 0
                buf.finish_path(v)  # 一个episode
                epoch_reward += ep_ret
                epoch_tx += ep_tx
                epoch_newbytes += ep_newbytes
                ep_tx, ep_capacity, ep_ret, ep_len, ep_newbytes,  = 0, 0, 0, 0, 0
                if epoch_ended:
                    # sum_tx = o_satis['total_txdata'].sum()
                    # final_waiting = o_satis['waitingdata'].sum()
                    ep_tx = epoch_tx / int(steps_per_epoch / max_ep_len)
                    #ep_capacity = epoch_capacity / int(steps_per_epoch / max_ep_len)
                    ep_ret = epoch_reward / int(steps_per_epoch / max_ep_len)
                    ep_waiting = epoch_waiting / int(steps_per_epoch / max_ep_len)
                    ep_newbytes = epoch_newbytes / int(steps_per_epoch / max_ep_len)
                    if len(y) >= 2:
                        if y[-1] - y[-2] >= 15:
                            error = 11111
                    logger.store(Ep_ret=ep_ret, Ep_tx=ep_tx, EP_new=ep_newbytes,
                                 EP_finalwiat=final_waiting,Ep_delay=ep_delay,Ep_delay_normal=ep_delay_normal,Ep_throu=ep_throu,
                                 sumtx=sum_tx, Ep_capacity=ep_capacity,
                                 Error=error)
                s0,o = env.reset()
    
        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_state({'env': env}, None)
        x.append(epoch + 1)
        y.append(epoch_reward)
        ax.cla()  # clear plot
        ax.plot(x, y, 'r', lw=1)  # draw line chart
        plt.pause(0.1)
        plt.savefig('./ppo-sate-bs.jpg')
        start_time = time.time()
        update()
        end_time = time.time()
        print("update time", end_time - start_time)
        logger.log_tabular('epoch       ', epoch)
        logger.log_tabular("ep_ret      ", ep_ret)
        logger.log_tabular("ep_tx       ", ep_tx)
        logger.log_tabular("newbytes    ", ep_newbytes)
        logger.log_tabular("ep_delay   ", ep_delay)
        logger.log_tabular("ep_delay_normal   ", ep_delay_normal)
        logger.log_tabular("ep_throu   ", ep_throu)
        logger.dump_tabular()
        # input()


if __name__ == '__main__':

    from spinup.utils.run_utils import setup_logger_kwargs
    import os

    trace_dir = os.getcwd() + "/result"
    logger_kwargs = setup_logger_kwargs("ppo-beam", data_dir=trace_dir, datestamp=True)#时间戳
    ppo(Satellite_run,
        actor_critic=ppo_core.RA_ActorCritic, ac_kwargs={"hidden_sizes": (256, 512,1024, 512, 256)},
        steps_per_epoch=50, epochs=1000, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-4, train_pi_iters=50, train_v_iters=50, lam=0.97, max_ep_len=50,
        logger_kwargs=logger_kwargs, use_cuda=True)
    # ppo(statellite_run,
    #     actor_critic=core_beam.RA_ActorCritic, ac_kwargs={"hidden_sizes": (256, 512, 1024, 512, 256)},
    #     steps_per_epoch=50, epochs=10000, gamma=0.99, clip_ratio=0.2, pi_lr=3e-5,
    #     vf_lr=1e-5, train_pi_iters=50, train_v_iters=50, lam=0.97, max_ep_len=50,
    #     logger_kwargs=logger_kwargs, use_cuda=True)
