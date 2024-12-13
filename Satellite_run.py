from collections import deque
import itertools
import numpy as np
import pandas as pd
import User_Init 
import Satellite_Bs
import User_Update
import Parameters
import matplotlib.pyplot as plt
import ppo_core
from SINR_Calculate import *
from collections import defaultdict
import matplotlib
matplotlib.use('tkagg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ppo_reward
import Tool_Calculate
#先维度再经度

class Env:

    def __init__(self):
        #卫星参数
        # self.beam, self.lat_log = setInitBeamCenterPos(0, [0, 0, 0], type='IRIDIUM')#波束中心的xyz坐标集合
        self.sate_lla = Parameters.sate_lla #[36.0735, 99.0898, 1.13468e+06] #卫星的经纬度坐标
        self.sate_xyz = Parameters.sate_xyz #[-958408, 5.99098e+06, 4.41951e+06] #卫星的笛卡尔坐标
        self.earth_radius = Parameters.R_earth #地球半径
        self.earthspheroidtype = {'SPHERE': 0, 'GRS80': 1, 'WGS84': 2}  # 三种地球模型
        self.center_latitude = Parameters.center_latitude   #36.0735    中心维度
        self.center_longitude = Parameters.center_longitude #99.0898    中心经度
        self.center_sat = [self.center_latitude, self.center_longitude, 0]
        self.cover_range = Parameters.cover_range # 500 * 1000     覆盖半径500km
        self.beam_open = Parameters.beam_open #6

        #基站参数
        self.bs_num = Parameters.bs_num
        self.bs_lla, self.bs_xyz,self.bs_ridth = Satellite_Bs.setBS(self.center_longitude,self.center_latitude, self.cover_range,self.bs_num)
        self.BS_INT_TH = Parameters.BS_INT_TH #基站的干扰阈值dBW
        self.bs_list = list(range(0,len(self.bs_lla), 1)) #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

        #用户参数
        self.user_number = Parameters.user_number
        self.user_info_len = Parameters.user_info_len
        self.ontime = Parameters.ontime
        self.offtime = Parameters.offtime

        #强化学习所需参数
        self.observation_space = {'ReqData': (self.user_number, self.user_info_len),
                                  'Req_list':(self.user_number)}
        #self.action_space={'beam_choice':(self.user_number,1)}
        #self.action_space=math.factorial(self.user_number) // (math.factorial(self.beam_open) * math.factorial(self.user_number - self.beam_open))
        self.action_space = {"action_num":((self.user_number+1)*self.beam_open)}
        #状态参数
        # self.beam_list = list(range(0, len(self.beam), 1))
        self.userlist = 0
        self.request_list = 0
        self.tti=0
        #16*7--16*3
        self.maxqueue = 20
        
        self.extra_infor = {}
        self.last_tti_state = 0
        # self.onedB = np.asarray([0.4814, 0.4562, 0.4675, 0.4799, 0.4921, 0.4068, 0.5146, 0.4669, 0.4242, 0.5031, 0.5241, 0.4426])
        self.center = 0
        self.center_xyz_list=np.array([0, 0, 0], dtype="float64")
        self.newdata=0
        self.beam_queue = []
        self.move=1
        self.ue_xyz, self.ue_lla = self.generate_position(self.center_sat, self.cover_range)
    def reset(self):
        """
        初始化环境信息：
        用户位置，用户请求信息，确定卫星、基站等位置关系
        """
        #初始化覆盖范围内的用户，获取用户的列表
        self.userlist = User_Init.initial_userAll(self.center_sat, self.cover_range, self.bs_xyz, self.user_number, \
                                        self.sate_xyz, self.sate_lla, self.ontime, self.offtime, self.ue_xyz, self.ue_lla)
        #获取初始化后的用户的任务传输情况
        self.last_tti_task = User_Init.initial_userTask_info(self.userlist)
        #避免初始化所有的用户都没有请求信息，采用while循环
        while True:
            S0, requeset_list = User_Init.initial_userTraffic_info(self.userlist)
            if len(requeset_list) !=[]:
                break
        self.ReqData = S0.iloc[:, 0 : self.user_info_len].to_numpy()
        self.mask_req = self.build_mask(self.user_number, requeset_list)
        S0_to_PPO = {"ReqData" : self.ReqData.flatten(),"Req_list" : self.mask_req}
        self.last_tti_state = S0
        self.request_list = requeset_list
        return S0, S0_to_PPO


    def step(self, action_beam,epoch=0,action_bs=0):
        """
        根据动作信息，进行信息更新
        动作为卫星从 user_number 个用户中选择进行 beam_num个用户进行通信(做一个掩码，只选择有请求信息的用户)
        """
        
        last_request_list = self.request_list
        Action_beam = self.action_beam_reshaping(action_beam) if len(action_beam)==Parameters.beam_open else action_beam
        Action_bs   = action_bs #TODO 这里需要重写方法
        self.userlist,DOWN_Rate,MAX_DOWN_Rate = User_Update.update_all_user(self.userlist, self.center_sat, self.cover_range, self.bs_xyz,
                                                    self.last_tti_state,self.last_tti_task,last_request_list, 
                                                    Action_beam,Action_bs,self.ontime, self.offtime,self.bs_lla)
        #更新所有的用户请求信息，采用while循环，避免都没有产生新的用户
        while True:
            S_next, next_request_list = User_Update.update_user_traffic_info(self.userlist)
            if len(next_request_list) !=[]:
                break
        #获取需要计算的奖励信息
        self.extra_infor = self.generate_extra_info(next_request_list, S_next, Action_beam,)
        #print(S_next)
        # print("查看一下",self.extra_infor)
        self.ReqData = S_next.iloc[:, 0: self.user_info_len].to_numpy()
        self.mask_req = self.build_mask(self.user_number, next_request_list)
        S_Next_to_PPO = {'ReqData': self.ReqData.flatten(),"Req_list" : self.mask_req}
        self.last_tti_state = S_next
        self.request_list = next_request_list
        rrr,r_sa,r_bs,reward_array = ppo_reward.get_paper_reward_info(self.extra_infor,MAX_DOWN_Rate/1000) #mbps 
        ##########排除重复动作******************************
        counts = np.bincount(action_beam+1)
        rrr = 0 if np.any(counts[1::,] > 1)  else rrr
        # reward_array= [0]*(Parameters.user_number+1) if np.any(counts[1::,] > 1)  else reward_array
        ####################### ############################
        if epoch>20:
            Tool_Calculate.plot_user_position(S_next["Lat"],S_next["Lon"],S_next["BsIfServ"],DOWN_Rate,MAX_DOWN_Rate,
                                          self.bs_lla,self.bs_ridth,epoch)
        done = False
        return S_next, S_Next_to_PPO, self.extra_infor,rrr,r_sa,r_bs,reward_array,done

    def build_mask(self,user_num, req_list):
        """
        根据用户请求信息，构造掩码矩阵
        """
        mask_req = np.zeros(user_num,dtype=int)
        mask_req[req_list] = 1
        return mask_req
    

    def action_beam_reshaping(self, action):
        """
        [ 3 17 14 11 -1 -1]
        动作重塑, 根据ppo产生的动作, 确定新的动作
        """
        Action_beam = np.zeros(self.user_number)
        for i in action:
            if(i != -1):
                Action_beam[i] = 1
        return Action_beam
    def action_bs_reshaping(self, action):
        """
        [ 3 17 14 11 -1 -1]
        动作重塑, 根据ppo产生的动作, 确定新的动作
        """
        Action_beam = np.zeros(self.user_number)
        for i in action:
            if(i != -1):
                Action_beam[i] = 1
        return Action_beam
    
    def generate_position(self, center, maxdistance):
        """
        位置类函数1
        初始化用户位置, 方法: 以region_center为用户区域中心, 随机在maxdistance范围内产生经纬度坐标;
        输入参数: 用户区域中心位置, [lat, lon, alt]; 用户区域半径, maxdistance;
        返回: 用户位置, [x, y, z]和[lat, lon, alt];
        """
        ue_xyz_list = []
        ue_lla_list = []
        for _ in range(Parameters.user_number):
            originlatitude, originlongitude, maxaltitude = center[0], center[1], center[2]
            # 除去南北极
            if originlatitude >= 90:
                originlatitude = 89.999
            elif originlatitude <= -90:
                originlatitude = -89.999
            if maxaltitude < 0:
                maxaltitude = 0
            originlatituderadians = originlatitude * (np.pi / 180)
            originlongituderadians = originlongitude * (np.pi / 180)
            origincolatitude = (np.pi / 2) - originlatituderadians
            a = 0.99 * maxdistance / self.earth_radius    # 圆心角弧度数的最大值
            if a > np.pi:
                a = np.pi
            d = np.random.uniform(0, self.earth_radius - self.earth_radius * np.cos(a))
            phi = np.random.uniform(0, np.pi * 2)
            alpha = m.acos((self.earth_radius - d) / self.earth_radius)
            theta = np.pi / 2 - alpha
            randpointlatitude = m.asin(m.sin(theta) * m.cos(origincolatitude) + m.cos(theta) * m.sin(origincolatitude) * m.sin(phi))
            intermedlong = m.asin((m.sin(randpointlatitude) * m.cos(origincolatitude) - m.sin(theta)) / (m.cos(randpointlatitude) * m.sin(origincolatitude)))
            intermedlong = intermedlong + np.pi / 2
            if phi > (np.pi / 2) and phi <= ((3 * np.pi) / 2):
                intermedlong = -intermedlong
            randpointlongtude = intermedlong + originlongituderadians
            randaltitude = np.random.uniform(0, maxaltitude)
            ue_lat = randpointlatitude * (180 / np.pi)      # 用户坐标, 纬度
            ue_lon = randpointlongtude * (180 / np.pi)      # 用户坐标, 经度
            ue_alt = randaltitude                           # 用户坐标, 海拔
            ue_xyz = Tool_Calculate.GeographicToCartesianCoordinates(ue_lat, ue_lon, ue_alt, self.earthspheroidtype['GRS80'])   # 用户位置, [x, y, z]
            ue_lat_lon_alt = [ue_lat, ue_lon, ue_alt]                                                                             # 用户位置, [lat, lon, alt]
            center_xyz = Tool_Calculate.GeographicToCartesianCoordinates(center[0], center[1], center[2], self.earthspheroidtype['GRS80'])
            ue_xyz_list.append(ue_xyz)
            ue_lla_list.append(ue_lat_lon_alt)
        return ue_xyz_list, ue_lla_list
    def generate_extra_info(self, cur_request_list, cur_state, Action_Beam):
        """
        获取需要计算的奖励信息，划分基站用户和卫星用户
        卫星用户的奖励包括 Cn/Rn * (距离基站的距离)   地面用户的话包括  Cn/Rn
        """
        if len(cur_request_list) == 0:
            return self.extra_infor
        self.extra_infor={'Sate_User': cur_state[[bool(i) for i in Action_Beam]].to_dict(orient='records'),  # 获取指定用户的数据
                          'Bs_User': cur_state[[not bool(i) for i in Action_Beam]].to_dict(orient='records')}   # 获取剩余用户的数据}
        return self.extra_infor
    
if __name__ == '__main__':
    env = Env()

    on = 5
    off = 5
    np.random.seed(1)
    s0,s0_PPO=env.reset()
    print(s0)
    action1 = np.array( [4,5,6,2,13,16,38,39,40,41])
    print("action1",action1)

    a,b,c,d,e,f,g=env.step(action1,24)
    print("info",a)

    action2 = np.array( [33,22,21,46,39,4,48,9,49,42])
    print("action2",action2)

    a,b,c,d,e,f,g=env.step(action2,25)
    print("info",a)

    action3 = np.array( [11,13,14,16,18,19])
    print("action3",action3)
    a,b,c,d,e,f,g=env.step(action3,24)
    print("info",a)

    action4 = np.array( [11,13,14,16,18,19])
    print("action4",action4)
    a,b,c,d,e,f,g=env.step(action4,24)
 
