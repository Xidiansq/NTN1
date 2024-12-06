# -*-coding:utf-8-*-
import math
import numpy as np
import math as m
import Parameters
import Satellite_run
from scipy.special import jn
import Satellite_Bs
import Tool_Calculate


class downlink_transmision_tool:
    def __init__(self):
        #用户参数
        self.req_user_num = 0
        self.otheruser_conbeam_num = 0
        self.Gr_user =  Parameters.Gain_user-16  #  30 dBi

        #通信参数
        self.bw = Parameters.bw          #带宽 500e6 HZ
        self.frequency = Parameters.frequency #中心频率2e10   # Hz
        self.velocity = 3e8
        self.gama = 0.5
        self.noisy = 2.5118864315095823e-12
        self.Beam_Power = []            #卫星波束分配情况(w)
        #卫星参数
        self.sate_lla = Parameters.sate_lla
        self.sate_xyz = Parameters.sate_xyz
        self.Hleo = Parameters.HLeo           #550000   # m
        self.Power_SateTotal = Parameters.Power_SateTotal   #30dBW
        self.Power_Beam_average = self.Power_SateTotal-10*np.log10(Parameters.beam_open)
        self.Power_BeamMax = Parameters.Power_BeamMax     #0.8*total
        self.Gain_beam = Parameters.Gain_Beam            #42dBi
        self.sate_threedB = Parameters.sate_threedB      #1dB
        # self.path_loss = -209.53778  # dBw
        # self.PowerT_beam = 10  #dBw   #3W
        
        #基站参数
        self.BS_TNT_TH = Parameters.BS_INT_TH   #干扰阈值
        self.Power_bs = Parameters.Power_bs
        self.Gain_bs = Parameters.Gain_bs
        self.front_to_back_ratio = Parameters.front_to_back_ratio  #前后比
        self.R_earth = Parameters.R_earth
        self.bs_threedB=Parameters.bs_threedB #3dB角
        self.antenna_num=Parameters.antenna_num #基站天线数量
        #最终输出
        self.sinr_matrix = np.zeros(Parameters.user_number)
        self.max_sinr_matrix = np.zeros(Parameters.user_number)
        self.Power_Allcation_Sign = Parameters.Power_Allcation_Sign #是否进行功率控制
        self.Allcation_by_SINR = True
        self.interference_bs2sa = np.zeros((Parameters.bs_num,Parameters.user_number))#基站用户对卫星用户的干扰
        self.factor = np.zeros(Parameters.bs_num)+1 #基站的功率因数
        self.antenna_req_num = np.zeros((Parameters.user_number))+1
    def get_sa_loss_path(self, req_user_info, req_list):
        """
        计算路径损耗：根据用户和卫星的距离计算
        传入参数：有请求用户的所有信息(req_user_info), 有请求用户的id列表(req_list)
        输入: 每个有请求用户的路径损耗
        """
        self.req_user_num = len(req_user_info)
        LOSS_PATH = np.zeros((1, self.req_user_num))
        for i in range(self.req_user_num):
            index_n = i
            #获取当前用户的位置（维度、经度、海拔）
            userN_position = np.radians(np.array([req_user_info.at[index_n, 'Lat'], req_user_info.at[index_n, 'Lon'], req_user_info.at[index_n, 'Alt']]))
            distance_userN2sate = np.sqrt(self.R_earth**2 + (self.R_earth + self.Hleo)**2 - 
                        2 * self.R_earth * (self.R_earth + self.Hleo) * 
                        (np.sin(userN_position[0]) * np.sin(np.radians(self.sate_lla[0])) + 
                         np.cos(userN_position[0]) * np.cos(np.radians(self.sate_lla[0])) * 
                         np.cos(userN_position[1] - np.radians(self.sate_lla[1]))))
            loss_path = ((4 * m.pi * self.frequency * distance_userN2sate) / self.velocity) ** (-2)    # W
            Loss_path = 10 * np.log10(loss_path)   # dBw
            LOSS_PATH[0][i] = Loss_path
        return LOSS_PATH

    def get_sa_gain(self, req_user_info, req_list):
        """
        计算增益矩阵
        传入参数：有请求用户的所有信息(req_user_info), 有请求用户的id列表(req_list)
        输入: 用户的增益矩阵
        """
        self.req_user_num = len(req_user_info)#获取有请求用户的数量
        self.otheruser_conbeam_num = self.req_user_num#同样对他可以产生干扰的用户也是有请求的
        theta_matrix = np.zeros((self.req_user_num, self.otheruser_conbeam_num))#构造一个角度矩阵，表示每个用户之间的角度
        Gain_matrix = np.full((self.req_user_num, self.otheruser_conbeam_num), float(self.Gain_beam))#构造一个增益矩阵，表示信道增益
        # print("初始值",Gain_matrix)
        # input()
        # distance_matrix = np.zeros((self.req_user_num, self.otheruser_conbeam_num))#构造一个距离矩阵，计算每个用户之间的距离
        for i in range(self.req_user_num):
            index_k = i #记录当前用户的id
            #获取当前用户的位置信息
            userK_position = np.radians(np.array([req_user_info.at[index_k, 'Lat'], req_user_info.at[index_k, 'Lon'], req_user_info.at[index_k, 'Alt']]))
            for j in range(self.otheruser_conbeam_num):#内层遍历其他用户
                index_n = j #记录当前用户id
                if(index_n == index_k):
                    continue
                else:
                
                    angle_K2N = np.radians(Satellite_Bs.angle_between_users(np.array([req_user_info.at[index_k, 'Lat'], req_user_info.at[index_k, 'Lon'], req_user_info.at[index_k, 'Alt']]),
                                                                 np.array([req_user_info.at[index_n, 'Lat'], req_user_info.at[index_n, 'Lon'], req_user_info.at[index_n, 'Alt']]),
                                                                 Parameters.sate_lla)
                    )
                    
                    u = 2.01723 * np.sin(angle_K2N) / np.sin(np.radians(self.sate_threedB))
                    #计算贝塞尔函数值
                    j1_u = jn(1, u)
                    j3_u = jn(3, u)
                    Gain_K2N = self.Gain_beam * ( (j1_u / (2 * u)) + (36 * j3_u / (u**3)) )**2
                    theta_matrix[i][j] = np.degrees(angle_K2N)
                    Gain_matrix[i][j] = Gain_K2N #dBi
                    # Gain_matrix[i][j]= self.Gain_beam - ((12 * (10 ** (self.Gain_beam / 10))) / self.gama) * np.square(theta_matrix[i][j] / (70 * np.pi))#
        Gain_matrix = 10 ** (Gain_matrix / 10)#转化为功率值
        return Gain_matrix

    def get_sa_sinr(self, action, req_user_info, req_list):
        """
        获得卫星用户的信干噪比
        输入：
        action:波束分配策略
        req_user_info:用户信息

        """
        # print("下行传输动作", action)
        
        #action 代表选择了哪些用户,将action转化为一个一维矩阵，亮的地方是1，不亮的是0

        Gain_matrix = self.get_sa_gain(req_user_info, req_list)#获取增益矩阵
        Path_loss_matrxi = self.get_sa_loss_path(req_user_info, req_list)#计算每个用户的路径损耗
        self.req_user_num = len(req_user_info)
        Gain_self = 10 * np.log10(Gain_matrix/10) #dB
        selectted_user = np.where(action == 1)[0] #用户id（从小到大）
        h_sa = 10 ** ((Gain_self + self.Gr_user + Path_loss_matrxi[0]) /10)/self.noisy
        h_sa = h_sa[np.ix_(selectted_user, selectted_user)] #得到卫星用户的信道系数矩阵
        ##################进行功率分配###############################################
        self.Beam_Power=action.copy()
        if self.Power_Allcation_Sign:
            self.Beam_Power[self.Beam_Power==1] = Tool_Calculate.Power_Allocation(h_sa) 
        else:
            self.Beam_Power[self.Beam_Power==1] = 10**(self.Power_Beam_average/10)
        print("self.Beam_Power",self.Beam_Power)
        ##################################################################################
        for i in range(self.req_user_num):
            interference=0
            if(action[i] == 0): #i 就是代表这个请求用户会被服务，计算这个用户的sinr就行
                continue
            else:
                #首先这个用户会接受来自自己的增益
                Gain_self = 10 * np.log10(Gain_matrix[i][i]) #dBi
                power_self = 10 ** ((Gain_self + self.Gr_user + Path_loss_matrxi[0][i]) /10) * self.Beam_Power[i] #W
                
                for j in range(self.req_user_num):
                    if i == j or action[j] == 0:
                        continue
                    else:
                        Gain_interf = 10 * np.log10(Gain_matrix[j][i]) #dBi
                        interf = 10 ** ((Gain_interf+self.Gr_user + Path_loss_matrxi[0][i]) /10) * self.Beam_Power[j]#  #其他波束干扰+bs干扰阈值
                        interference += interf
                interference += 10 ** (self.BS_TNT_TH / 10) #最后加上来自基站的干扰
                #interference = 10 ** ((self.Gr_user + Path_loss_matrxi[0][i]) / 10) * interference 
                sinr = power_self / (self.noisy + interference)
                max_sinr = power_self / (self.noisy + 10 ** (self.BS_TNT_TH / 10))
                self.sinr_matrix[i] = sinr
                self.max_sinr_matrix[i] = max_sinr
        return Gain_matrix,Path_loss_matrxi

    def get_bs_loss_path(self, req_user_info):
        """
        计算路径损耗：根据用户和基站的距离计算
        传入参数：有请求用户的所有信息(req_user_info), 有请求用户的id列表(req_list)
        输出: 每个有请求的基站用户的路径损耗
        输出：自由空间损耗(dbi)
        """
        LOSS_PATH = np.array([((4 * m.pi * self.frequency * req_user_info.at[user_id,"Dis_Bs"]) / self.velocity) ** (-2)
                              for user_id in range(len(req_user_info))])
        

        return 10*np.log10(LOSS_PATH)
    def get_antenna_peruser_gain(self,angle_antenna_user):
        """
        计算基站天线对基站用户的增益
        angle_antenna_user：天线与用户间的夹角
        G_BS_max：最大方向增益
        注意：
        1:这里没有考虑的是天线与用户有夹角
        """
        return 10**((self.Gain_bs/10)+(np.max([-0.6*(angle_antenna_user/self.bs_threedB)**2,-(self.front_to_back_ratio/10)])))
    def get_bs_diffraction_loss(self,req_user_info):
        """
        计算绕射损失，正在调研中
        """
        DIFFRACTION_LOSS=np.array([-1 for _ in range(len(req_user_info))])
        return DIFFRACTION_LOSS
    def get_bs_gain(self, req_user_info,bs_lla):
        """
        计算增益矩阵
        传入参数：有请求用户的所有信息(req_user_info),基站位置信息(bs_lla)
        输入: 用户的增益矩阵
        """
        self.req_user_num = len(req_user_info)#获取有请求用户的数量
        self.bs_num = len(bs_lla)#同样对他可以产生干扰的用户也是有请求的
        theta_matrix = np.zeros((self.req_user_num, self.req_user_num))#构造一个角度矩阵，表示每个用户之间关于基站的角度
        Gain_matrix = np.full((self.req_user_num, self.req_user_num), 0)#构造一个增益矩阵，表示信道增益
        # distance_matrix = np.zeros((self.req_user_num, self.otheruser_conbeam_num))#构造一个距离矩阵，计算每个用户之间的距离
        for this_user_id in range(self.req_user_num ):
            #获取当前用户的位置信息
            this_user_position=np.array([req_user_info.at[this_user_id, 'Lat'], req_user_info.at[this_user_id, 'Lon'], req_user_info.at[this_user_id, 'Alt']])
            bs_id=int(req_user_info.at[this_user_id,'BsID'])
            bsM_position = np.array([bs_lla[bs_id][1],bs_lla[bs_id][2],bs_lla[bs_id][3]])
            for other_user_id in range(self.req_user_num ):  # 遍历用户
                if req_user_info.at[other_user_id,'BsID']==bs_id:
                    other_user_position = np.array([req_user_info.at[other_user_id, 'Lat'], req_user_info.at[other_user_id, 'Lon'], req_user_info.at[other_user_id, 'Alt']])
                    # 计算目标用户和其他用户关于基站之间的夹角
                    angle = Satellite_Bs.angle_between_users(this_user_position,other_user_position,bsM_position)
                    theta_matrix[this_user_id][other_user_id]=angle
                    # 计算增益
                    gain = self.get_antenna_peruser_gain(angle)
                    Gain_matrix[this_user_id][other_user_id] = gain
        return Gain_matrix

    def get_bs_sinr(self, req_user_info,bs_lla,bs_state,Gain_sa_matrix,Path_loss_sa):
        """
        获得基站服务用户的信干噪比
        输入：
        action:波束分配策略
        req_user_info:用户信息
        bs_lla:基站位置信息
        bs_state:基站状态
        Gain_sa_matrix:卫星用户的增益（用来计算干扰）
        Path_loss_sa:卫星用户的损失(用来计算路径损失)

        """
        # print("下行传输动作", action)
        # action 代表基站选择了哪些用户,将action转化为一个一维矩阵，亮的地方是1，不亮的是0

        Gain_matrix = self.get_bs_gain(req_user_info,bs_lla)#获取增益矩阵
        Path_loss_matrxi = self.get_bs_loss_path(req_user_info)#计算每个用户的路径损耗
        Diffraction_loss_matrxi = self.get_bs_diffraction_loss(req_user_info)
        self.req_user_num = len(req_user_info)
        for bs_id,bs_id_state in enumerate(bs_state):          #遍历每个基站                  
            sinr_bs_temp = np.zeros(self.req_user_num)                                  #此基站范围内的用户信干噪比情况
            user_sa=bs_id_state["user_sa"]                      #当前小区的卫星用户
            user_bs=bs_id_state["user_bs"] if  bs_id_state["choose_by_SINR"] == False else  bs_id_state["user_bs_req"]      #当前小区的基站服务用户
            antenna_req_num = len(user_bs) if bs_id_state["choose_by_SINR"] == False else min(self.antenna_num,len(user_bs))
            
            for user_bs_id in user_bs:
                self.antenna_req_num[user_bs_id] = antenna_req_num
                interference=0
                #基站给此用户的增益
                Gain_self = 10 * np.log10(Gain_matrix[user_bs_id][user_bs_id]) #dBi
                
                power_self = 10 ** ((Gain_self + 
                                    self.Gr_user +       
                                    Path_loss_matrxi[user_bs_id] + 
                                    Diffraction_loss_matrxi[user_bs_id]) /10) * (10**(self.Power_bs*self.factor[bs_id]/(10*antenna_req_num))) #W 除以天线的数量 后面需要改
                # for other_user_bs_id in user_bs: #?基站用户之间的干扰(暂不考虑)
                #     if user_bs_id == other_user_bs_id:  #遍历基站内的其他用户
                #         continue
                #     else:
                #         Gain_interf = 10 * np.log10(Gain_matrix[other_user_bs_id][user_bs_id]) #dBi
                #         interf = 10 ** ((Gain_interf+self.Gr_user + Path_loss_matrxi[user_bs_id]) /10) * (10 ** (self.Power_bs*self.factor[bs_id]/(10*len(user_bs)))) #  其他基站内被基站服务的用户的干扰
                #         interference += interf
                for user_sa_id in user_sa:
                    Gain_sa_interf = 10 * np.log10(Gain_sa_matrix[user_sa_id][user_bs_id]) #dBi  来自其他基站内卫星用户的增益矩阵
                    interf = 10 ** ((Gain_sa_interf+self.Gr_user + Path_loss_sa[0][user_bs_id]) /10) * self.Beam_Power[user_sa_id] #  其他基站内被卫星服务的用户的干扰
                    interference += interf
                sinr = power_self / (self.noisy + interference)
                max_sinr = power_self / self.noisy
                sinr_bs_temp[user_bs_id] = sinr
                self.sinr_matrix[user_bs_id] = sinr
                self.max_sinr_matrix[user_bs_id] = max_sinr

            if bs_id_state["choose_by_SINR"] == True: # 判断是否按照SINR进行连接分配
                idx = np.argsort(sinr_bs_temp)[-antenna_req_num:] if antenna_req_num!=0 else []
                bs_state[bs_id]["user_bs"]=list(idx)
                bs_state[bs_id]["user_unserve"]=list(np.setdiff1d(bs_state[bs_id]["user_bs_req"], bs_state[bs_id]["user_bs"]))
                for id in user_bs:
                    if np.isin(id,idx)==False:
                        self.sinr_matrix[id] = 0
                        self.max_sinr_matrix[id] = 0
                




            


    def bs2sa_inter(self,bs_state,req_user_info,bs_lla):
        """
        计算初始基站对卫星用户的干扰，从而计算基站功率系数
        """
        Gain_matrix = self.get_bs_gain(req_user_info,bs_lla)#获取增益矩阵
        Path_loss_matrxi = self.get_bs_loss_path(req_user_info)#计算每个用户的路径损耗
        for bs_id,bs_id_state in enumerate(bs_state):                                                        
            user_bs=bs_id_state["user_bs"] 
            user_sa=bs_id_state["user_sa"] 
            for user_sa_id in user_sa: 
                for user_bs_id in user_bs:
                    Gain_interf = 10 * np.log10(Gain_matrix[user_bs_id][user_sa_id]) #dBi
                    interf = 10 ** ((Gain_interf+self.Gr_user + Path_loss_matrxi[user_sa_id]) /10) * (10 ** (self.Power_bs/(10*len(user_bs)))) #  其他基站内被基站服务的用户的干扰
                    self.interference_bs2sa[bs_id][user_sa_id] += interf
    def caculate_bs_power_factor(self):
        NTN_matrix = np.zeros(Parameters.bs_num)+10 ** (self.BS_TNT_TH / 10)
        row_max = np.amax(self.interference_bs2sa, axis=1)
        self.factor = NTN_matrix/row_max
        # 将结果中的 inf 替换为 1
        self.factor[self.factor == np.inf] = 1
        # 若大于1则不进行基站功率控制
        self.factor[self.factor>1] = 1


def calculate_datarate(Action_beam, req_user_info, req_list,bs_lla,bs_state):
    """
    根据选定动作计算传输速率
    输入参数: Action_beam(波束选择用户的动作) req_user_info(用户的状态信息)   req_list(用户的请求id是一个数组)

    """
    ## 下行传输
    downlink_tool = downlink_transmision_tool()
    downlink_tool.bs2sa_inter(bs_state,req_user_info,bs_lla) #计算基站-卫星干扰
    downlink_tool.caculate_bs_power_factor()# 进行基站功率控制
    print("factor",downlink_tool.factor)
    
    Gain_sa_matrix,Path_loss_sa= downlink_tool.get_sa_sinr(Action_beam, req_user_info, req_list) #获得卫星用户的SINR
    downlink_tool.get_bs_sinr(req_user_info,bs_lla,bs_state,Gain_sa_matrix,Path_loss_sa) #获得基站用户的SINR
    DOWN_Rate = np.log2(1 + downlink_tool.sinr_matrix) * downlink_tool.bw/downlink_tool.antenna_req_num
    MAX_DOWN_Rate = np.log2(1 + downlink_tool.max_sinr_matrix) * downlink_tool.bw/downlink_tool.antenna_req_num
    print("DOWN_Rate", downlink_tool.sinr_matrix)
    print("MAX_DOWN_Rate",downlink_tool.max_sinr_matrix)
    return DOWN_Rate,MAX_DOWN_Rate






if __name__ == '__main__':
    label = []
    tool = downlink_transmision_tool()
    for i in range(12):
        power_self = 10 ** ((tool.G_peak[i] + tool.Gr_user + tool.path_loss) / 10) * (10 ** (tool.PowerT_beam[i]/10))
        sinr = power_self / tool.noisy
        cap = np.log2(sinr + 1) * tool.bw / 1000
        label.append(cap)
    print("label", label)
    label_1 = [1170427.0388427917, 1108976.7032083324, 1137409.6666052616, 1166643.4797618384, 1195810.576419014, 982913.1870934572, 1249107.7273640872, 1136223.9501351079, 1028573.0350210491, 1221446.8528336838, 1269427.1150338312, 1075346.1870396722]
    print("****************", np.sum(np.array(label_1))*50)

