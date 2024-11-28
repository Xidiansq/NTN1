# -*-coding:utf-8-*-
import math
import numpy as np
import math as m
import Parameters
import Satellite_run
from scipy.special import jn
from scipy.optimize import minimize
import Satellite_Bs
import cvxpy as cp


class downlink_transmision_tool:
    def __init__(self):
        #用户参数
        self.req_user_num = 0
        self.otheruser_conbeam_num = 0
        self.Gr_user =  Parameters.Gain_user #  30 dBi

        #通信参数
        self.bw = Parameters.bw          #带宽 500e6 HZ
        self.T_b = Parameters.T_b          #时隙时间
        self.frequency = Parameters.frequency #中心频率2e10   # Hz
        self.velocity = 3e8
        self.gama = 0.5
        self.noisy = 2.5118864315095823e-12

        #卫星参数
        self.sate_lla = Parameters.sate_lla
        self.sate_xyz = Parameters.sate_xyz
        self.Hleo = Parameters.HLeo           #550000   # m
        self.Power_SateTotal = Parameters.Power_SateTotal   #30dBW
        self.Power_Beam_average = self.Power_SateTotal/ Parameters.beam_open
        self.Power_BeamMax = Parameters.Power_BeamMax     #0.8*total
        self.Gain_beam = Parameters.Gain_Beam            #42dBi
        self.sate_threedB = Parameters.sate_threedB      #1dB
        # self.path_loss = -209.53778  # dBw
        # self.PowerT_beam = 10  #dBw   #3W
        
        #基站参数
        self.barrier_height = Parameters.barrier_height  # 障碍物高度
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
                    #获取对应的弧度值
                    #userN_position = np.radians(np.array([req_user_info.at[index_n, 'Lat'], req_user_info.at[index_n, 'Lon'], req_user_info.at[index_n, 'Alt']]))
                    #计算偏移角
                    # angle_K2N = np.arccos(np.sin(userK_position[0]) * np.sin(userN_position[0]) + 
                    #                             np.cos(userK_position[0]) * np.cos(userN_position[0]) * np.cos(userK_position[1] - userN_position[1]))
                    #print("req_user_info.at[index_k]",np.array([req_user_info.at[index_k, 'Lat'], req_user_info.at[index_k, 'Lon'], req_user_info.at[index_k, 'Alt']]))
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
        for i in range(self.req_user_num):
            interference=0
            if(action[i] == 0): #i 就是代表这个请求用户会被服务，计算这个用户的sinr就行
                continue
            else:
                #首先这个用户会接受来自自己的增益
                Gain_self = 10 * np.log10(Gain_matrix[i][i]) #dBi
                power_self = 10 ** ((Gain_self + self.Gr_user + Path_loss_matrxi[0][i]) /10) * (10**(self.Power_Beam_average/10)) #W
                for j in range(self.req_user_num):
                    if i == j or action[j] == 0:
                        continue
                    else:
                        Gain_interf = 10 * np.log10(Gain_matrix[j][i]) #dBi
                        interf = 10 ** ((Gain_interf+self.Gr_user + Path_loss_matrxi[0][i]) /10) * (10 ** (self.Power_Beam_average/10)) #  #其他波束干扰+bs干扰阈值
                        interference += interf
                interference += 10 ** (self.BS_TNT_TH / 10) #最后加上来自基站的干扰
                #interference = 10 ** ((self.Gr_user + Path_loss_matrxi[0][i]) / 10) * interference 
                sinr_average = power_self / (self.noisy + interference)
                max_sinr = power_self / (self.noisy + 10 ** (self.BS_TNT_TH / 10))
                self.sinr_matrix[i] = sinr_average
                self.max_sinr_matrix[i] = max_sinr
        ################## 功率优化求满足最优条件的Power_Beam————CVX方法 ##################
        temp_average_matrix = self.sinr_matrix[self.sinr_matrix != 0]
        N_k = np.count_nonzero(action == 1)
        # 定义功率分配变量
        P = cp.Variable(N_k, nonneg=True)  # 每个用户的功率分配

        #获取哪些用户被照射
        selectted_user = np.where(action == 1)[0]#用户id
        #计算SINR
        sinr_expr = []
        for i in selectted_user:
            #首先这个用户会接受来自自己的增益
            Gain_self = 10 * np.log10(Gain_matrix[i][i]) #dBi
            user_in_selected = np.where(selectted_user == i)[0][0]
            #直接用w
            h1 =  10 ** ((Gain_self + self.Gr_user + Path_loss_matrxi[0][i]) /10)
            power_self = h1 *P[user_in_selected]
            # power_self = cp.exp(cp.multiply(np.log(10), (Gain_self + self.Gr_user + Path_loss_matrxi[0][i]) / 10)) * cp.exp(cp.multiply(np.log(10), P[user_in_selected] / 10))

            #计算干扰
            interference=0
            for j in range(self.req_user_num):
                if i == j or action[j] == 0:
                    continue
                else:
                    Gain_interf = 10 * np.log10(Gain_matrix[j][i]) #dBi
                    interf_index_in_selected = np.where(selectted_user == j)[0][0]
                    h2 = 10 ** ((Gain_interf+self.Gr_user + Path_loss_matrxi[0][i]) /10)
                    interf = h2 *P[interf_index_in_selected]
                    # interf = cp.exp(cp.multiply(np.log(10), (Gain_interf + self.Gr_user + Path_loss_matrxi[0][i]) / 10)) * cp.exp(cp.multiply(np.log(10), P[interf_index_in_selected] / 10))  # 其他波束干扰+bs干扰阈值
                    interference += interf
            # interference += 10 ** (self.BS_TNT_TH / 10) #最后加上来自基站的干扰
            #interference = 10 ** ((self.Gr_user + Path_loss_matrxi[0][i]) / 10) * interference 
            sinr_current = power_self / (self.noisy + interference)
            print("当前的power",power_self)
            print("当前的inter",interference)

            print("当前的sinr",sinr_current)
            print("对数转化后",cp.log(1+sinr_current))
            input()
            sinr_expr.append(cp.log(1 + sinr_current))

        #目标函数： 最大化SINR
        alpha = (np.array(temp_average_matrix)**2) / (1+np.array(temp_average_matrix))
        alpha = cp.Parameter(len(sinr_expr), value = alpha)
    
        objective = cp.Maximize(-cp.sum(cp.multiply(alpha,cp.hstack(sinr_expr))))

        #计算约束
        constraints = [cp.sum(P) <= self.Power_BeamMax]

        # 定义和求解优化问题
        problem = cp.Problem(objective, constraints)
        problem.solve()
        # 输出结果
        if problem.status == cp.OPTIMAL:
            print("Optimal Power Allocation:", P.value)
        else:
            print("Optimization failed:", problem.status)

        input()
        ############# 功率优化求满足最优条件的Power_Beam————scipy.optimize方法 #############
        # temp_average_matrix = self.sinr_matrix[self.sinr_matrix != 0]
        # alpha = (temp_average_matrix**2) / (1 + temp_average_matrix)
        # N_k = np.count_nonzero(action == 1)
        
        # # 假设已经初始化以下参数：
        # # temp_average_matrix, temp_current_matrix, alpha, N_k, self.Power_SateTotal, self.bw, self.T_b
        # # 定义目标函数和约束条件
        # def objective(P, alpha, N_k):
        #     # temp_current_matrix 使用 P 重新计算
        #     temp_current_matrix = []
        #     temp_i = 0
        #     for i in range(self.req_user_num):
        #         interference=0
        #         if(action[i] == 0): #i 就是代表这个请求用户会被服务，计算这个用户的sinr就行
        #             continue
        #         else:
        #             #首先这个用户会接受来自自己的增益
        #             Gain_self = 10 * np.log10(Gain_matrix[i][i]) #dBi
        #             power_self = 10 ** ((Gain_self + self.Gr_user + Path_loss_matrxi[0][i]) /10) * (10**(P[temp_i]/10))
        #             temp_j = 0
        #             for j in range(self.req_user_num):
        #                 if i == j or action[j] == 0:
        #                     continue
        #                 else:
        #                     Gain_interf = 10 * np.log10(Gain_matrix[j][i]) #dBi
        #                     interf = 10 ** ((Gain_self + self.Gr_user + Path_loss_matrxi[0][i]) /10) * (10**(P[temp_j]/10)) #W     # 其他波束干扰+bs干扰阈值
        #                     interference += interf
        #                     temp_j += 1
        #             interference += 10 ** (self.BS_TNT_TH / 10) #最后加上来自基站的干扰
        #             sinr_current = power_self / (self.noisy + interference)
        #             # sinr_current = cp.log(power_self) - cp.log(self.noisy + interference)
        #             temp_current_matrix.append(sinr_current)
        #             temp_i += 1
        #     return -np.sum(alpha / np.array(temp_current_matrix))
        # def power_constraint(P, total_power):
        #     return total_power - np.sum(P)
        # def shannon_capacity_constraint(P, bw, T_b):
        #     temp_current_matrix = []
        #     temp_i = 0
        #     for i in range(self.req_user_num):
        #         interference=0
        #         if(action[i] == 0): #i 就是代表这个请求用户会被服务，计算这个用户的sinr就行
        #             continue
        #         else:
        #             #首先这个用户会接受来自自己的增益
        #             Gain_self = 10 * np.log10(Gain_matrix[i][i]) #dBi
        #             power_self = 10 ** ((Gain_self + self.Gr_user + Path_loss_matrxi[0][i]) /10) * (10**(P[temp_i]/10))
        #             temp_j = 0
        #             for j in range(self.req_user_num):
        #                 if i == j or action[j] == 0:
        #                     continue
        #                 else:
        #                     Gain_interf = 10 * np.log10(Gain_matrix[j][i]) #dBi
        #                     interf = 10 ** ((Gain_self + self.Gr_user + Path_loss_matrxi[0][i]) /10) * (10**(P[temp_j]/10)) #W     # 其他波束干扰+bs干扰阈值
        #                     interference += interf
        #                     temp_j += 1
        #             interference += 10 ** (self.BS_TNT_TH / 10) #最后加上来自基站的干扰
        #             sinr_current = power_self / (self.noisy + interference)
        #             # sinr_current = cp.log(power_self) - cp.log(self.noisy + interference)
        #             temp_current_matrix.append(sinr_current)
        #             temp_i += 1
        #     capacities = np.array([bw * T_b * np.log2(1 + temp_current_matrix[k]) for k in range(N_k)])
        #     return 100 * 1e3 - np.min(capacities)
        # # 初始功率分配
        # P0 = np.ones(N_k) * (self.Power_SateTotal / N_k)
        # # 定义约束
        # constraints = [
        #     {'type': 'ineq', 'fun': power_constraint, 'args': (self.Power_SateTotal,)},
        #     {'type': 'ineq', 'fun': shannon_capacity_constraint, 'args': (self.bw, self.T_b)},
        # ]
        # # 执行优化
        # result = minimize(
        #     objective,
        #     P0,
        #     args=(alpha, N_k),
        #     constraints=constraints,
        #     bounds=[(0, None) for _ in range(N_k)]  # 非负功率分配
        # )
        # # 输出结果
        # if result.success:
        #     print("Optimal Power Allocation (Scipy Optimization):", result.x)
        # else:
        #     print("Optimization failed:", result.message)
        #################################################################################
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
    def get_bs_diffraction_loss(self, req_user_info, h, frequency):
        """
        计算单刀刃绕射模型的损耗。

        参数:
        d1 (float): 发射端与障碍物的距离。
        d2 (float): 接收端与障碍物的距离。
        h (float): 障碍物的高度。
        lambda_ (float): 波长。

        返回:
        float: 绕射损耗（单位：分贝）。
        """
        # 计算公式中的菲涅尔-基尔霍夫绕射参数v
        d1 = np.array(req_user_info["Dis_Bs"])/2
        d2 = np.array(req_user_info["Dis_Bs"])/2
        lambda_ = self.velocity / frequency
        v = h * np.sqrt((2 * (d1 + d2)) / (lambda_ * d1 * d2))
        # 计算总损耗
        DIFFRACTION_LOSS = 20 * np.log10(np.sqrt((v-0.1)**2 + 1) + v - 0.1) + 6.9
        print("DIFFRACTION_LOSS",DIFFRACTION_LOSS)
        return - DIFFRACTION_LOSS
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
            this_user_position=np.radians(np.array([req_user_info.at[this_user_id, 'Lat'], req_user_info.at[this_user_id, 'Lon'], req_user_info.at[this_user_id, 'Alt']]))
            bs_id=int(req_user_info.at[this_user_id,'BsID'])
            bsM_position = np.radians(np.array([bs_lla[bs_id][1],bs_lla[bs_id][2],bs_lla[bs_id][3]]))
            for other_user_id in range(self.req_user_num ):  # 遍历用户
                if req_user_info.at[other_user_id,'BsID']==bs_id:
                    other_user_position = np.radians(np.array([req_user_info.at[other_user_id, 'Lat'], req_user_info.at[other_user_id, 'Lon'], req_user_info.at[other_user_id, 'Alt']]))
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
        Diffraction_loss_matrxi = self.get_bs_diffraction_loss(req_user_info, self.barrier_height, self.frequency)
        self.req_user_num = len(req_user_info)
        for _,bs_id_state in enumerate(bs_state):          #遍历每个基站                                                        #i 就是代表这个请求用户会被服务，计算这个用户的sinr就行
            user_sa=bs_id_state["user_sa"]                      #当前小区的卫星用户
            user_bs=bs_id_state["user_bs"]                      #当前小区的基站服务用户
            for user_bs_id in user_bs:
                interference=0
                #基站给此用户的增益
                Gain_self = 10 * np.log10(Gain_matrix[user_bs_id][user_bs_id]) #dBi
                power_self = 10 ** ((Gain_self + self.Gr_user +       
                                     Path_loss_matrxi[user_bs_id] + 
                                     Diffraction_loss_matrxi[user_bs_id]) /10) * (10**(self.Power_bs/(10*len(user_bs)))) #W 除以天线的数量 后面需要改
                for other_user_bs_id in user_bs:
                    if user_bs_id == other_user_bs_id:  #遍历基站内的其他用户
                        continue
                    else:
                        Gain_interf = 10 * np.log10(Gain_matrix[other_user_bs_id][user_bs_id]) #dBi
                        interf = 10 ** ((Gain_interf+self.Gr_user + Path_loss_matrxi[user_bs_id]) /10) * (10 ** (self.Power_bs/(10*len(user_bs)))) #  其他基站内被基站服务的用户的干扰
                        interference += interf
                for user_sa_id in user_sa:
                    
                    Gain_sa_interf = 10 * np.log10(Gain_sa_matrix[user_sa_id][user_bs_id]) #dBi  来自其他基站内卫星用户的增益矩阵
                    interf = 10 ** ((Gain_sa_interf+self.Gr_user + Path_loss_sa[0][user_bs_id]) /10) * (10 ** (self.Power_Beam_average/10)) #  其他基站内被卫星服务的用户的干扰
                    interference += interf

                #interference = 10 ** ((self.Gr_user + Path_loss_matrxi[user_bs_id]) / 10) * interference 
                sinr = power_self / (self.noisy + interference)
                max_sinr = power_self / self.noisy
                self.sinr_matrix[user_bs_id] = sinr
                self.max_sinr_matrix[user_bs_id] = max_sinr
        # print('self.sinr_matrix',self.sinr_matrix)
        return True
def calculate_datarate(Action_beam, req_user_info, req_list,bs_lla,bs_state):
    """
    根据选定动作计算传输速率
    输入参数: Action_beam(波束选择用户的动作) req_user_info(用户的状态信息)   req_list(用户的请求id是一个数组)

    """
    ## 下行传输
    downlink_tool = downlink_transmision_tool()
    Gain_sa_matrix,Path_loss_sa= downlink_tool.get_sa_sinr(Action_beam, req_user_info, req_list) #获得卫星用户的SINR
    downlink_tool.get_bs_sinr( req_user_info,bs_lla,bs_state,Gain_sa_matrix,Path_loss_sa) #获得基站用户的SINR
    DOWN_Rate = np.log2(1 + downlink_tool.sinr_matrix) * downlink_tool.bw
    MAX_DOWN_Rate = np.log2(1 + downlink_tool.max_sinr_matrix) * downlink_tool.bw
    # print("DOWN_Rate", downlink_tool.sinr_matrix)
    # print("MAX_DOWN_Rate",downlink_tool.max_sinr_matrix)
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

