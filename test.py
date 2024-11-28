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
        #�û�����
        self.req_user_num = 0
        self.otheruser_conbeam_num = 0
        self.Gr_user =  Parameters.Gain_user #  30 dBi

        #ͨ�Ų���
        self.bw = Parameters.bw          #���� 500e6 HZ
        self.T_b = Parameters.T_b          #ʱ϶ʱ��
        self.frequency = Parameters.frequency #����Ƶ��2e10   # Hz
        self.velocity = 3e8
        self.gama = 0.5
        self.noisy = 2.5118864315095823e-12

        #���ǲ���
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
        
        #��վ����
        self.barrier_height = Parameters.barrier_height  # �ϰ���߶�
        self.BS_TNT_TH = Parameters.BS_INT_TH   #������ֵ
        self.Power_bs = Parameters.Power_bs
        self.Gain_bs = Parameters.Gain_bs
        self.front_to_back_ratio = Parameters.front_to_back_ratio  #ǰ���
        self.R_earth = Parameters.R_earth
        self.bs_threedB=Parameters.bs_threedB #3dB��
        self.antenna_num=Parameters.antenna_num #��վ��������
        #�������
        self.sinr_matrix = np.zeros(Parameters.user_number)
        self.max_sinr_matrix = np.zeros(Parameters.user_number)
    def get_sa_loss_path(self, req_user_info, req_list):
        """
        ����·����ģ������û������ǵľ������
        ����������������û���������Ϣ(req_user_info), �������û���id�б�(req_list)
        ����: ÿ���������û���·�����
        """
        self.req_user_num = len(req_user_info)
        LOSS_PATH = np.zeros((1, self.req_user_num))
        for i in range(self.req_user_num):
            index_n = i
            #��ȡ��ǰ�û���λ�ã�ά�ȡ����ȡ����Σ�
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
        �����������
        ����������������û���������Ϣ(req_user_info), �������û���id�б�(req_list)
        ����: �û����������
        """
        self.req_user_num = len(req_user_info)#��ȡ�������û�������
        self.otheruser_conbeam_num = self.req_user_num#ͬ���������Բ������ŵ��û�Ҳ���������
        theta_matrix = np.zeros((self.req_user_num, self.otheruser_conbeam_num))#����һ���ǶȾ��󣬱�ʾÿ���û�֮��ĽǶ�
        Gain_matrix = np.full((self.req_user_num, self.otheruser_conbeam_num), float(self.Gain_beam))#����һ��������󣬱�ʾ�ŵ�����
        # print("��ʼֵ",Gain_matrix)
        # input()
        # distance_matrix = np.zeros((self.req_user_num, self.otheruser_conbeam_num))#����һ��������󣬼���ÿ���û�֮��ľ���
        for i in range(self.req_user_num):
            index_k = i #��¼��ǰ�û���id
            #��ȡ��ǰ�û���λ����Ϣ
            userK_position = np.radians(np.array([req_user_info.at[index_k, 'Lat'], req_user_info.at[index_k, 'Lon'], req_user_info.at[index_k, 'Alt']]))
            for j in range(self.otheruser_conbeam_num):#�ڲ���������û�
                index_n = j #��¼��ǰ�û�id
                if(index_n == index_k):
                    continue
                else:
                    #��ȡ��Ӧ�Ļ���ֵ
                    #userN_position = np.radians(np.array([req_user_info.at[index_n, 'Lat'], req_user_info.at[index_n, 'Lon'], req_user_info.at[index_n, 'Alt']]))
                    #����ƫ�ƽ�
                    # angle_K2N = np.arccos(np.sin(userK_position[0]) * np.sin(userN_position[0]) + 
                    #                             np.cos(userK_position[0]) * np.cos(userN_position[0]) * np.cos(userK_position[1] - userN_position[1]))
                    #print("req_user_info.at[index_k]",np.array([req_user_info.at[index_k, 'Lat'], req_user_info.at[index_k, 'Lon'], req_user_info.at[index_k, 'Alt']]))
                    angle_K2N = np.radians(Satellite_Bs.angle_between_users(np.array([req_user_info.at[index_k, 'Lat'], req_user_info.at[index_k, 'Lon'], req_user_info.at[index_k, 'Alt']]),
                                                                 np.array([req_user_info.at[index_n, 'Lat'], req_user_info.at[index_n, 'Lon'], req_user_info.at[index_n, 'Alt']]),
                                                                 Parameters.sate_lla)
                    )
                    
                    u = 2.01723 * np.sin(angle_K2N) / np.sin(np.radians(self.sate_threedB))
                    #���㱴��������ֵ
                    j1_u = jn(1, u)
                    j3_u = jn(3, u)
                    Gain_K2N = self.Gain_beam * ( (j1_u / (2 * u)) + (36 * j3_u / (u**3)) )**2
                    theta_matrix[i][j] = np.degrees(angle_K2N)
                    Gain_matrix[i][j] = Gain_K2N #dBi
                    # Gain_matrix[i][j]= self.Gain_beam - ((12 * (10 ** (self.Gain_beam / 10))) / self.gama) * np.square(theta_matrix[i][j] / (70 * np.pi))#
        Gain_matrix = 10 ** (Gain_matrix / 10)#ת��Ϊ����ֵ
        return Gain_matrix

    def get_sa_sinr(self, action, req_user_info, req_list):
        """
        ��������û����Ÿ����
        ���룺
        action:�����������
        req_user_info:�û���Ϣ

        """
        # print("���д��䶯��", action)
        
        #action ����ѡ������Щ�û�,��actionת��Ϊһ��һά�������ĵط���1����������0

        Gain_matrix = self.get_sa_gain(req_user_info, req_list)#��ȡ�������
        Path_loss_matrxi = self.get_sa_loss_path(req_user_info, req_list)#����ÿ���û���·�����
        self.req_user_num = len(req_user_info)
        for i in range(self.req_user_num):
            interference=0
            if(action[i] == 0): #i ���Ǵ�����������û��ᱻ���񣬼�������û���sinr����
                continue
            else:
                #��������û�����������Լ�������
                Gain_self = 10 * np.log10(Gain_matrix[i][i]) #dBi
                power_self = 10 ** ((Gain_self + self.Gr_user + Path_loss_matrxi[0][i]) /10) * (10**(self.Power_Beam_average/10)) #W
                for j in range(self.req_user_num):
                    if i == j or action[j] == 0:
                        continue
                    else:
                        Gain_interf = 10 * np.log10(Gain_matrix[j][i]) #dBi
                        interf = 10 ** ((Gain_interf+self.Gr_user + Path_loss_matrxi[0][i]) /10) * (10 ** (self.Power_Beam_average/10)) #  #������������+bs������ֵ
                        interference += interf
                interference += 10 ** (self.BS_TNT_TH / 10) #���������Ի�վ�ĸ���
                #interference = 10 ** ((self.Gr_user + Path_loss_matrxi[0][i]) / 10) * interference 
                sinr_average = power_self / (self.noisy + interference)
                max_sinr = power_self / (self.noisy + 10 ** (self.BS_TNT_TH / 10))
                self.sinr_matrix[i] = sinr_average
                self.max_sinr_matrix[i] = max_sinr
        ################## �����Ż�����������������Power_Beam��������CVX���� ##################
        temp_average_matrix = self.sinr_matrix[self.sinr_matrix != 0]
        N_k = np.count_nonzero(action == 1)
        # ���幦�ʷ������
        P = cp.Variable(N_k, nonneg=True)  # ÿ���û��Ĺ��ʷ���

        #��ȡ��Щ�û�������
        selectted_user = np.where(action == 1)[0]#�û�id
        #����SINR
        sinr_expr = []
        for i in selectted_user:
            #��������û�����������Լ�������
            Gain_self = 10 * np.log10(Gain_matrix[i][i]) #dBi
            user_in_selected = np.where(selectted_user == i)[0][0]
            #ֱ����w
            h1 =  10 ** ((Gain_self + self.Gr_user + Path_loss_matrxi[0][i]) /10)
            power_self = h1 *P[user_in_selected]
            # power_self = cp.exp(cp.multiply(np.log(10), (Gain_self + self.Gr_user + Path_loss_matrxi[0][i]) / 10)) * cp.exp(cp.multiply(np.log(10), P[user_in_selected] / 10))

            #�������
            interference=0
            for j in range(self.req_user_num):
                if i == j or action[j] == 0:
                    continue
                else:
                    Gain_interf = 10 * np.log10(Gain_matrix[j][i]) #dBi
                    interf_index_in_selected = np.where(selectted_user == j)[0][0]
                    h2 = 10 ** ((Gain_interf+self.Gr_user + Path_loss_matrxi[0][i]) /10)
                    interf = h2 *P[interf_index_in_selected]
                    # interf = cp.exp(cp.multiply(np.log(10), (Gain_interf + self.Gr_user + Path_loss_matrxi[0][i]) / 10)) * cp.exp(cp.multiply(np.log(10), P[interf_index_in_selected] / 10))  # ������������+bs������ֵ
                    interference += interf
            # interference += 10 ** (self.BS_TNT_TH / 10) #���������Ի�վ�ĸ���
            #interference = 10 ** ((self.Gr_user + Path_loss_matrxi[0][i]) / 10) * interference 
            sinr_current = power_self / (self.noisy + interference)
            print("��ǰ��power",power_self)
            print("��ǰ��inter",interference)

            print("��ǰ��sinr",sinr_current)
            print("����ת����",cp.log(1+sinr_current))
            input()
            sinr_expr.append(cp.log(1 + sinr_current))

        #Ŀ�꺯���� ���SINR
        alpha = (np.array(temp_average_matrix)**2) / (1+np.array(temp_average_matrix))
        alpha = cp.Parameter(len(sinr_expr), value = alpha)
    
        objective = cp.Maximize(-cp.sum(cp.multiply(alpha,cp.hstack(sinr_expr))))

        #����Լ��
        constraints = [cp.sum(P) <= self.Power_BeamMax]

        # ���������Ż�����
        problem = cp.Problem(objective, constraints)
        problem.solve()
        # ������
        if problem.status == cp.OPTIMAL:
            print("Optimal Power Allocation:", P.value)
        else:
            print("Optimization failed:", problem.status)

        input()
        ############# �����Ż�����������������Power_Beam��������scipy.optimize���� #############
        # temp_average_matrix = self.sinr_matrix[self.sinr_matrix != 0]
        # alpha = (temp_average_matrix**2) / (1 + temp_average_matrix)
        # N_k = np.count_nonzero(action == 1)
        
        # # �����Ѿ���ʼ�����²�����
        # # temp_average_matrix, temp_current_matrix, alpha, N_k, self.Power_SateTotal, self.bw, self.T_b
        # # ����Ŀ�꺯����Լ������
        # def objective(P, alpha, N_k):
        #     # temp_current_matrix ʹ�� P ���¼���
        #     temp_current_matrix = []
        #     temp_i = 0
        #     for i in range(self.req_user_num):
        #         interference=0
        #         if(action[i] == 0): #i ���Ǵ�����������û��ᱻ���񣬼�������û���sinr����
        #             continue
        #         else:
        #             #��������û�����������Լ�������
        #             Gain_self = 10 * np.log10(Gain_matrix[i][i]) #dBi
        #             power_self = 10 ** ((Gain_self + self.Gr_user + Path_loss_matrxi[0][i]) /10) * (10**(P[temp_i]/10))
        #             temp_j = 0
        #             for j in range(self.req_user_num):
        #                 if i == j or action[j] == 0:
        #                     continue
        #                 else:
        #                     Gain_interf = 10 * np.log10(Gain_matrix[j][i]) #dBi
        #                     interf = 10 ** ((Gain_self + self.Gr_user + Path_loss_matrxi[0][i]) /10) * (10**(P[temp_j]/10)) #W     # ������������+bs������ֵ
        #                     interference += interf
        #                     temp_j += 1
        #             interference += 10 ** (self.BS_TNT_TH / 10) #���������Ի�վ�ĸ���
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
        #         if(action[i] == 0): #i ���Ǵ�����������û��ᱻ���񣬼�������û���sinr����
        #             continue
        #         else:
        #             #��������û�����������Լ�������
        #             Gain_self = 10 * np.log10(Gain_matrix[i][i]) #dBi
        #             power_self = 10 ** ((Gain_self + self.Gr_user + Path_loss_matrxi[0][i]) /10) * (10**(P[temp_i]/10))
        #             temp_j = 0
        #             for j in range(self.req_user_num):
        #                 if i == j or action[j] == 0:
        #                     continue
        #                 else:
        #                     Gain_interf = 10 * np.log10(Gain_matrix[j][i]) #dBi
        #                     interf = 10 ** ((Gain_self + self.Gr_user + Path_loss_matrxi[0][i]) /10) * (10**(P[temp_j]/10)) #W     # ������������+bs������ֵ
        #                     interference += interf
        #                     temp_j += 1
        #             interference += 10 ** (self.BS_TNT_TH / 10) #���������Ի�վ�ĸ���
        #             sinr_current = power_self / (self.noisy + interference)
        #             # sinr_current = cp.log(power_self) - cp.log(self.noisy + interference)
        #             temp_current_matrix.append(sinr_current)
        #             temp_i += 1
        #     capacities = np.array([bw * T_b * np.log2(1 + temp_current_matrix[k]) for k in range(N_k)])
        #     return 100 * 1e3 - np.min(capacities)
        # # ��ʼ���ʷ���
        # P0 = np.ones(N_k) * (self.Power_SateTotal / N_k)
        # # ����Լ��
        # constraints = [
        #     {'type': 'ineq', 'fun': power_constraint, 'args': (self.Power_SateTotal,)},
        #     {'type': 'ineq', 'fun': shannon_capacity_constraint, 'args': (self.bw, self.T_b)},
        # ]
        # # ִ���Ż�
        # result = minimize(
        #     objective,
        #     P0,
        #     args=(alpha, N_k),
        #     constraints=constraints,
        #     bounds=[(0, None) for _ in range(N_k)]  # �Ǹ����ʷ���
        # )
        # # ������
        # if result.success:
        #     print("Optimal Power Allocation (Scipy Optimization):", result.x)
        # else:
        #     print("Optimization failed:", result.message)
        #################################################################################
        return Gain_matrix,Path_loss_matrxi

    def get_bs_loss_path(self, req_user_info):
        """
        ����·����ģ������û��ͻ�վ�ľ������
        ����������������û���������Ϣ(req_user_info), �������û���id�б�(req_list)
        ���: ÿ��������Ļ�վ�û���·�����
        ��������ɿռ����(dbi)
        """
        LOSS_PATH = np.array([((4 * m.pi * self.frequency * req_user_info.at[user_id,"Dis_Bs"]) / self.velocity) ** (-2)
                              for user_id in range(len(req_user_info))])
        

        return 10*np.log10(LOSS_PATH)
    def get_antenna_peruser_gain(self,angle_antenna_user):
        """
        �����վ���߶Ի�վ�û�������
        angle_antenna_user���������û���ļн�
        G_BS_max�����������
        ע�⣺
        1:����û�п��ǵ����������û��мн�
        """
        return 10**((self.Gain_bs/10)+(np.max([-0.6*(angle_antenna_user/self.bs_threedB)**2,-(self.front_to_back_ratio/10)])))
    def get_bs_diffraction_loss(self, req_user_info, h, frequency):
        """
        ���㵥��������ģ�͵���ġ�

        ����:
        d1 (float): ��������ϰ���ľ��롣
        d2 (float): ���ն����ϰ���ľ��롣
        h (float): �ϰ���ĸ߶ȡ�
        lambda_ (float): ������

        ����:
        float: ������ģ���λ���ֱ�����
        """
        # ���㹫ʽ�еķ�����-���������������v
        d1 = np.array(req_user_info["Dis_Bs"])/2
        d2 = np.array(req_user_info["Dis_Bs"])/2
        lambda_ = self.velocity / frequency
        v = h * np.sqrt((2 * (d1 + d2)) / (lambda_ * d1 * d2))
        # ���������
        DIFFRACTION_LOSS = 20 * np.log10(np.sqrt((v-0.1)**2 + 1) + v - 0.1) + 6.9
        print("DIFFRACTION_LOSS",DIFFRACTION_LOSS)
        return - DIFFRACTION_LOSS
    def get_bs_gain(self, req_user_info,bs_lla):
        """
        �����������
        ����������������û���������Ϣ(req_user_info),��վλ����Ϣ(bs_lla)
        ����: �û����������
        """
        self.req_user_num = len(req_user_info)#��ȡ�������û�������
        self.bs_num = len(bs_lla)#ͬ���������Բ������ŵ��û�Ҳ���������
        theta_matrix = np.zeros((self.req_user_num, self.req_user_num))#����һ���ǶȾ��󣬱�ʾÿ���û�֮����ڻ�վ�ĽǶ�
        Gain_matrix = np.full((self.req_user_num, self.req_user_num), 0)#����һ��������󣬱�ʾ�ŵ�����
        # distance_matrix = np.zeros((self.req_user_num, self.otheruser_conbeam_num))#����һ��������󣬼���ÿ���û�֮��ľ���
        for this_user_id in range(self.req_user_num ):
            #��ȡ��ǰ�û���λ����Ϣ
            this_user_position=np.radians(np.array([req_user_info.at[this_user_id, 'Lat'], req_user_info.at[this_user_id, 'Lon'], req_user_info.at[this_user_id, 'Alt']]))
            bs_id=int(req_user_info.at[this_user_id,'BsID'])
            bsM_position = np.radians(np.array([bs_lla[bs_id][1],bs_lla[bs_id][2],bs_lla[bs_id][3]]))
            for other_user_id in range(self.req_user_num ):  # �����û�
                if req_user_info.at[other_user_id,'BsID']==bs_id:
                    other_user_position = np.radians(np.array([req_user_info.at[other_user_id, 'Lat'], req_user_info.at[other_user_id, 'Lon'], req_user_info.at[other_user_id, 'Alt']]))
                    # ����Ŀ���û��������û����ڻ�վ֮��ļн�
                    angle = Satellite_Bs.angle_between_users(this_user_position,other_user_position,bsM_position)
                    theta_matrix[this_user_id][other_user_id]=angle
                    # ��������
                    gain = self.get_antenna_peruser_gain(angle)
                    Gain_matrix[this_user_id][other_user_id] = gain
        return Gain_matrix

    def get_bs_sinr(self, req_user_info,bs_lla,bs_state,Gain_sa_matrix,Path_loss_sa):
        """
        ��û�վ�����û����Ÿ����
        ���룺
        action:�����������
        req_user_info:�û���Ϣ
        bs_lla:��վλ����Ϣ
        bs_state:��վ״̬
        Gain_sa_matrix:�����û������棨����������ţ�
        Path_loss_sa:�����û�����ʧ(��������·����ʧ)

        """
        # print("���д��䶯��", action)
        # action �����վѡ������Щ�û�,��actionת��Ϊһ��һά�������ĵط���1����������0

        Gain_matrix = self.get_bs_gain(req_user_info,bs_lla)#��ȡ�������
        Path_loss_matrxi = self.get_bs_loss_path(req_user_info)#����ÿ���û���·�����
        Diffraction_loss_matrxi = self.get_bs_diffraction_loss(req_user_info, self.barrier_height, self.frequency)
        self.req_user_num = len(req_user_info)
        for _,bs_id_state in enumerate(bs_state):          #����ÿ����վ                                                        #i ���Ǵ�����������û��ᱻ���񣬼�������û���sinr����
            user_sa=bs_id_state["user_sa"]                      #��ǰС���������û�
            user_bs=bs_id_state["user_bs"]                      #��ǰС���Ļ�վ�����û�
            for user_bs_id in user_bs:
                interference=0
                #��վ�����û�������
                Gain_self = 10 * np.log10(Gain_matrix[user_bs_id][user_bs_id]) #dBi
                power_self = 10 ** ((Gain_self + self.Gr_user +       
                                     Path_loss_matrxi[user_bs_id] + 
                                     Diffraction_loss_matrxi[user_bs_id]) /10) * (10**(self.Power_bs/(10*len(user_bs)))) #W �������ߵ����� ������Ҫ��
                for other_user_bs_id in user_bs:
                    if user_bs_id == other_user_bs_id:  #������վ�ڵ������û�
                        continue
                    else:
                        Gain_interf = 10 * np.log10(Gain_matrix[other_user_bs_id][user_bs_id]) #dBi
                        interf = 10 ** ((Gain_interf+self.Gr_user + Path_loss_matrxi[user_bs_id]) /10) * (10 ** (self.Power_bs/(10*len(user_bs)))) #  ������վ�ڱ���վ������û��ĸ���
                        interference += interf
                for user_sa_id in user_sa:
                    
                    Gain_sa_interf = 10 * np.log10(Gain_sa_matrix[user_sa_id][user_bs_id]) #dBi  ����������վ�������û����������
                    interf = 10 ** ((Gain_sa_interf+self.Gr_user + Path_loss_sa[0][user_bs_id]) /10) * (10 ** (self.Power_Beam_average/10)) #  ������վ�ڱ����Ƿ�����û��ĸ���
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
    ����ѡ���������㴫������
    �������: Action_beam(����ѡ���û��Ķ���) req_user_info(�û���״̬��Ϣ)   req_list(�û�������id��һ������)

    """
    ## ���д���
    downlink_tool = downlink_transmision_tool()
    Gain_sa_matrix,Path_loss_sa= downlink_tool.get_sa_sinr(Action_beam, req_user_info, req_list) #��������û���SINR
    downlink_tool.get_bs_sinr( req_user_info,bs_lla,bs_state,Gain_sa_matrix,Path_loss_sa) #��û�վ�û���SINR
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

