# -*-coding:utf-8-*-
import math
import numpy as np
import math as m


class calculate_tool:
    def __init__(self):
        self.Gr_user = 40#用户接收天线的增益
        self.velocity = 3e8    #光速
        self.frequency = 2e10   # Hz
        self.beam_number = 0
        self.Hgeo = 35786000   # m  卫星高度
        self.path_loss = -209.53778  # dBw   路径损耗
        self.gama = 0.5
        self.noisy = 2.5118864315095823e-12
        self.bw = 500e6#系统总带宽
        # self.rbg_number = 6#6个资源块
        self.rb_bw = self.bw #/ self.rbg_number#每个资源块的带宽
        #发射功率
        power_action = np.random.randint(20, 21, size=19)
        self.PowerT_beam = power_action
        # self.PowerT_beam = [24.2041, 23.2720, 23.7058, 24.1472, 24.5833, 21.2876, 25.3695, 23.6879, 22.0189, 24.9629, 25.6658, 22.7529,23.7058, 24.1472, 24.5833, 21.2876]
        self.G_peak = 65 - self.PowerT_beam
        #信道增益
        # self.G_peak = [39.3776, 39.8437, 39.6268, 39.4061, 39.1880, 40.8359, 38.7949, 39.6357, 40.4702, 38.9983, 38.6468, 40.1032,38.7949,39.6357,40.4702, 38.9983]
        #1dB角
        # self.onedB = [0.4822, 0.4570, 0.4686, 0.4807, 0.4929, 0.4077, 0.5157, 0.4681, 0.4252, 0.5038, 0.5246, 0.4436]

    def get__PowerT_beam1(self,clusters_xyz):
        # print(clusters_xyz)
        # input()
        three_db=[]
        one_db=[]
        G_peak_db=[]
        for i in range(len(clusters_xyz)):
            # threedb = m.degrees(m.atan(clusters_xyz[i] / self.Hgeo)*2)
            threedb = (m.degrees(m.atan(clusters_xyz[i] / self.Hgeo)*2))*4
            # onedb=threedb.copy
            # G_peak=10*math.log10((70 * math.pi / threedb)**2) - 10* math.log10(self.gama)
            G_peak= 10 * math.log10((70 * math.pi / threedb)**2) - 10 * math.log10(2)
            three_db.append(threedb)
            G_peak_db.append(G_peak)
            #onedb = 70 * m.pi * m.sqrt(0.5 / (12 * m.pow(10, G_peak / 10)))
            # one_db.append(onedb)     
        self.G_peak=G_peak_db

        # print("three_db",three_db)
        # print("self.G_peak",self.G_peak)
        # print("self.onedB",self.onedB)
        # input()
        # self.onedB=one_db
        # self.onedB=three_db.copy()
        # self.PowerT_beam = 65 -np.array(self.G_peak)
        self.PowerT_beam = np.random.randint(20, 21, size=19)
        # print(f"self.power_beam:{self.PowerT_beam}  self.Gpeak : {self.G_peak}")
        # input()
        # print("three_db",three_db)
        # print("self.G_peak",self.G_peak)
        # print("self.onedB",self.onedB)
        # input()
        array=three_db.copy()
        factor = 0.35  # 使用浮点数
        result = [x * factor for x in array]
        one_db = np.asarray(result)
        # print(three_db,one_db)
        # input()
        return three_db,one_db
    #路径损耗参数
    def get_beam_loss_path(self,center_xyz):
        self.beam_number = len(center_xyz)
        LOSS_PATH = np.zeros(self.beam_number)
        for i in range(self.beam_number):
            beam_position = center_xyz[i]
            distance_beam_statecenter = np.sqrt(np.sum((beam_position) ** 2))
            distance_beam_state = np.sqrt(distance_beam_statecenter ** 2 + self.Hgeo ** 2)
            loss_path = ((4 * m.pi * self.frequency * distance_beam_state) / self.velocity) ** (-2)    # W
            Loss_path = 10 * np.log10(loss_path)   # dBw
            LOSS_PATH[i] = Loss_path
        return LOSS_PATH
    
    #卫星发射天线的增益
    #self.G_peak：
    def get_beam_gain(self, center_xyz):
        self.beam_number = len(center_xyz)
        theta_matrix = np.zeros((self.beam_number,self.beam_number))
        Gain_matrix = np.zeros((self.beam_number,self.beam_number))
        for i in range(self.beam_number):
            beam_position_i = center_xyz[i]
            # distance_i_statecenter = np.sqrt(np.sum((beam_position_i) ** 2))
            # distance_i_state = np.sqrt(distance_i_statecenter ** 2 + self.Hgeo ** 2)
            for j in range(self.beam_number):
                if i==j:
                    Gain_matrix[i][j]=self.G_peak[i]
                else:
                    beam_position_j = center_xyz[j]
                    distance = np.sqrt(np.sum((beam_position_i - beam_position_j) ** 2))
                    # distance_j_statecenter = np.sqrt(np.sum((beam_position_j) ** 2))
                    # distance_j_state = np.sqrt(distance_j_statecenter ** 2 + self.Hgeo ** 2)
                    # distance_i_j = np.sqrt(np.sum(( beam_position_i - beam_position_j) ** 2))
                    # theta = math.acos((distance_i_state**2 + distance_j_state**2 - distance_i_j**2)/(2*distance_i_state*distance_j_state))#角度计算
                    #distance_matrix[i][j] = distance
                    #theta = np.degrees(np.arctan(distance / self.Hgeo))
                    theta = np.degrees(np.arctan(distance / self.Hgeo))
                    theta_matrix[i][j]= theta
                    #根据方向性调整波束增益
                    Gain_matrix[i][j]= self.G_peak[j] - ((12 * (10 ** (self.G_peak[j] / 10))) / self.gama) * np.square(theta_matrix[i][j] / (70 * np.pi))#
        Gain_matrix = 10 ** (Gain_matrix / 10)
        return Gain_matrix


    #每个波束在连接时的sinr
    def get_beam_sinr(self, action, position_info):
        self.beam_number = len(position_info)
        sinr_matrix = np.zeros(self.beam_number)
        Gain_matrix = self.get_beam_gain(position_info)
        Path_loss_matrxi = self.get_beam_loss_path(position_info)
        interference=0
        for i in range(self.beam_number):
            if(action[i] == 0):
                continue
            else:
                Gain_self = 10 * np.log10(Gain_matrix[i][i])
                power_self = 10 ** ((Gain_self + self.Gr_user + Path_loss_matrxi[i]) / 10) * (10 ** (self.PowerT_beam[i]/10))
                for j in range(self.beam_number):
                    if i == j or action[j] == 0:
                        continue
                    else:
                        Gain_interf = 10 * np.log10(Gain_matrix[j][i])
                        interf = 10 ** ((Gain_interf) / 10) * (10 ** (self.PowerT_beam[j] / 10))
                        # print(" i 对 k的ganrao",i,j,interf)
                        interference += interf
                interference = 10 ** ((self.Gr_user + Path_loss_matrxi[i]) / 10) * interference
                #print("总干扰：",interference)
                sinr = power_self / (self.noisy + interference)
                sinr_matrix[i] = sinr
        return sinr_matrix



def get_tb(action,position_info,clusters_xyz):
    tool = calculate_tool()
    tool.get__PowerT_beam1(clusters_xyz)
    # 有请求用户的位置信息,position_info,(用户编号,用户位置,波束位置,星下点位置)
    sinr = tool.get_beam_sinr(action, position_info)
    capacity = np.log2(sinr + 1) * tool.rb_bw / 1000
    # print(capacity)

    return capacity


def get_capacity(beam_num,clusters_xyz):
    tool = calculate_tool()
    tool.get__PowerT_beam1(clusters_xyz)
    cap_label = []
    for i in range(beam_num):
        power_self = 10 ** ((tool.G_peak[i] + tool.Gr_user + tool.path_loss) / 10) * (
                    10 ** (tool.PowerT_beam[i] / 10))
        sinr = power_self / tool.noisy
        cap = np.log2(sinr + 1) * tool.bw / 1000
        cap_label.append(cap)
    return cap_label


if __name__ == '__main__':
    center_xyz=np.loadtxt("./all_data/center-shy.txt",usecols=(1,2,3))
    action=[1,1,0,1,1,0,1,1,1,0,1,1,0,0,0,1]
    action = np.array(action).reshape(-1, 1)
    tool = calculate_tool()
    a, b=tool.get_beam_sinr(action,center_xyz,0)
