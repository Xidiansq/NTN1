import numpy as np
import pandas as pd
import math as m
import random
import Tool_Calculate
from k_means import KMeans
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.patches import Circle
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import Parameters
#from sklearn.cluster import KMeans


class user:
    def __init__(self, center_sat, cover_range, bs_list, sate_xyz,sate_lla, ontime, offtime):
        self.center_sat = center_sat #用户区域中心，用于设置用户分布[33.935, 108.445]
        self.cover_range = cover_range #距离用户区域中心的最大距离， 用于设置用户分布 500 *1000 m

        self.sate_xyz = sate_xyz #卫星的xyz坐标
        self.sate_lla = sate_lla #卫星的经纬度坐标

        self.bs_list = bs_list #所有地面基站的坐标信息， 用于确定用户所属基站id
        self.earth_radius = Parameters.R_earth #地球半径
        self.earthspheroidtype = {'SPHERE': 0, 'GRS80': 1, 'WGS84': 2}  # 三种地球模型
        self.angle_user2sate = 0 # 用户的仰角
        self.angle_user2bs = 0 # 用户和基站天线中心的夹角
        self.ontime = 0    # 业务持续时间
        self.offtime = 0   # 业务关闭时间
        self.qci = 0 #业务类型

        self.cbrrate = Parameters.cbrrate    # 固定码率0.5Mb, 可修改为可变码率VBR;
        self.current_txdata = 0   # 当前的传输数据大小
        self.current_newdata = 0   # 当前的新到数据大小
        self.current_begin_wait = 0 #当前时隙开始的时候等待传输数据的大小

        self.init_datasize = []
        self.achieved_datasize = []
        self.unachieved_datasize = []
        self.consume_dalay = []
        self.final_dalay = []
        self.Action_Offload = []
        self.Action_Computation = []
        self.Action_Communication = []

        self.init_user_position(self.center_sat, self.cover_range, self.bs_list, self.sate_lla)#初始化用户的位置
        self.bs_if_serv = 0 # TODO:添加了bs_if_serv
        self.init_user_traffic(ontime, offtime, self.cbrrate)    # 初始化用户业务类信息


    def init_user_position(self, center, maxdistance, bs_list, sate_lla):
        """
        初始化用户位置类信息:
        (1) 初始化用户位置, 返回用户位置[x, y, z]和[lat, lon, alt];
        (2) 根据用户位置，计算用户所属基站的范围;
        (3) 根据用户位置与卫星位置, 计算用户仰角;
        """
        self.ue_xyz, self.ue_lat_lon_alt = self.generate_position(center, maxdistance)
        self.bs_of_ID, self.bs_of_xyz, self.distance_ue2bs = self.generate_connected_bs(self.ue_xyz, bs_list)
        #self.angle_user2sate = self.generate_angele(sate_lla, self.ue_xyz)
        self.angle_user2sate = Tool_Calculate.get_elevation_angle_geocentric(self.ue_xyz,Parameters.sate_xyz)
        

    def generate_position(self, center, maxdistance):
        """
        位置类函数1
        初始化用户位置, 方法: 以region_center为用户区域中心, 随机在maxdistance范围内产生经纬度坐标;
        输入参数: 用户区域中心位置, [lat, lon, alt]; 用户区域半径, maxdistance;
        返回: 用户位置, [x, y, z]和[lat, lon, alt];
        """
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
        self.ue_xyz = Tool_Calculate.GeographicToCartesianCoordinates(ue_lat, ue_lon, ue_alt, self.earthspheroidtype['GRS80'])   # 用户位置, [x, y, z]
        self.ue_lat_lon_alt = [ue_lat, ue_lon, ue_alt]                                                                             # 用户位置, [lat, lon, alt]
        center_xyz = Tool_Calculate.GeographicToCartesianCoordinates(center[0], center[1], center[2], self.earthspheroidtype['GRS80'])
        return self.ue_xyz, self.ue_lat_lon_alt
    

    def generate_connected_bs(self, ue_position, all_bs):
        """
        位置类函数2
        计算用户所属基站的范围, 方法: 遍历所有基站中心位置, 选择与用户位置最接近的基站确定接入;
        输入参数: ue_position为用户位置, all_bs_position为所有基站xyz坐标及id, all_beam_ID为所有波束的ID;
        返回: bs_ID所属基站范围的id, bs_connect_xyz所属基站的xyz坐标, distance_ue2bs用户距离基站的距离;
        """
        all_bs_position = all_bs[:, 1:]
        distance_max = np.inf
        bs_ID = 0
        bs_connect_xyz = [0, 0, 0]
        for i in range(len(all_bs_position)):
            distance = np.sqrt(np.sum((ue_position-all_bs_position[i])**2))   
            if distance < distance_max:
                distance_max = distance
                bs_connect_xyz = all_bs_position[i]
                bs_ID = all_bs[i][0]    # 所属基站id
                distance_ue2bs = distance
        return bs_ID, bs_connect_xyz, distance_ue2bs


    def generate_angele(self, sate_lla, user_position):
        """
        位置类函数3
        计算卫星-用户与卫星—波束中心之间的夹角, 以及用户仰角;
        输入参数: sate_lla 卫星的经纬度 user_position为用户位置[x, y, z];
        返回: self.angle_user2sate用户仰角, 单位: 度°;
        """
        sate_position_latlonalt = sate_lla   # 所接入卫星的位置

        satepoint_position_latlonalt = [sate_position_latlonalt[0], sate_position_latlonalt[1], 0] #卫星星下点的经纬度

        sate_position = Tool_Calculate.GeographicToCartesianCoordinates(sate_position_latlonalt[0], 
                                                                        sate_position_latlonalt[1], 
                                                                        sate_position_latlonalt[2], 
                                                                        self.earthspheroidtype['GRS80'])# 卫星位置, [x, y, z];
        point_position = Tool_Calculate.GeographicToCartesianCoordinates(satepoint_position_latlonalt[0], 
                                                                         satepoint_position_latlonalt[1], 
                                                                         satepoint_position_latlonalt[2], 
                                                                         self.earthspheroidtype['GRS80'])  # 卫星星下点位置, [x, y, z];
        sate2point = m.sqrt((sate_position[0]-point_position[0])**2+
                            (sate_position[1]-point_position[1])**2+
                            (sate_position[2]-point_position[2])**2)   # 卫星到星下点的距离;
        user2point = m.sqrt((user_position[0]-point_position[0])**2+
                            (user_position[1]-point_position[1])**2+
                            (user_position[2]-point_position[2])**2)   # 用户到星下点的距离;

        tan_angle_user2sate = m.atan(sate2point / user2point)
        angle_user2sate = m.degrees(tan_angle_user2sate)     # 夹角, 度数
        return angle_user2sate
            
            
        return angle
    def init_user_traffic(self, ontime, offtime, cbrrate):
        """
        初始化用户业务信息;
        (1) 初始化用户业务类型, 以及业务持续时间; 
        (2) 初始化用户的其他业务信息;
        """
        self.generate_traffic_duration(ontime, offtime)
        self.generate_traffic_info(cbrrate)

    def generate_traffic_duration(self, init_ontime, init_offtime):
        """
        业务类函数1
        初始化用户业务类型, 以及业务持续时间;
        方法：随机选择三种业务并按照指数分布随机产生业务的持续时间;
        输入参数: 业务持续时间分布均值, 业务关闭时间分布均值;
        返回: 采样后的持续时间, 采样后的关闭时间；
        """
        self.traffictype = {'text': init_ontime/4, 'voice': init_ontime/2, 'video': init_ontime}  # 业务类型指数分布均值参数
        self.qci_type = {'None': 0, 'text': 1, 'voice': 2, 'video': 3}
        self.offtime = np.random.exponential(init_offtime)
        if self.offtime > 0:
            self.offtime -= 1
            if self.offtime < 0:
                self.offtime = 0
                traffic_choice = np.random.choice([1, 2, 3])
                if traffic_choice == 1:   # 文本类型
                    self.ontime = np.random.exponential(self.traffictype['text'])
                    self.qci = self.qci_type['text']
                elif traffic_choice == 2:  # 音频类型
                    self.ontime = np.random.exponential(self.traffictype['voice'])
                    self.qci = self.qci_type['voice']
                else:                    # 视频类型
                    self.ontime = np.random.exponential(self.traffictype['video'])
                    self.qci = self.qci_type['video']
        elif self.offtime == 0 and self.ontime > 0:
            self.ontime -= 1
            if self.ontime < 0:
                self.ontime = 0
                self.offtime = np.random.exponential(init_offtime)
                self.qci = self.qci_type['None']
        return self.offtime, self.ontime
    
    def generate_traffic_info(self, cbrrate):
        """
        业务类函数2
        初始化用户的其他业务信息;
        包含: 当前的传输数据大小, 新到数据大小, 时隙开始的等待传输数据大小, 时隙结束的等待传输数据大小, 
              所需RBG数量, 是否有请求的标识, 时延, 总计传输数据大小, 瞬时吞吐, 平均吞吐
        输入参数: 固定码率CBR, 可修改为可变码率VBR;
        """
        # 上一时隙末的信息
        self.last_final_wait = 0     # 上次时隙结束的等待传输数据大小
        # 当前时隙的信息
        self.current_finish_data = 0   # 当前的完成数据大小
        #self.ontime = random.uniform(0.5, 1)
        self.current_newdata = cbrrate * 1 if self.ontime > 1 else cbrrate * self.ontime   # 当前的新到数据大小
        self.current_begin_wait = self.last_final_wait + self.current_newdata    # 当前时隙开始的等待传输数据大小
        if self.current_begin_wait > 0 and self.angle_user2sate >= Parameters.Elevation_Angle:# 当前时隙是否有请求, 判断用户是否在卫星覆盖范围的最小仰角内；
            self.current_request = 1
        else:
            self.current_request = 0
        self.current_time_delay = 0  # 当前的时延
        self.current_down_txdata = 0

        self.current_downlink_throughput =((self.current_down_txdata/0.001))/(1024**2)  # 当前的下行链路瞬时吞吐, 单位Mbps
        


#以上为user类，都是类内函数
#以上为user类，都是类内函数
#以上为user类，都是类内函数
#以上为user类，都是类内函数
#以上为user类，都是类内函数
def initial_userAll(center_sat,cover_range, bs_list, user_num, sate_xyz, sate_lla, ontime, offtime):
    """
    初始化用户，包含用户数量、用户的位置
    输入参数：
    center_sat:覆盖区域中心经纬度
    cover_range:距离用户区域中心的最大距离, 用于设置用户分布,单位 m
    user_num: 用户的数量
    bs_list: 地面所有基站的xyz坐标和编号
    sate_xyz: 卫星的笛卡尔坐标
    sate_lla: 卫星的经纬度坐标
    ontime: 业务持续时间
    offtime:业务关闭时间
    返回：
    user_list:用户列表
    """
    userlist = [user(center_sat,cover_range,bs_list, sate_xyz, sate_lla, ontime, offtime) for i in range(user_num)]
    return userlist

def initial_userTask_info(userlist):
    """
    获取初始化后的用户的任务信息;
    输入参数： 用户列表
    输出参数： 用户的信息
    """
    task_info = []
    for i in range(len(userlist)):
        task_info.append((i, userlist[i].init_datasize, userlist[i].unachieved_datasize, userlist[i].achieved_datasize, userlist[i].consume_dalay, userlist[i].final_dalay,
                          userlist[i].Action_Offload, userlist[i].Action_Computation, userlist[i].Action_Communication))
    task_info = pd.DataFrame(task_info, columns=['UserID', 'Init_DataSize', 'UnAchieved_DataSize','Achieved_DataSize', 'Consume_Delay', 'Final_Delay',
                                                 'Action_Offload', 'Action_Computation', 'Action_Communication'])
    return task_info

def initial_userTraffic_info(userlist):
    """
    获取初始化后的所有用户信息; 
    输入: 用户列表
    输出:
    'UseID'用户ID, 'Lat'用户位置(维度), 'Lon'用户位置(经度), 'Alt'用户位置(海拔), 'Angle'用户夹角, 'WaitData'等待传输数据, 'NewData'新到达数据,  'ReqID'是否有请求, 'Last_TxData'上个时隙传输, 
    'Time_Delay'时延, 'QCI'业务类型, 'RBG_Needed'所需RBG数目, 'Total_TxData'总计传输, 'Throughput(mbps)'瞬时吞吐, 'Aver_throughput'平均吞吐, 'Capacity'单个RBG传输上限
    """
    user_request = []
    traffic_info = []
    for i in range(len(userlist)):
        if userlist[i].current_request == 1:
            user_request.append(i)#只记录有请求信息的用户 # TODO：添加了'BsIfServ':bs_if_serv
        traffic_info.append((i, userlist[i].bs_of_ID, userlist[i].bs_if_serv, userlist[i].ue_lat_lon_alt[0], userlist[i].ue_lat_lon_alt[1], userlist[i].ue_lat_lon_alt[2], userlist[i].angle_user2sate, userlist[i].distance_ue2bs, 
                            userlist[i].current_request, userlist[i].qci, 
                            userlist[i].current_newdata, userlist[i].last_final_wait, userlist[i].current_begin_wait, userlist[i].current_finish_data, 
                            userlist[i].current_time_delay, userlist[i].current_down_txdata, userlist[i].current_downlink_throughput))
    traffic_info = np.array(traffic_info, dtype = 'float64')
    traffic_info = pd.DataFrame(traffic_info, columns=['UserID', 'BsID', 'BsIfServ', 'Lat', 'Lon', 'Alt', 'Angle', 'Dis_Bs', 
                                                       'ReqID', 'QCI', 
                                                       'NewData', 'Last_WaitData', 'Total_WaitData', 'Finish_Data', 
                                                       'Time_Delay', 'Down_TxData', 'Down_Throughput'])
    return traffic_info, user_request

# 获取全部用户的位置和编号
def get_alluser_position(user):
    alluserlist = user
    alluser_position_XYZ = []
    alluser_position_log_lat_coordinates = []
    # 随机选择len(index)个用户来产生业务，len(index)服从泊松分布
    for i in range(len(alluserlist)):
        position = alluserlist[i].areauser_position_xyz
        position2 = alluserlist[i].lat_long_user
        alluser_position_XYZ.append(position)  # 只保留位置信息
        alluser_position_log_lat_coordinates.append(position2)
    return alluser_position_XYZ, alluser_position_log_lat_coordinates

# 获取发起业务请求用户的位置和编号
def get_user_position(user):
    # 初始化用户和泊松分布均值
    userlist = user
    user_position_XYZ = []
    user_position_log_lat_coordinates = []
    # 随机选择len(index)个用户来产生业务，len(index)服从泊松分布
    for i in range(len(userlist)):
        if userlist[i].request == 1:
            position = userlist[i].areauser_position_xyz
            position2 = userlist[i].lat_long_user
            user_position_XYZ.append(position)  # 只保留位置信息
            user_position_log_lat_coordinates.append(position2)
    return user_position_XYZ, user_position_log_lat_coordinates


#
def get_user_traffic_info(user):
    userlist = user
    user_request = []
    traffic_info = []
    for i in range(len(userlist)):
        if userlist[i].request == 1:
            user_request.append(i)
        traffic_info.append(
            (i, userlist[i].lat_long_user[0], userlist[i].lat_long_user[1], userlist[i].lat_long_user[2],
             userlist[i].angle_user, userlist[i].current_waiting_data, userlist[i].newarrivaldata, userlist[i].request,
             userlist[i].current_txdata, userlist[i].time_delay, userlist[i].qci,
             userlist[i].number_of_rbg_nedded, userlist[i].total_txdata, 
             userlist[i].throughput, userlist[i].average_throughput,
             userlist[i].capacity))
    traffic_info = np.array(traffic_info, dtype='float')
    #纬度  经度  高度'Lat', 'Lon' , 'Alt', 
    traffic_info = pd.DataFrame(traffic_info,
                                columns=['Use_ID', 'Lat', 'Lon', 'Alt',
                                          'Angle', 'waitingdata', 'newdata', 'Req',
                                         'last_time_txdata', 'time_delay', 'CQI', 
                                         'number_of_rbg_nedded', 'total_txdata',
                                         'throughput(mbps)', 'average_throughput',
                                        'capacity'])
    return traffic_info, user_request
def get_distance(userlist,center_xyz_list,lat_log):
    max=0
    id_list=get_user_beam_id(userlist)
    counts = {i: 0 for i in range(19)}  # 创建一个字典，键为0到19，值初始化为0
    for row in id_list:
        if row in counts:
            counts[row] += 1  # 对应键的值加1
    # print(counts)
    radiu=[]
    max_lat_long_list=[]
    start_idx = 0
    for i in range(len(center_xyz_list)):
        max=0
        max_lat_long=0
        # print("start_idx",start_idx)
        end_idx = start_idx + counts[i]
        
        # 获取对应的center和随机数范围
        for j in range(start_idx, end_idx):
            distance = np.sqrt((userlist[j].areauser_position_xyz[0] - center_xyz_list[i][0]) ** 2+
                                   (userlist[j].areauser_position_xyz[1] - center_xyz_list[i][1]) ** 2+
                                   (userlist[j].areauser_position_xyz[2] - center_xyz_list[i][2]) ** 2)
            distance_lat_long=np.sqrt((userlist[j].lat_long_user[0] - lat_log[i][1]) ** 2+
                                   (userlist[j].lat_long_user[1] - lat_log[i][2]) ** 2+
                                   (0 - 0) ** 2)
            # print("distance_lat_long",distance_lat_long)
            if max<distance:
                max=distance
            if max_lat_long<distance_lat_long:
                max_lat_long=distance_lat_long
        radiu.append(max)
        max_lat_long_list.append(max_lat_long)
        # for m in range(start_idx, end_idx):
        #     userlist[m].gain_radiu=max
        start_idx = end_idx 
    radiu=np.array(radiu)
    max_lat_long_list=np.array(max_lat_long_list)
    return radiu,max_lat_long_list
    

def get_all_user_position(user):
    # 初始化用户和泊松分布均值
    userlist = user
    user_position_XYZ = []
    user_position_log_lat_coordinates = []
    # 随机选择len(index)个用户来产生业务，len(index)服从泊松分布
    for i in range(len(userlist)):
        position = userlist[i].areauser_position_xyz
        position2 = userlist[i].lat_long_user

        user_position_XYZ.append(position)  # 只保留位置信息
        user_position_log_lat_coordinates.append(position2)
    return user_position_XYZ, user_position_log_lat_coordinates

def get_all_user_position_and_request(user):
    userlist = user
    position_and_req = []
    for i in range(len(userlist)):
        position = userlist[i].areauser_position_xyz.tolist()
        position_and_req.append((i, position[0], position[1], position[2], userlist[i].request))
    position_and_req = np.array(position_and_req, dtype='float')
    return position_and_req

# def get_all_center_lat_long(center):
#     center_lat_long=[]
#     for i in range(len(center)):
#         lat_long=center[i]
#         center_lat_long.append(lat_long)
#     return center_lat_long

#获取用户连接波束
def get_user_beam_id(user):
    userlist = user
    beam_id_list = []
    for i in range(len(userlist)):
        beam_id = userlist[i].beam_id
        beam_id_list.append(beam_id)
    return beam_id_list

def calculate_beam_State(S0):
    beam_S0 = S0.groupby('beam_id').agg({
        'Req':'sum',
        'waitingdata':'sum',#current_waiting_data
        'newdata':'sum',#newarrivaldata
        'last_time_txdata':'sum',#current_txdata
        'total_txdata':'sum',#total_txdata
        'time_delay':'sum',#current_txdata
    }).reset_index()
    for i in range(len(beam_S0)):
        if beam_S0['Req'].iat[i]==0:
            beam_S0['Req'].iat[i]=0
        else:
            beam_S0['Req'].iat[i]=1
    return beam_S0

#修改队列信息
def process_queue(q):
    # 将deque转换为列表
        lst = list(q)
        
        # 检查列表长度，如果少于20，则添加0直到长度为20
        while len(lst) < 20:
            lst.append(0)
        
        # 将列表转换为NumPy数组并返回
        return np.array(lst)

#获取波束中心的坐标
def get_center_user_xyz(center_list):
    for i in range(len(center_list)):
        center_list[i] = GeographicTocartesianCoordinate(center_list[i][0],center_list[i][1],center_list[i][2],"SPHERE")
    return center_list
#maxdistance,num_users,beam_lat_log


#k_means聚类效果不好  修改了分布
def initial_user(beam_radius_meters,num_users,nums_beams,beam_lat_log):
    gain_radiu_list=[]
    data=[]
    beam_lat_log = beam_lat_log[:,1:3]
    for i, center in enumerate(beam_lat_log):
        if i in [0,1,2,3,4,5,6]:  # 指定要随机分布的波束序号
            for _ in range(num_users):
                user_coord = np.random.uniform(low=center -1 , high=center + 1, size=2)  # 生成随机坐标
                while np.linalg.norm(user_coord - center) > beam_radius_meters:
                    user_coord = np.random.uniform(low=center -1, high=center + 1, size=2)  # 确保在圆内
                data.append(np.append(user_coord, 0))  # 添加分布类型列
                gain_radiu=240000
                gain_radiu_list.append(gain_radiu)
        else:
            for _ in range(num_users):
                user_coord = np.random.normal(loc=center, scale=0.2, size=2)  # 正态分布坐标
                while np.linalg.norm(user_coord - center) > beam_radius_meters:
                    user_coord = np.random.normal(loc=center, scale=0.2, size=2)  # 确保在圆内
                data.append(np.append(user_coord, 1))  # 添加分布类型列
                gain_radiu=180000
                gain_radiu_list.append(gain_radiu)
    user_lat_log = pd.DataFrame(data,columns=['lat','lng','id']).drop(columns='id')

    clusters,data_with_clusters = K_means_user(user_lat_log,beam_lat_log,nums_beams)
    # print(data)
    # print(data_with_clusters)
    # data_with_clusters=np.array(data.copy())
    # for i in range(len(data_with_clusters)):
    #     data_with_clusters[i][2]=i//num_users
    # print("aaaaa",data_with_clusters)
    # input()
    # print("beam_lat_log",beam_lat_log)
    # input()
    # print("user_lat_log",user_lat_log)
    # print("clusters",clusters)
    # print("data_with_clusters",data_with_clusters)
    # input()
    # cString = ['#005EAD', '#AF6DE5', '#719FFB', '#1CAC99', '#FE9499', '#4A8FDE', '#F8A13E', 
    #        '#4DE890', '#2178B8', '#77A2E8', '#F86067', '#FBBF72', '#FA9B97', '#30A02D',
    #        '#361D32', '#543C52', '#F65A53', '#EED2CB', '#DBD873', '#F1E8E8']
    # plt.scatter(user_lat_log['lat'], user_lat_log['lng'],label='user distribution point')
    # plt.xlabel('latitude')
    # plt.ylabel('longitude')
    # plt.style.use('seaborn')
    # plt.legend(bbox_to_anchor=(0.62, 1.08), loc='upper left')
    # #plt.title('Initial User Distribution Plot')
    # plt.savefig('./all_data/User_Distribution.jpg')
    # for i, cluster_mean in enumerate(beam_lat_log):
    #     data_cluster_i = data_with_clusters[ data_with_clusters[:, -1] == i ]
    #     plt.scatter(data_cluster_i[:, 0], data_cluster_i[:, 1],color=cString[i])
    #     #plt.plot(cluster_mean[0], cluster_mean[1], label = 'C' + str(i), marker='*', markersize=15, markeredgecolor="k", markeredgewidth=1,color=cString[i])
    #     plt.plot(cluster_mean[0], cluster_mean[1], marker='*', markersize=15, markeredgecolor="k", markeredgewidth=1,color=cString[i])
    #     plt.xlabel('latitude')
    #     plt.ylabel('longitude')
    #     plt.style.use('seaborn')
    #     #plt.title('User Clustering Distribution Plot')
    #     legend_elements = [plt.Line2D([0], [0], marker='*', color='none', markersize=15, markeredgecolor="k", markeredgewidth=1, label='Cluster center')]
    #     plt.legend(handles=legend_elements, bbox_to_anchor=(0.67, 1.08), loc='upper left', facecolor='none', edgecolor='none')  # 添加图例
    #     # legend_elements = [plt.Line2D([0], [0], marker='o', color='none', markersize=15, markeredgecolor="k", markeredgewidth=1, label='user distribution point')]
    #     # plt.legend(handles=legend_elements, bbox_to_anchor=(0.62, 1.15), loc='upper left', facecolor='none', edgecolor='none')  # 添加图例
    #     plt.savefig('./all_data/beam_kmeans.jpg')
    # fig = plt.figure(figsize=(10, 6))
    # ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    # # 绘制局部放大地图
    # plot_zoomed_map(ax, (-3.6683936558045714e-16,0.0019151850530515673), zoom_factor=10)  # 根据需要调整中心点和缩放因子
    # # 绘制波束覆盖范围和用户位置
    # # 循环处理每个波束中心
    # for beam_center_latlon in beam_lat_log:
    #     plot_beam_coverage(beam_center_latlon, beam_radius_meters, data, ax)
    # plt.show()
    # plt.savefig('./all_data/guss.jpg')
    # print("clusters:",beam_lat_log)
    # print("data_with_clusters",data_with_clusters)
    # input()
    return clusters,data_with_clusters,gain_radiu_list

# def step_initial_user(userlist):
#     pd.set_option('display.max_rows', None)
#     data=userlist
#     user_lat_log = pd.DataFrame(data,columns=['lat','lng','id']).drop(columns='id')
#     clusters,data_with_clusters= K_means_user(user_lat_log)
#     return clusters,data_with_clusters

def K_means_user(user_lat_log,beam_lat_log,num_beams):
    model = KMeans(max_iter = 500, tolerance = 0.001, n_clusters =num_beams, runs = 100,cluster_means = beam_lat_log)
    (clusters, data_with_clusters) = model.fit(user_lat_log)
    data_with_clusters = np.array((sorted(data_with_clusters, key=lambda x: x[2])))
    return clusters,data_with_clusters
# def K_means_user(user_lat_log):
#     model = KMeans(max_iter=500, tol=0.001, n_clusters=19, n_init=100)
#     model.fit(user_lat_log)
#     clusters = model.labels_
#     cluster_centers = model.cluster_centers_
#     data_with_clusters = np.hstack((user_lat_log, clusters.reshape(-1, 1)))
#     data_with_clusters = np.array(sorted(data_with_clusters, key=lambda x: x[2]))
#     return cluster_centers, data_with_clusters

def GeographicTocartesianCoordinate1(latitude, longitude, altitude, sphType):
    latitudeRadians = latitude * m.pi / 180
    longitudeRadians = longitude * m.pi / 180
    # a: semi - major axis of earth
    # e: first eccentricity of earth
    EARTH_RADIUS = 6371e3
    EARTH_GRS80_ECCENTRICITY = 0.0818191910428158
    EARTH_WGS84_ECCENTRICITY = 0.0818191908426215
    EARTH_SEMIMAJOR_AXIS = 6378137
    EARTH_SEMIMAJOR_BXIS = 6356752.3142451793
    if sphType == "SPHERE":
        a = EARTH_RADIUS
        e = 0
    if sphType == "GRS80":
        a = EARTH_SEMIMAJOR_AXIS
        e = EARTH_GRS80_ECCENTRICITY
    else:  # if sphType == WGS84
        a = EARTH_SEMIMAJOR_AXIS
        e = EARTH_WGS84_ECCENTRICITY
    Rn = a / (m.sqrt(1 - pow(e, 2) * pow(m.sin(latitudeRadians), 2)))  # radius of  curvature
    x = (Rn + altitude) * m.cos(latitudeRadians) * m.cos(longitudeRadians)
    y = (Rn + altitude) * m.cos(latitudeRadians) * m.sin(longitudeRadians)
    z = (Rn + altitude) * m.sin(latitudeRadians)
    cartesianCoordinates = np.array([x, y, z], dtype='float64')
    return cartesianCoordinates
#绘制带经纬度格波束半径的图
def plot_zoomed_map(ax, center_latlon, zoom_factor):
    lat_center, lon_center = center_latlon
    ax.set_extent([lon_center - zoom_factor, lon_center + zoom_factor,
                   lat_center - zoom_factor, lat_center + zoom_factor], crs=ccrs.PlateCarree())
    ax.coastlines()
    # 添加经纬度网格线和标签
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linestyle='--')
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    gl.xlabels_top = False
    gl.ylabels_right = False


def plot_beam_coverage(beam_center_latlon, radius_meters, user_coords, ax):
    lat_center, lon_center = beam_center_latlon

    # 计算波束覆盖范围上的点
    circle_points = []
    for azimuth in range(0, 360, 10):
        lat = np.degrees(np.arcsin(np.sin(np.radians(lat_center)) * np.cos(radius_meters / 6371000) +
                                    np.cos(np.radians(lat_center)) * np.sin(radius_meters / 6371000) * np.cos(np.radians(azimuth))))
        lon = np.degrees(np.radians(lon_center) + np.arctan2(np.sin(np.radians(azimuth)) * np.sin(radius_meters / 6371000) * np.cos(np.radians(lat_center)),
                                                             np.cos(radius_meters / 6371000) - np.sin(np.radians(lat_center)) * np.sin(np.radians(lat))))
        circle_points.append((lat, lon))

    circle_lat, circle_lon = zip(*circle_points)

    # 绘制波束覆盖范围
    ax.plot(circle_lon, circle_lat, 'r-', transform=ccrs.PlateCarree(), alpha=0.3)

    # 绘制波束中心点
    ax.plot(lon_center, lat_center, 'ro', markersize=10, transform=ccrs.PlateCarree())

    # 绘制用户位置
    for user_coord in user_coords:
        if user_coord[2] == 0:  # 随机分布用户
            ax.plot(user_coord[1], user_coord[0], 'go', markersize=5, transform=ccrs.PlateCarree(),label='Random Users')
        else:  # 高斯分布用户
            ax.plot(user_coord[1], user_coord[0], 'bo', markersize=5, transform=ccrs.PlateCarree(),label='Gaussian Users')
    ax.legend(['Beam Coverage', 'Beam Center'], bbox_to_anchor=(1, 1),loc='upper right')
    ax.set_title('Beam Coverage and User Distribution (Zoomed)')

#LAT LONG ALTI------XYZ
def GeographicTocartesianCoordinate(latitude, longitude, altitude, sphType):
    latitudeRadians = latitude * m.pi / 180
    longitudeRadians = longitude * m.pi / 180
    # a: semi - major axis of earth
    # e: first eccentricity of earth
    EARTH_RADIUS = 6371e3
    EARTH_GRS80_ECCENTRICITY = 0.0818191910428158
    EARTH_WGS84_ECCENTRICITY = 0.0818191908426215
    EARTH_SEMIMAJOR_AXIS = 6378137
    EARTH_SEMIMAJOR_BXIS = 6356752.3142451793
    if sphType == "SPHERE":
        a = EARTH_RADIUS
        e = 0
    if sphType == "GRS80":
        a = EARTH_SEMIMAJOR_AXIS
        e = EARTH_GRS80_ECCENTRICITY
    else:  # if sphType == WGS84
        a = EARTH_SEMIMAJOR_AXIS
        e = EARTH_WGS84_ECCENTRICITY
    Rn = a / (m.sqrt(1 - pow(e, 2) * pow(m.sin(latitudeRadians), 2)))  # radius of  curvature
    x = (Rn + altitude) * m.cos(latitudeRadians) * m.cos(longitudeRadians)
    y = (Rn + altitude) * m.cos(latitudeRadians) * m.sin(longitudeRadians)
    z = (Rn + altitude) * m.sin(latitudeRadians)
    cartesianCoordinates = np.array([x, y, z], dtype='float64')
    return cartesianCoordinates

def trafficduration_beam(user):
    type = 'None'
    ontime_list=[]
    offtime_list=[]
    for i in range(len(user)):
        if user[i].offtime > 0:
            user[i].offtime -= 1
            if user[i].offtime < 0:
                user[i].offtime = 0
                ################
                traffic_choice = np.random.choice([1, 2, 3])
                if traffic_choice == 1:
                    user[i].ontime = np.random.exponential(user[i].traffictype['text'])
                    type = 'text'
                    user[i].qci = user[i].qci_type[type]
                elif traffic_choice == 2:
                    user[i].ontime = np.random.exponential(user[i].traffictype['voice'])
                    type = 'voice'
                    user[i].qci = user[i].qci_type[type]
                else:
                    user[i].ontime = np.random.exponential(user[i].traffictype['video'])
                    type = 'video'
                    user[i].qci = user[i].qci_type[type]
        elif user[i].offtime == 0 and user[i].ontime > 0:
            user[i].ontime -= 1
            if user[i].ontime < 0:
                user[i].ontime = 0
                user[i].offtime = np.random.exponential(user[i].offtime_restore)
                # user[i].qci = 0
        ontime_list.append(user[i].ontime)
        offtime_list.append(user[i].offtime)
    # print(ontime_list)
    # print(offtime_list)
    # input()
    return ontime_list,offtime_list

def new_data_beam(userlist,last_state):
    ontime,offtime=trafficduration_beam(userlist)
    #新到数据
    beam_number=get_user_beam_id(userlist)
    beam_number_unique = list(set(beam_number))
    newdata_beam = [0] * len(beam_number_unique)
    for i in range(len(userlist)):
        newdata=userlist[i].cbrrate * 1 if ontime[i] > 1 else userlist[i].cbrrate * ontime[i]
        for j in range(len(newdata_beam)):
            if userlist[i].beam_id==j:
                newdata_beam[j]+=newdata
        last_state.loc[j, 'newdata'] = newdata_beam[j]
        last_state.loc[j,'waitingdata'] += newdata_beam[j]
        if last_state['waitingdata'][j]>0:
            last_state['Req']==1
        else:
            last_state['Req']==0
    return last_state


if __name__ == '__main__':
    #s=initial_all_user(520000, 180)
    a,b,c,d=updata(0,0,0,0,0)
    #print(s)