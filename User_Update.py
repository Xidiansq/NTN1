# -*-coding:utf-8-*-
import numpy as np
import pandas as pd
import math as m
import random
import GEO_BeamDesign
import Tool_Calculate
import User_Init
import Parameters
# import User_Update_Tool
import User_Transmission
from copy import deepcopy
import time

class user_update:
    # userlist[i] = user_update(center_sat, cover_ange, ontime, offtime, 
    #                                   last_user_info.iloc[i, :], last_user_task.iloc[i, :], Action_beam, DOWN_Rate, index)
    def __init__(self, center_sat, cover_ange, all_bs,ontime, offtime, 
                 bs_if_serv,last_user_info, last_user_task, Action_beam, DOWN_Rate,user_index):

        # 上一时刻的用户信息 作为 当前时刻更新前的用户信息
        self.curr_UserID = last_user_info['UserID']
        self.curr_BsID = int(last_user_info['BsID'])
        self.curr_Lat = last_user_info['Lat']     
        self.curr_Lon = last_user_info['Lon']       
        self.curr_Alt = last_user_info['Alt']     
        self.curr_angle_user2sate = last_user_info['Angle']     
        self.curr_DisBs = last_user_info['Dis_Bs']
        self.curr_bs_of_xyz = all_bs[self.curr_BsID, -3:]
        self.curr_ReqID = last_user_info['ReqID']
        self.curr_QCI = last_user_info['QCI']      
        self.curr_NewData = last_user_info['NewData']     # 已更新
        self.last_final_wait = last_user_info['Last_WaitData'] 
        self.curr_WaitData = last_user_info['Total_WaitData']   # 已更新
        self.curr_FinishData = last_user_info['Finish_Data']    # 已更新
        self.curr_Time_Delay = last_user_info['Time_Delay']
        
        self.curr_Down_TxData = last_user_info['Down_TxData']   # 已更新
    
        self.curr_Downlink_Throughput = last_user_info['Down_Throughput']    # 已更新
        
        self.curr_Init_DataSize = last_user_task['Init_DataSize']
        self.curr_Achieved_DataSize = last_user_task['Achieved_DataSize']
        self.curr_UnAchieved_DataSize = last_user_task['UnAchieved_DataSize']
        self.curr_consume_dalay = last_user_task['Consume_Delay']
        self.curr_final_dalay = last_user_task['Final_Delay']
        self.curr_Action_Offload = last_user_task['Action_Offload']
        self.curr_Action_Computation = last_user_task['Action_Computation']
        self.curr_Action_Communication = last_user_task['Action_Communication']
        self.finish_task_num = 0
        self.UEcenter = center_sat       # 用户区域中心，用于设置用户分布, [0, 0, 0]
        self.UEmaxdistance = cover_ange   # 距离用户区域中心的最大距离，用于设置用户分布
        self.all_bs = all_bs                #所有的基站id和位置信息
        self.movespeed = Parameters.movespeed     # 用户移动速度, 单位m/s
        self.earth_radius = Parameters.R_earth  # 地球半径
        self.earthspheroidtype = {'SPHERE': 0, 'GRS80': 1, 'WGS84': 2}  # 三种地球模型
        self.randomangle = self.generate_random_angle()  # 基于用户速度, 随机产生经纬度的变化角度
        
        self.cbrrate = Parameters.cbrrate    # 固定码率0.5Mb, 可修改为可变码率VBR;
        self.ontime = 0  # 业务持续时间
        self.offtime = 0   # 业务关闭时间
        self.update_user_position(self.UEcenter, self.UEmaxdistance, self.all_bs, self.randomangle)     # 更新用户位置类信息  
        self.update_user_traffic(ontime, offtime, self.cbrrate, Action_beam, DOWN_Rate ,user_index)    # 更新用户业务类信息
        self.curr_BsIfServ = bs_if_serv[user_index]   # TODO:已更新
    def generate_random_angle(self):
        """
        基于用户速度, 产生经纬度的变化角度
        """
        direction = np.cos(np.random.uniform(0, m.pi, size=3))
        randomangle = direction * self.movespeed
        randomangle = (randomangle / (2 * np.pi * self.earth_radius)) * 360
        randomangle[2] = 0
        return randomangle
    
    def update_user_position(self, center, maxdistance, bs_lla, random_angle):
        """
        更新用户位置类信息;
        (1) 更新用户位置, 返回用户位置[x, y, z]和[lat, lon, alt];
        (2) 根据更新后的用户位置，计算用户所接入的波束;
        (3) 根据更新后的用户位置与接入波束, 计算用户夹角;
        """
        self.updated_ue_xyz, self.updated_ue_latlonalt, self.movedistance = self.update_position(center, maxdistance, random_angle)
        self.curr_BsID, self.curr_bs_of_xyz, self.curr_DisBs = self.update_connected_beam(self.updated_ue_xyz, bs_lla)
        #self.curr_angle_user2sate = self.update_angele(Parameters.sate_lla, self.updated_ue_xyz)
        self.curr_angle_user2sate = Tool_Calculate.get_elevation_angle_geocentric(self.updated_ue_xyz,Parameters.sate_xyz)
        
    

    def update_position(self, center, maxdistance, random_angle):
        """
        位置类更新函数1
        更新用户位置, 两种用户位置更新模型任选其一
        输入: center为用户区域中心, maxdistance为离用户区域中心的最大距离, random_angle为经纬度变化角;
        返回：更新后的用户位置, [x, y, z]和[lat, lon, alt], 以及用户移动距离;
        """
        # self.updated_ue_xyz, self.updated_ue_latlonalt, self.movedistance = self.model1_update(center, maxdistance, random_angle)  # 用户位置更新模型1
        self.updated_ue_xyz, self.updated_ue_latlonalt, self.movedistance = self.model2_update(center, maxdistance, random_angle)  # 用户位置更新模型2
        return self.updated_ue_xyz, self.updated_ue_latlonalt, self.movedistance


    def update_connected_beam(self, ue_position, all_bs):
        """
        位置类更新函数2
        更新用户接入波束, 方法:遍历所有基站中心位置, 选择与用户位置最接近的基站确定接入;
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
                distance_ue2bs = np.sqrt(np.sum((ue_position - all_bs_position[i])**2))
        return bs_ID, bs_connect_xyz, distance_ue2bs
    

    def update_angele(self, sate_lla, user_position):
        """
        位置类更新函数3
        更新卫星-用户与卫星—波束中心之间的夹角, 以及用户仰角;
        输入参数: all_sate_position为所有卫星位置, sate_connect_ID为所接入卫星的ID, beam_center为所接入波束的中心, user_position为用户位置;
        返回: self.angle_user卫星-用户与卫星—波束中心之间的夹角, 单位: 度°;
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
    
        
    # def model1_update(self, center, maxdistance, random_angle, time_duration=0.001):
    #     """
    #     用户位置更新模型1
    #     方法: 每次更新选择随机的移动方向和行进速率;
    #     返回: 更新后的用户位置, [x, y, z]和[lat, lon, alt], 以及用户移动距离;
    #     """
    #     self.randomangle = self.generate_random_angle()   # 先随机产生一个经纬度变化角
    #     last_ue_xyz = Satellite_Beam.GeographicToCartesianCoordinates(self.curr_Lat, self.curr_Lon, self.curr_Alt, self.earthspheroidtype['SPHERE'])  # 计算未更新时的用户位置, [x, y, z]
    #     self.curr_Lat += self.randomangle[0] * time_duration   # 更新用户纬度
    #     self.curr_Lon += self.randomangle[1] * time_duration   # 更新用户经度
    #     self.curr_Alt += self.randomangle[2] * time_duration   # 更新用户海拔
    #     if self.curr_Alt < 0:
    #         self.curr_Alt = 0
    #     updated_ue_xyz = Satellite_Beam.GeographicToCartesianCoordinates(self.curr_Lat, self.curr_Lon, self.curr_Alt, self.earthspheroidtype['SPHERE'])  # 计算更新后的用户位置, [x, y, z]
    #     center_xyz = Satellite_Beam.GeographicToCartesianCoordinates(center[0], center[1], center[2], self.earthspheroidtype['SPHERE'])  # 计算用户区域中心位置, [x, y, z]
    #     distance_ue2center = np.sum(np.square(np.array(updated_ue_xyz) - np.array(center_xyz))) ** 0.5    # 计算用户到区域中心的距离
    #     if distance_ue2center >= maxdistance:  # 边界情况, 用户位置不变
    #         self.randomangle = -self.randomangle   # 经纬度变化角为[0, 0, 0]
    #         self.curr_Lat += self.randomangle[0] * time_duration
    #         self.curr_Lon += self.randomangle[1] * time_duration
    #         self.curr_Alt += self.randomangle[2] * time_duration
    #         if self.curr_Alt < 0:
    #             self.curr_Alt = 0
    #         updated_ue_xyz = Satellite_Beam.GeographicToCartesianCoordinates(self.curr_Lat, self.curr_Lon, self.curr_Alt, self.earthspheroidtype['SPHERE']) 
    #     self.updated_ue_xyz = updated_ue_xyz
    #     self.updated_ue_latlonalt = [self.curr_Lat, self.curr_Lon, self.curr_Alt]
    #     self.movedistance = np.sum(np.square(np.array(self.updated_ue_xyz) - np.array(last_ue_xyz))) ** 0.5   # 计算移动距离
    #     return self.updated_ue_xyz, self.updated_ue_latlonalt, self.movedistance

    
    def model2_update(self, center, maxdistance, random_angle, time_duration=0.001):
        """
        用户位置更新模型2
        方法: 按照恒定的行进速率和方向前进直到到达边界, 到达边界后重新调用random_angle函数产生随机方向和距离;
        返回: 更新后的用户位置, [x, y, z]和[lat, lon, alt], 以及用户移动距离;
        """
        
        last_ue_xyz = GEO_BeamDesign.GeographicToCartesianCoordinates(self.curr_Lat, self.curr_Lon, self.curr_Alt, self.earthspheroidtype['GRS80'])  # 计算未更新时的用户位置, [x, y, z]
        self.curr_Lat += random_angle[0] * time_duration   # 非边界情况下, 更新用户纬度
        self.curr_Lon += random_angle[1] * time_duration   # 非边界情况下, 更新用户经度
        self.curr_Alt += random_angle[2] * time_duration   # 非边界情况下, 更新用户海拔
        if self.curr_Alt < 0:    
            self.curr_Alt = 0
        updated_ue_xyz = GEO_BeamDesign.GeographicToCartesianCoordinates(self.curr_Lat, self.curr_Lon, self.curr_Alt, self.earthspheroidtype['GRS80'])  # 计算更新后的用户位置, [x, y, z]
        # print('updated_ue_xyz', updated_ue_xyz)
        center_xyz = GEO_BeamDesign.GeographicToCartesianCoordinates(center[0], center[1], center[2], self.earthspheroidtype['GRS80'])  # 计算用户区域中心位置, [x, y, z]
        distance_ue2center = np.sum(np.square(np.array(updated_ue_xyz) - np.array(center_xyz))) ** 0.5    # 计算用户到区域中心的距离
        if distance_ue2center <= maxdistance:  # 非边界情况
            self.updated_ue_xyz = updated_ue_xyz
            self.updated_ue_latlonalt = [self.curr_Lat, self.curr_Lon, self.curr_Alt]
        else:  # 边界情况
            while (True):
                self.randomangle = self.generate_random_angle()   # 随机再产生一个经纬度变化角
                self.curr_Lat += self.randomangle[0] * time_duration   # 边界情况下, 更新用户纬度
                self.curr_Lon += self.randomangle[1] * time_duration   # 边界情况下, 更新用户精度
                self.curr_Alt += self.randomangle[2] * time_duration   # 边界情况下, 更新用户海拔
                if self.curr_Alt < 0:
                    self.curr_Alt = 0
                updated_ue_xyz = GEO_BeamDesign.GeographicToCartesianCoordinates(self.curr_Lat, self.curr_Lon, self.curr_Alt, self.earthspheroidtype['SPHERE'])  # 计算更新后的用户位置, [x, y, z]
                distance_ue2center = np.sum(np.square(np.array(updated_ue_xyz) - np.array(center_xyz))) ** 0.5   # 计算用户到区域中心的距离
                if distance_ue2center <= maxdistance:    # 非边界情况
                    self.updated_ue_xyz = updated_ue_xyz
                    self.updated_ue_latlonalt = [self.curr_Lat, self.curr_Lon, self.curr_Alt]
                    break
        self.movedistance = np.sum(np.square(np.array(self.updated_ue_xyz) - np.array(last_ue_xyz))) ** 0.5    # 计算移动距离
        return self.updated_ue_xyz, self.updated_ue_latlonalt, self.movedistance

    def update_user_traffic(self, ontime, offtime, cbrrate, Action_Beam, DOWN_Rate,user_index):
        """
        更新用户业务信息;
        (1) 更新用户业务类型, 以及业务持续时间; 
        (2) 更新用户的其他业务信息;
        """
        self.generate_traffic_duration(ontime, offtime)#更新用户业务持续时间
        self.update_traffic_info(cbrrate, DOWN_Rate,Action_Beam, user_index)
    

    def generate_traffic_duration(self, init_ontime, init_offtime):
        """
        业务类更新函数1
        更新用户业务类型, 以及业务持续时间;
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
                    self.curr_QCI = self.qci_type['text']
                elif traffic_choice == 2:  # 音频类型
                    self.ontime = np.random.exponential(self.traffictype['voice'])
                    self.curr_QCI = self.qci_type['voice']
                else:                    # 视频类型
                    self.ontime = np.random.exponential(self.traffictype['video'])
                    self.curr_QCI = self.qci_type['video']
        elif self.offtime == 0 and self.ontime > 0:
            self.ontime -= 1
            if self.ontime < 0:
                self.ontime = 0
                self.offtime = np.random.exponential(init_offtime)
                self.curr_QCI = self.qci_type['None']
        return self.offtime, self.ontime
    


    def update_traffic_info(self, cbrrate, DOWN_Rate, Action_Beam, user_index):
        """
        业务类更新函数3
        更新用户的其他业务信息;
        输入: cbrrate, Action_Offload, Action_Compute, Action_Commun,
        包含: 当前的传输数据大小, 新到数据大小, 时隙开始的等待传输数据大小, 时隙结束的等待传输数据大小, 所需RBG数量, 是否有请求的标识, 时延, 总计传输数据大小, 瞬时吞吐, 平均吞吐
        输入参数: 固定码率CBR, 可修改为可变码率VBR;
        """
        # if(Action_Beam[user_index] == 1):
        #     Downlink_Rate = DOWN_Rate[user_index]/1000  #bpms
        # else:
        #     Downlink_Rate = 0
        
        self.curr_Down_TxData = DOWN_Rate[user_index]/1000 # bpms   
        #self.ontime = random.uniform(0.5, 1)
        self.curr_NewData = cbrrate * 1 if self.ontime > 1 else cbrrate * self.ontime   # 当前的新到数据大小, 更新self.curr_NewData
        self.last_final_wait = self.curr_WaitData
        
        if(self.last_final_wait <= self.curr_Down_TxData): 
            self.curr_WaitData = self.curr_NewData    # 当前时隙开始的waitdata = 上一时隙结束的waitdata + 当前时隙的newdata, 更新self.curr_WaitData
            self.curr_Down_TxData = self.last_final_wait
        else:
            self.curr_WaitData = self.last_final_wait - self.curr_Down_TxData + self.curr_NewData

        # self.curr_Capacity = 200000    # 每个RBG最大传输上限, 更新self.curr_RBG_Needed  ！！！！改！！！！！
        # self.curr_RBG_Needed = m.ceil(self.curr_WaitData/self.curr_Capacity)    # 当前的所需RBG数量, 更新self.curr_RBG_Needed
        if self.curr_WaitData > 0 and self.curr_angle_user2sate >= Parameters.Elevation_Angle:     # 当前时隙是否有请求, 更新self.curr_ReqID, 并且判断用户是否在卫星覆盖范围内；
            self.curr_ReqID = 1
        else:
            self.curr_ReqID = 0
        self.curr_Time_Delay = 0  # 当前的时延
        self.curr_Downlink_Throughput = ((self.curr_Down_TxData/0.001))/(1024**2)   # 当前的下行瞬时吞吐, 单位Mbps, 更新self.curr_Downlink_Throughput
    


def update_all_user(userlist, center_sat, cover_ange, bs_xyz,
                    last_user_info, last_user_task, last_request_list,
                    Action_beam,Action_bs,ontime, offtime,bs_lla):
    """
    更新用户,
    输入参数: 
    userlist: 用户列表
    center_sat: 覆盖区域中心, 用于设置用户分布, [0, 0, 0];
    cover_ange: 覆盖区域半径, 用于设置用户分布, 单位m; 

    last_user_info:上一时刻用户的状态信息
    last_user_task:上一时刻用户的任务传输情况
    last_request_list: 上一时刻有用户请求的列表

    Action_beam:波束的跳变策略配置
    Action_bs  :基站的策略(暂时未知,需要重写)
    ontime: 业务持续时间; 
    offtime: 业务关闭时间;
    bs_lla:基站位置信息
    返回: 用户列表;
    """

    # #只提取有用户请求的列进行更新
    req_user_info = last_user_info.iloc[last_request_list, :]
    # 每个小区内用户的服务情况
    #print("req_user_info", req_user_info)
    bs_if_serv,bs_state = choose_user_bsifservice(last_user_info, 
                                                  Action_beam)
    DOWN_Rate,MAX_DOWN_Rate= User_Transmission.calculate_datarate(Action_beam, 
                                                          last_user_info, 
                                                          last_request_list,
                                                          bs_lla,
                                                          bs_state)
    for i in range(len(userlist)):
        # print("============================================================更新", i, "============================================================")
        userlist[i] = user_update(center_sat, cover_ange, bs_xyz,ontime, offtime, 
                                      bs_if_serv,last_user_info.iloc[i, :], last_user_task.iloc[i, :], Action_beam, DOWN_Rate, i)
    return userlist,DOWN_Rate,MAX_DOWN_Rate



def update_user_task_info(userlist):
    """
    获取更新后的所有用户的任务信息; 
    输入: 用户列表;
    返回信息, 包含:
    """
    task_info = []
    for i in range(len(userlist)): 
        task_info.append((i, userlist[i].curr_Init_DataSize, userlist[i].curr_UnAchieved_DataSize, userlist[i].curr_Achieved_DataSize, 
                          userlist[i].curr_consume_dalay, userlist[i].curr_final_dalay))
    task_info = pd.DataFrame(task_info, columns=['UserID', 'Init_DataSize', 'UnAchieved_DataSize', 'Achieved_DataSize', 
                                                 'Consume_Delay', 'Final_Delay'])
    # print("\n-----------------------task_info---------------------\n", task_info)
    return task_info


def update_user_traffic_info(userlist):
    """
    获取更新后的所有用户信息; 
    输入: 用户列表;
    返回信息, 包含:
    'UseID'用户ID, 'Lat'用户位置(维度), 'Lon'用户位置(经度), 'Alt'用户位置(海拔), 'Angle'用户夹角, 'WaitData'等待传输数据, 'NewData'新到达数据,  'ReqID'是否有请求, 'Last_TxData'上个时隙传输, 
    'Time_Delay'时延, 'QCI'业务类型, 'RBG_Needed'所需RBG数目, 'Total_TxData'总计传输, 'Throughput(mbps)'瞬时吞吐, 'Aver_throughput'平均吞吐, 'Capacity'单个RBG传输上限
    """
    user_request = []
    traffic_info = []
    for i in range(len(userlist)):  # 输出每一个用户的信息
        if userlist[i].curr_ReqID == 1:
            user_request.append(i)
        traffic_info.append((i, userlist[i].curr_BsID, userlist[i].curr_BsIfServ,userlist[i].curr_Lat, userlist[i].curr_Lon, userlist[i].curr_Alt, userlist[i].curr_angle_user2sate, userlist[i].curr_DisBs,
                             userlist[i].curr_ReqID,userlist[i].curr_QCI,
                             userlist[i].curr_NewData, userlist[i].last_final_wait, userlist[i].curr_WaitData, userlist[i].curr_FinishData,
                             userlist[i].curr_Time_Delay, userlist[i].curr_Down_TxData, userlist[i].curr_Downlink_Throughput))
    traffic_info = np.array(traffic_info, dtype='float64')
    traffic_info = pd.DataFrame(traffic_info, columns=['UserID','BsID','BsIfServ', 'Lat', 'Lon', 'Alt', 'Angle', 'Dis_Bs', 
                                                       'ReqID', 'QCI', 
                                                       'NewData', 'Last_WaitData', 'Total_WaitData', 'Finish_Data', 
                                                       'Time_Delay', 'Down_TxData', 'Down_Throughput'])
    return traffic_info, user_request
def choose_user_bsifservice(last_user_info, action_beam):    # TODO:添加函数
    '''
    选择用户是否被基站连接进行服务,根据用户优先级判断
    输出:
    bs_if_service:-2代表是卫星用户,-1代表不分
    served_users:具体每个小区的分配情况

    '''
    user_num=Parameters.user_number
    bs_if_service = [0] * user_num  # 存储用户是否被基站服务
    bs_state=[{
        "user_sa": [],
        "user_bs": [],
        "user_unserve": []
    } for _ in range((Parameters.bs_num))] #创建长度为基站数量的列表，用来存放每个基站内用户的服务情况
    # 按QCI优先级对用户排序
    user_priority = choose_by_random(last_user_info)
    # 按优先级分配基站资源
    for user_idx, _ in user_priority:
    # for user_idx in range(len(userlist)):
        bs = int(last_user_info['BsID'][user_idx])  # 获取基站id
        if action_beam[user_idx] == 0:  # 只考虑基站服务的用户
            if last_user_info['ReqID'][user_idx] == 1:  # 只有有请求的用户才能被服务               # init和update中的请求变量名不一样
                if len(bs_state[bs]['user_bs']) <= Parameters.antenna_num:  # 基站还有可用天线
                    bs_if_service[user_idx] = bs    # 该用户被第bsID个基站服务服务
                    bs_state[bs]['user_bs'].append(user_idx)
                else:
                    bs_if_service[user_idx] = -1    # 基站天线已满, 该用户不服务
                    bs_state[bs]['user_unserve'].append(user_idx)  # 记录未被服务的用户
            else:
                bs_if_service[user_idx] = -1  # 无请求用户不服务
                bs_state[bs]['user_unserve'].append(user_idx)
        else:
            bs_if_service[user_idx] = -2    # 该用户被卫星服务
            bs_state[bs]['user_sa'].append(user_idx)
    #print("bs_if_service:", bs_if_service)
    return bs_if_service,bs_state






def choose_by_distance(last_user_info):
    """
    按照距离进行排序
    """
    user_priority=[]
    for i in range(Parameters.user_number):
        user_priority.append((i, last_user_info['Dis_Bs'][i]))                             # init和update中的距离变量名不一样
    user_priority.sort(key=lambda x: x[1])  # 按距离从小到大排序(距离越小优先级越高)
    return user_priority
def choose_by_random():
    """
    随机排序
    """
    user_priority=[]
    for i in range(Parameters.user_number):
        user_priority.append((i,0))    
    random.shuffle(user_priority)                      
    return user_priority

