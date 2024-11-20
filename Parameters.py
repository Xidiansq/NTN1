# -*-coding:utf-8-*-




#覆盖参数
center_latitude = 36.0735  #°                中心维度    
center_longitude = 99.0898 #°               中心经度
cover_range = 200 * 1000 #m                 卫星覆盖范围
cbrrate = 3000000


#用户参数
user_number = 20 #                          用户状态
user_info_len = 16#                         用户状态长度                          
Gain_user = 30 #dBi                         用户接收增益
movespeed = 1000 #m/s                        用户移动速度

# Traffic参数
ontime = 7   # 业务持续时间分布的均值
offtime = 3   # 业务关闭时间分布的均值


#卫星参数
sate_lla = [36.0735, 99.0898, 1.13468e+06]      #卫星的经纬度坐标
sate_xyz = [-958408, 5.99098e+06, 4.41951e+06]  #卫星的xyz坐标

beam_open = 8  #                           波束数量 
frequency = 2e10  #Hz                       中心频率
HLeo = 1300 * 1000 #m                        卫星高度
Power_SateTotal = 30 # 30-35dBW             总功率
Gain_Beam = 42 #dBi                         波束增益            
Power_BeamMax = 0.8 * Power_SateTotal #dBW  波束最大功率
noisy = -117 # dbW                          噪声
Elevation_Angle = 40 #°                     最小仰角
bw = 500e6 #Hz                              带宽
sate_threedB = 3.5  #°                        3dB角  
R_earth = 6371e3 # m                        地球半径

#基站参数
BS_INT_TH = -123 #dBW                       干扰阈值
bs_num = 7 #                               基站数量
Power_bs = 17 #17-22dBW                     基站功率
Gain_bs = 20 #dBi                           基站增益
bs_threedB = 30  #°                         3dB角 
bs_tianxian = 3
front_to_back_ratio = 30#dB                 前后比（定向天线的前后比是指主瓣的最大辐射方向（规定为0°）的功率通量密度与相反方向附近
#                                           （规定为180°±20°范围内）的最大功率通量密度之比值。它表明了天线对后瓣抑制的好坏。
#                                           前后比越大，天线的后向辐射（或接收）越小。）
antenna_num = 3 #                           基站的天线数量
side_length= 137454.51384890845#            小区六边形边长
# # Stalink参数
# TLE_PATH = "./STARLINK_TLE.txt"
# Type = "STARLINK"
# maxArcDistFromSubPos_list = [573500 /2, 940700 /2]       # 单星覆盖arc范围, 两个版本: (1) 40°, 573.5km radius; (2) 25°, 940.7km radius;
# Elevation_Angle_list = [62.4641, 49.4635]    # 40°, 25°
# Beam_Design_Para = 0
# maxArcDistFromSubPos = maxArcDistFromSubPos_list[Beam_Design_Para]
# Elevation_Angle = Elevation_Angle_list[Beam_Design_Para]


# #zuixao

# # Environment参数
# regioncenter = [0, 0, 0]   # 观测区域中心
# regionradius = 10000    # 观测区域半径
# UEcenter = [0, 0, 0]   # 用户观测区域中心
# UEmaxdistance = 500000   # 用户观测区域半径
# user_num = 20   # 用户总数
# rbg_num = 12


# # Traffic参数
# ontime = 10   # 业务持续时间分布的均值
# offtime = 2   # 业务关闭时间分布的均值

# # Offload 卸载
# Offload_Ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# # Compute 计算
# Computation_Ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# Edge_Satellite_Num = 3
# Cloud_Compute_Maximum = 50_000_000_000     # CPU cycles/s
# Cloud_CPU_needed_per_bit = 1000            # CPU cycles/bit
# Local_Compute_Maximum = 400_000_000        # CPU cycles/s
# Local_CPU_needed_per_bit = 1000            # CPU cycles/bit
# Edge_Compute_Maximum = 10_000_000_000      # CPU cycles/s
# Edge_CPU_needed_per_bit = 1000             # CPU cycles/s
# DataSize_over_compute = 20000


# # Communication 通信
# Communication_Ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]