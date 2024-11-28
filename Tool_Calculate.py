import math as m
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import Parameters
import cvxpy as cp
import warnings
warnings.filterwarnings("ignore", message="This use of ``*`` has resulted in matrix multiplication.")




#from geopy.distance import geodesic



#经纬度转化的计算
def GeographicToCartesianCoordinates(latitude, longitude, altitude, sphType):
    """
    坐标转换: [lat, lon, alt]转为[x, y, z];
    输入参数：[lat, lon, alt], 以及类型；
    返回: [x, y, z];
    a: semi - major axis of earth
    e: first eccentricity of earth
    """
    latitudeRadians = latitude * m.pi / 180
    longitudeRadians = longitude * m.pi / 180
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
    Rn = a / (m.sqrt(1 - pow(e, 2) * pow(m.sin(latitudeRadians), 2)))  # radius of curvature
    x = (Rn + altitude) * m.cos(latitudeRadians) * m.cos(longitudeRadians)
    y = (Rn + altitude) * m.cos(latitudeRadians) * m.sin(longitudeRadians)
    z = (Rn + altitude) * m.sin(latitudeRadians)
    cartesianCoordinates = [x, y, z]
    return cartesianCoordinates


def ConstructFromVector(x, y, z, sphType):
    """
    坐标转换: [x, y, z]转为[lat, lon, alt];
    输入参数：[x, y, z], 以及类型；
    返回: [lat, lon, alt];
    a: semi - major axis of earth
    e: first eccentricity of earth
    """
    EARTH_RADIUS = 6371e3
    EARTH_SEMIMAJOR_AXIS = 6378137
    EARTH_GRS80_ECCENTRICITY = 0.0818191910428158
    EARTH_WGS84_ECCENTRICITY = 0.0818191908426215
    if sphType == "SPHERE":
        a = EARTH_RADIUS
        e = 0
    if sphType == "GRS80":
        a = EARTH_SEMIMAJOR_AXIS
        e = EARTH_GRS80_ECCENTRICITY
    else:  # if sphType == WGS84
        a = EARTH_SEMIMAJOR_AXIS
        e = EARTH_WGS84_ECCENTRICITY
    latitudeRadians = m.asin(z / m.sqrt(x ** 2 + y ** 2 + z ** 2))
    latitude = latitudeRadians * 180 / m.pi
    if x == 0 and y > 0:
        longitude = 90
    elif x == 0 and y < 0:
        longitude = -90
    elif x < 0 and y >= 0:
        longitudeRadians = m.atan(y / x)
        longitude = longitudeRadians * 180 / m.pi + 180
    elif x < 0 and y <= 0:
        longitudeRadians = m.atan(y / x)
        longitude = longitudeRadians * 180 / m.pi - 180
    else:
        longitudeRadians = m.atan(y / x)
        longitude = longitudeRadians * 180 / m.pi
    Rn = a / (m.sqrt(1 - pow(e, 2) * pow(m.sin(latitudeRadians), 2)))
    altitude = m.sqrt(x**2+y**2+z**2)-Rn
    return [latitude, longitude, altitude]


def get_elevation_angle_geocentric(g, s):
        """
        计算地面终端与卫星或高空平台之间的仰角
        g: 地面终端坐标 (x, y, z)
        s: 卫星或高空平台坐标 (x, y, z)
        返回值: 仰角（单位：度）
        """
        elev_angle = 0
        # 计算分子
        numerator = -g[0] * (s[0] - g[0]) - g[1] * (s[1] - g[1]) - g[2] * (s[2] - g[2])
        # 计算分母
        denominator = m.sqrt(g[0]**2 + g[1]**2 + g[2]**2) * m.sqrt((s[0] - g[0])**2 + (s[1] - g[1])**2 + (s[2] - g[2])**2)
        # 计算余弦值
        x = numerator / denominator
        # 防止 acos 输入值超出[-1, 1]范围
        if x > 1:
            x = 1
        # 计算仰角（将弧度转换为度并减去90度）
        elev_angle = (180.0 * m.acos(x)) / m.pi - 90
        # 确保仰角不是NaN
        if m.isnan(elev_angle):
            raise ValueError("acos returned a NaN value")
        return elev_angle
def plot_user_position(lat,lon,req,DOWN_Rate,MAX_DOWN_Rate,bs_xyz,bs_ridth,epoch):

    DOWN_Sinr = np.power(2,DOWN_Rate/Parameters.bw)-1
    MAX_DOWN_Sinr = np.power(2,MAX_DOWN_Rate/Parameters.bw)-1
    latitudes = lat  # 纬度
    longitudes = lon  # 经度
    is_served = req # 服务标志
    user_ids = np.arange(len(lat) )
    x,y,z=np.zeros((len(lat))),np.zeros((len(lat))),np.zeros((len(lat)))
    for i in user_ids:
        x[i],y[i],z[i]=GeographicToCartesianCoordinates(lat[i],lon[i],0,"GRS80")
    # 设置颜色：被服务为蓝色，未被服务为红色
    colors = ['blue' if served == -2 else 'red' for served in is_served]
    # 绘图
    plt.figure(figsize=(10, 6))

    sa_served_points = plt.scatter(
    lon[is_served == -2],
    lat[is_served == -2],
    c='blue',
    s=50,
    edgecolors='black',
    label='Sa_user'
    )

    not_served_points = plt.scatter(
        lon[is_served == -1],
        lat[is_served == -1],
        c='black',
        s=50,
        edgecolors='black',
        label='Unser'
    )
    bs_served_points = plt.scatter(
        lon[is_served >-1],
        lat[is_served >-1],
        c='red',
        s=50,
        edgecolors='black',
        label='Bs_user'
    )
    sate = plt.scatter(
        Parameters.sate_lla[1],
        Parameters.sate_lla[0],
        c='g',
        s=100,
        edgecolors='black',
        label='Sate'

    )
    for i, user_id in enumerate(user_ids):
        plt.text(lon[i], lat[i], f'{user_id,round(DOWN_Sinr[i],1),round(MAX_DOWN_Sinr[i],1)}', fontsize=9, ha='right', va='bottom', color='black')
#     for bs in (bs_xyz):
# # 绘制基站点
#         station_point = plt.scatter(
#             bs[1],
#             bs[2],
#             c='green',
#             s=100,
#             edgecolors='black',
#             marker='^',
#             label='BS'
#         ) if bs[0]==0 else plt.scatter(
#             bs[1],
#             bs[2],
#             c='green',
#             s=100,
#             edgecolors='black',
#             marker='^',
#         )
#         # 绘制覆盖范围
#         coverage_radius_deg = bs_ridth   # 覆盖半径转换为纬度差
#         circle = Circle((bs[1], bs[2]), coverage_radius_deg,  alpha=0.2,color='green')
#         plt.gca().add_patch(circle)
    # 添加图例和标注
    plt.xlabel('Lat', fontsize=12)
    plt.ylabel('Lon', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left')
    plt.show()
    plt.savefig("./result"+str(epoch)+".jpg")

def Power_Allocation(H_gain):
    """
    H_gain:信道系数矩阵(需要除以噪声完成归一化)
    """
    # 功率转换函数 (dBm 转线性功率)
    def db2pow(db):
        return 10 ** (db / 10)
    # 参数设置
    U = H_gain.shape[1]  # 用户数量
    C = H_gain.shape[1]  # 波束数量
    p_max = Parameters.Power_BeamMax
    P = Parameters.Power_SateTotal  # 最大功率 (dBm)
    Max_iter = 50  # 最大迭代次数
    epsilon = 1e-5  # 收敛阈值
    # 初始化变量
    p_temp = np.full(C, db2pow(P) / U)  # 等功率初始化
    A = np.ones((U, C)) - np.eye(U)  # 干扰矩阵
    sum_rate = []  # 记录每次迭代的和速率
    sum_rate_old = 10000  # 初始和速率
    iter_count = 0
    # 迭代优化
    while iter_count < Max_iter:
        iter_count += 1
        # Step 1: 计算中间变量 y_star
        interference = H_gain * A @ p_temp
        y_star = np.sqrt(np.diag(H_gain) * p_temp)/ (interference+1)
        # Step 2: 使用 cvxpy 优化功率分配
        p = cp.Variable((C,), nonneg=True)
        objective = cp.sum(
            cp.log(
                1 + 2 * cp.multiply(y_star , cp.sqrt(cp.multiply(cp.diag(H_gain) ,p)) ) -cp.multiply((y_star ** 2),cp.multiply(H_gain ,A ) * p + 1)
            )
        )
        constraints = [
            p >= 0,
            p<=db2pow(p_max),
            cp.sum(p) <= db2pow(P),
        ]
        problem = cp.Problem(cp.Maximize(objective), constraints)
        problem.solve()

        # 检查求解结果
        if p.value is not None and not np.isnan(problem.value):
            sum_rate.append(problem.value)

        # 更新功率分配
        p_temp = p.value

        # 判断收敛
        if abs(problem.value - sum_rate_old) / sum_rate_old < epsilon:
            break
        else:
            sum_rate_old = problem.value
    return p_temp