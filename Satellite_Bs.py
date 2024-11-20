import math as m
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
#from geopy.distance import geodesic
import Parameters
def setBS(center_longitude,center_latitude, cover_range, num_stations):
    """
    初始化基站的位置坐标
    输入:范围中心经度(center_longitude), 范围中心维度(center_latitude), 覆盖半径(cover_range), 基站数量(num_stations)
    输出:基站的经纬度海拔, 基站的xyz坐标
    """
    sphType = "GRS80"
    base_lla = []
    bs_xyz=[]
    # 将半径转换为米
    
    # 计算六边形网格的边长，使得网格点数量接近所需的基站数量
    area = m.pi * cover_range**2
    area_per_point = area / num_stations
    side_length = m.sqrt((2 * area_per_point) / (3 * m.sqrt(3)))

    # 计算需要的网格层数
    num_layers = int(m.ceil(cover_range / (m.sqrt(3) * side_length)))

    # 生成六边形网格点
    points = []
    for i in range(-num_layers, num_layers + 1):
        for j in range(-num_layers, num_layers + 1):
            x = side_length * (i * 3/2)
            y = side_length * (j * m.sqrt(3) + (i % 2) * m.sqrt(3)/2)
            # 检查点是否在圆内
            if x**2 + y**2 <= cover_range**2:
                points.append((x, y))

    # 如果点的数量超过所需基站数量，选择距离中心最近的点
    if len(points) > num_stations:
        points.sort(key=lambda p: p[0]**2 + p[1]**2)
        points = points[:num_stations]

    # 计算每个点的方位角和距离，并转换为经纬度
    for x, y in points:
        distance = m.hypot(x, y)
        azimuth = (m.degrees(m.atan2(x, y)) + 360) % 360
        lat, lon = vincenty_direct(center_latitude, center_longitude, azimuth, distance, sphType)
        base_lla.append((lat, lon,0))
        xyz = GeographicToCartesianCoordinates(lat, lon, 0, sphType)
        bs_xyz.append(xyz)

    bs_lla = np.array(base_lla)
    bs_xyz = np.array(bs_xyz)

    # 生成编号，从0到15
    indices = np.arange(len(bs_lla)).reshape(-1, 1)

    # 合并编号和原始数组
    bs_lla = np.hstack((indices, bs_lla))
    bs_xyz = np.hstack((indices, bs_xyz))
    return bs_lla,bs_xyz,side_length

def vincenty_direct(lat1, lon1, azimuth, distance, sphType):
    """
    Vincenty 正算公式，计算从起点出发，沿指定方位角和距离的终点坐标。
    """
    # GRS80 椭球参数
    if sphType == "GRS80":
        a = 6378137.0  # 长半轴
        f = 1 / 298.257222101  # 扁率
    else:
        raise ValueError("Unsupported spheroid type.")

    b = (1 - f) * a
    lat1 = m.radians(lat1)
    lon1 = m.radians(lon1)
    alpha1 = m.radians(azimuth)
    s = distance

    sin_alpha1 = m.sin(alpha1)
    cos_alpha1 = m.cos(alpha1)

    tanU1 = (1 - f) * m.tan(lat1)
    cosU1 = 1 / m.sqrt(1 + tanU1**2)
    sinU1 = tanU1 * cosU1

    sigma1 = m.atan2(tanU1, cos_alpha1)
    sin_alpha = cosU1 * sin_alpha1
    cos_sq_alpha = 1 - sin_alpha**2
    u_sq = cos_sq_alpha * (a**2 - b**2) / b**2
    A = 1 + u_sq / 16384 * (4096 + u_sq * (-768 + u_sq * (320 - 175 * u_sq)))
    B = u_sq / 1024 * (256 + u_sq * (-128 + u_sq * (74 - 47 * u_sq)))

    sigma = s / (b * A)
    sigma_p = 2 * m.pi
    while abs(sigma - sigma_p) > 1e-12:
        cos2sigma_m = m.cos(2 * sigma1 + sigma)
        sin_sigma = m.sin(sigma)
        cos_sigma = m.cos(sigma)
        delta_sigma = B * sin_sigma * (
            cos2sigma_m + B / 4 * (
                cos_sigma * (-1 + 2 * cos2sigma_m**2) - B / 6 * cos2sigma_m * (-3 + 4 * sin_sigma**2) * (-3 + 4 * cos2sigma_m**2)
            )
        )
        sigma_p = sigma
        sigma = s / (b * A) + delta_sigma

    tmp = sinU1 * sin_sigma - cosU1 * cos_sigma * cos_alpha1
    lat2 = m.atan2(
        sinU1 * cos_sigma + cosU1 * sin_sigma * cos_alpha1,
        (1 - f) * m.sqrt(sin_alpha**2 + tmp**2)
    )
    lam = m.atan2(
        sin_sigma * sin_alpha1,
        cosU1 * cos_sigma - sinU1 * sin_sigma * cos_alpha1
    )
    C = f / 16 * cos_sq_alpha * (4 + f * (4 - 3 * cos_sq_alpha))
    L = lam - (1 - C) * f * sin_alpha * (
        sigma + C * sin_sigma * (
            cos2sigma_m + C * cos_sigma * (-1 + 2 * cos2sigma_m**2)
        )
    )
    lon2 = lon1 + L

    lat2 = m.degrees(lat2)
    lon2 = m.degrees(lon2)
    return lat2, lon2



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

def latlon_to_cartesian(lat, lon, alt=0):
    """
    将经纬度转换为三维地心坐标系中的坐标
    lat: 纬度 (单位: 度)
    lon: 经度 (单位: 度)
    alt: 高度 (单位: 米)
    """
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    R=Parameters.R_earth
    # 计算三维坐标
    X = (R + alt ) * np.cos(lat_rad) * np.cos(lon_rad)
    Y = (R + alt ) * np.cos(lat_rad) * np.sin(lon_rad)
    Z = (R + alt ) * np.sin(lat_rad)
    
    return np.array([X, Y, Z])

def angle_between_users(user1, user2, base_station):
    """
    计算两个用户相对于基站之间的夹角
    user1, user2: 包含 'Lat', 'Lon', 'Alt' 的字典
    base_station: 基站的 'Lat', 'Lon', 'Alt'
    """
    # 将用户和基站的经纬度转换为三维坐标
    base_station_pos = np.array(GeographicToCartesianCoordinates(base_station[0], base_station[1], base_station[2],sphType="GRS80"))
    user1_pos = np.array(GeographicToCartesianCoordinates(user1[0], user1[1], user1[2],sphType="GRS80"))
    user2_pos = np.array(GeographicToCartesianCoordinates(user2[0], user2[1], user2[2],sphType="GRS80"))
    
    # 计算从基站到两个用户的向量
    vector1 = user1_pos - base_station_pos
    vector2 = user2_pos - base_station_pos
    
    # 计算夹角（使用点积公式）
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    
    cos_theta = dot_product / (magnitude1 * magnitude2)
    
    # 计算夹角（单位: 度）
    angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    
    return angle




if __name__ == "__main__":

    print("测试")