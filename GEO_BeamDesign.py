import math as m
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import User_Init
#from geopy.distance import geodesic

def setBS(center_longitude,center_latitude, cover_range, num_stations):
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
    return bs_lla,bs_xyz

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





def setInitBeamCenterPos(i, SateBLH, type):
    """
    初始化卫星的波束数量, 以及波束中心;
    输入参数: 卫星编号, 卫星位置[lat, lon, alt], 以及卫星类型；
    返回: 所有波束中心的位置, [[x, y, z], [lat, lon, alt]];
    """
    '''
    if type == "INTELSAT":
        # 波束的数量以及功率大小
        numPointsofC1 = 1
        numPointsofC2 = 5
        # 设定每一圈里中心距离orbit inclination : 86.402 degree, 1.508 rad
        maxArcDistFromSubPos = 2000000 # 2000km：单星覆盖arc长度
        arcDist1 = maxArcDistFromSubPos * 1 / 4
        arcDist2 = maxArcDistFromSubPos * 3 / 4
        # 设置每一层
        allocate1, lat_log1 = createInitBeamCenterPos(i, 0, SateBLH, numPointsofC1, arcDist1)
        allocate2, lat_log2 = createInitBeamCenterPos(i, numPointsofC1, SateBLH, numPointsofC2, arcDist2)
        allocate_x_y_z = np.array(allocate1 + allocate2)
        allocate_lat_lon_alt = np.array(lat_log1 + lat_log2)
    return allocate_x_y_z, allocate_lat_lon_alt
    '''
    #IRIDIUM  INTELSAT
    if type == "IRIDIUM":
    # 波束的数量以及功率大小
        numPointsofC1 = 1
        numPointsofC2 = 6
        numPointsofC3 = 12  # 新增第三层波束数量
        # 设定每一圈里中心距离orbit inclination : 86.402 degree, 1.508 rad
        maxArcDistFromSubPos = 1000000  # 2000km：单星覆盖arc长度
        arcDist1 = 0  # 调整第一层波束的距离
        arcDist2 = maxArcDistFromSubPos * 2 / 5  # 调整第二层波束的距离
        arcDist3 = maxArcDistFromSubPos * 3.7 / 5  # 新增第三层波束的距离
        # 设置每一层
        #当前波束总数numPointsof    numPoints每一层的波束个数
        allocate1, lat_log1 = createInitBeamCenterPos(i, 0, SateBLH, numPointsofC1, arcDist1)
        allocate2, lat_log2 = createInitBeamCenterPos(i, numPointsofC1, SateBLH, numPointsofC2, arcDist2)
        allocate3, lat_log3 = createInitBeamCenterPos(i, numPointsofC1 + numPointsofC2, SateBLH, numPointsofC3, arcDist3)  # 添加第三层波束
        allocate_x_y_z = np.array(allocate1 + allocate2+allocate3)  # 合并波束中心点坐标
        allocate_lat_lon_alt = np.array(lat_log1 + lat_log2+lat_log3)  # 合并经纬度
    return allocate_x_y_z, allocate_lat_lon_alt

def createInitBeamCenterPos(satnum, numPointsof, SateBLH, numPoints, arcDist):
    """
    初始化波束中心的位置;
    输入参数: 卫星编号, 内圈总体波束数量, 卫星位置[lat, lon, alt], 当前圈的波束数量, 中心距离;
    返回：波束中心的位置, [[x, y, z], [lat, lon, alt]];
    """
    beamCenterAlloc_x_y_z = []
    beamCenterAlloc_lat_lon_alt = []
    for i in range(numPoints):
        # 方位角设计
        cellid = satnum * 48 + numPointsof + i
        azimuth = (i + 1) * 360.0 / numPoints
        #a= 6371000
        a = 6378137.0
        b = 6356752.3142
        f = 1.0 / 298.257223563
        alpha1 = azimuth * m.pi / 180.0
        sinAlpha1 = m.sin(alpha1)
        cosAlpha1 = m.cos(alpha1)
        tanU1 = (1 - f) * m.tan(SateBLH[0] * m.pi / 180.0)
        cosU1 = 1 / m.sqrt((1 + tanU1 * tanU1))
        sinU1 = tanU1 * cosU1
        sigma1 = m.atan2(tanU1, cosAlpha1)
        sinAlpha = cosU1 * sinAlpha1
        cosSqAlpha = 1 - sinAlpha * sinAlpha
        uSq = cosSqAlpha * (a * a - b * b) / (b * b)
        A = 1 + uSq / 16384 * (4096 + uSq * (-768 + uSq * (320 - 175 * uSq)))
        B = uSq / 1024 * (256 + uSq * (-128 + uSq * (74 - 47 * uSq)))
        sigma = arcDist / (b * A)
        sigmaP = 2 * m.pi
        sinSigma = m.sin(sigma)
        cosSigma = m.cos(sigma)
        cos2SigmaM = m.cos(2 * sigma1 + sigma)
        while (abs(sigma - sigmaP) > 1e-12):
            deltaSigma = B * sinSigma * (cos2SigmaM + B / 4 * (cosSigma * (-1 + 2 * cos2SigmaM * cos2SigmaM) - B / 6 * cos2SigmaM * ( -3 + 4 * sinSigma * sinSigma) * (-3 + 4 * cos2SigmaM * cos2SigmaM)))
            sigmaP = sigma
            sigma = arcDist / (b * A) + deltaSigma
        tmp = sinU1 * sinSigma - cosU1 * cosSigma * cosAlpha1

        lat2 = m.atan(
            (sinU1 * cosSigma + cosU1 * sinSigma * cosAlpha1) / ((1 - f) * m.sqrt(sinAlpha * sinAlpha + tmp * tmp)))
        lambd = m.atan((sinSigma * sinAlpha1) / (cosU1 * cosSigma - sinU1 * sinSigma * cosAlpha1))
        C = f / 16 * cosSqAlpha * (4 + f * (4 - 3 * cosSqAlpha))
        L = lambd - (1 - C) * f * sinAlpha * (
            sigma + C * sinSigma * (cos2SigmaM + C * cosSigma * (-1 + 2 * cos2SigmaM * cos2SigmaM)))
        pointPosition = GeographicToCartesianCoordinates(lat2 * 180 / m.pi, SateBLH[1] + L * 180 / m.pi, 0, "GRS80")
        lat_log = ConstructFromVector(pointPosition[0], pointPosition[1], pointPosition[2], "GRS80")
        beamCenterAlloc_x_y_z.append([cellid, pointPosition[0], pointPosition[1], pointPosition[2]])
        beamCenterAlloc_lat_lon_alt.append([cellid, lat_log[0], lat_log[1], lat_log[2]])
    return beamCenterAlloc_x_y_z, beamCenterAlloc_lat_lon_alt


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


def userconnectsate(userposition, beamposition, request, usernumer):#,choose):
    """
    根据用户位置、波束位置、请求和用户数量，确定连接信息和波束编号。
    
    输入参数：
    - userposition: 包含用户位置信息的列表或数组
    - beamposition: 包含波束位置信息的列表或数组
    - request: 包含用户请求信息的列表或数组
    - usernumer: 用户数量
    
    函数返回值：
    - all_connect_info: 包含所有连接信息的数组
    - beam_number: 包含波束编号信息的数组
    """
    all_connect_info = []
    beam_number = np.zeros(usernumer)
    print(len(userposition))
    for i in range(len(userposition)):
        
        user = userposition[i]
        print(userposition[i])
        distance_max = np.inf
        connect_beam_position = 0
        for j in range(len(beamposition)):
            beam = beamposition[j][1:]#波束中心的xyz坐标
            distance = np.sqrt(np.sum((user-beam)**2))
            if distance < distance_max:
                distance_max = distance
                connect_beam_position = beam
                beam_number[request[i]] = beamposition[j][0]+1
        user_connect_info = np.hstack((request[i], user, connect_beam_position, [0, 0, 0], beam_number[request[i]]))
        all_connect_info.append(user_connect_info)
    all_connect_info = np.array(all_connect_info)
    return all_connect_info, beam_number
#用户请求编号
# 用户位置的 x 坐标
# 用户位置的 y 坐标
# 用户位置的 z 坐标
# 连接的波束中心位置的 x 坐标
# 连接的波束中心位置的 y 坐标
# 连接的波束中心位置的 z 坐标
# 预留的空白位置（可能有多个）
# 连接的波束编号
    # all_connect_info = []
    # beam_number = np.zeros(usernumer)
    # for i in range(len(userposition)):
    #     user = userposition[i]
    #     if choose[i] == 0:
    #         beam_number[request[i]] = (i//15)+1
    #         connect_beam_position = beamposition[i//15][1:]
    #     elif choose[i] != 0:
    #         beam_number[request[i]] = choose[i]
    #         connect_beam_position = beamposition[choose[i]-1][1:]
    #     user_connect_info = np.hstack((request[i], user, connect_beam_position, [0, 0, 0], beam_number[request[i]]))
    #     all_connect_info.append(user_connect_info)
    # all_connect_info = np.array(all_connect_info)
    # return all_connect_info, beam_number


if __name__ == "__main__":
    # allocate_x_y_z, allocate_lat_lon_alt = setInitBeamCenterPos(0, [0, 0, 0], "IRIDIUM")
    # print('allocate_lat_lon_alt', allocate_lat_lon_alt)
    # with open('./all_data/center-shy.txt', 'w') as f:
    #     for row in allocate_lat_lon_alt:
    #         f.write(' '.join(map(str, row)) + '\n')
    a,b,c=ConstructFromVector(17593162,18897777,32384.632388944305,'sphere')
    print(a,b,c)
    a,b,c=ConstructFromVector(17593162,18897777,88,'sphere')
    print(a,b,c)
    a,b,c=ConstructFromVector(30000000,18897777,0,'sphere')
    print(a,b,c)
    a,b,c=ConstructFromVector(30000000,18897777,88,'sphere')
    print(a,b,c)
    # a=[]
    # b=[]
    # userlist=np.loadtxt('./all_data/output.txt')
    # a,b=userconnectsate(userlist.)
    '''x0, y0 = 0, 0
    x1 = allocate_lat_lon_alt[:, 1]
    y1 = allocate_lat_lon_alt[:, 2]
    user = np.loadtxt('./BeamAdaptation/non-uniform2.txt', encoding='utf-8', delimiter=" ")
    x2 = user[:, 0]
    y2 = user[:, 1]
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.scatter(x0, y0, c='r', marker='^')
    ax1.scatter(x1, y1, c='r', marker='x')
    ax1.scatter(x2, y2, c='b', marker='.')

    # 波束的3dB角
    # threedB = [1.6626, 1.5749, 1.6161, 1.6571, 1.6993, 1.4050, 1.7788, 1.6144, 1.4654, 1.7377, 1.8095, 1.5284]
    # threedB = [1.8603, 1.4941, 1.5433, 1.5711, 1.8851, 1.3499, 1.6659, 1.5355, 1.7594, 1.6371, 1.6957, 1.4500]
    threedB = [1.5786, 1.4939, 1.5422, 1.5770, 1.6128, 1.9015, 1.6780, 1.5420, 0.0000, 1.8773, 1.9257, 1.4549]
    # 波束的1dB角
    # onedB = [0.4800, 0.4546, 0.4665, 0.4784, 0.4905, 0.4056, 0.5135, 0.4660, 0.4230, 0.5016, 0.5224, 0.4412]
    # onedB = [0.5370, 0.4313, 0.4455, 0.4535, 0.5442, 0.3897, 0.4809, 0.4432, 0.5079, 0.4726, 0.4895, 0.4186]
    onedB = [0.4557, 0.4313, 0.4452, 0.4552, 0.4656, 0.5489, 0.4844, 0.4452, 0.0000, 0.5419, 0.5559, 0.4200]
    # 业务变化剧烈的波束
    # bigonedB = [0.5370, 0, 0, 0, 0.5442, 0, 0, 0, 0.5079, 0, 0, 0]
    bigonedB = [0, 0, 0, 0, 0, 0.5489, 0, 0, 0, 0.5419, 0.5559, 0]

    for i in range(len(allocate_lat_lon_alt)):
        cir1 = Circle(xy=(x1[i], y1[i]), radius=35786000 * m.tan(m.radians(onedB[i]))/111000, alpha=0.4)
        ax1.add_patch(cir1)
        cir2 = Circle(xy=(x1[i], y1[i]), radius=35786000 * m.tan(m.radians(threedB[i]))/111000, alpha=0.2)
        ax1.add_patch(cir2)
        cir3 = Circle(xy=(x1[i], y1[i]), radius=35786000 * m.tan(m.radians(bigonedB[i])) / 111000, alpha=0.7)
        ax1.add_patch(cir3)
    cir3 = Circle(xy=(x0, y0), radius=1500000 / 111000, alpha=0.15)
    ax1.add_patch(cir3)

    font = {'family': 'Times New Roman', 'weight': 'normal', 'size': '16'}
    plt.xlabel("Longitude", font)
    plt.ylabel("Latitude", font)
    plt.xlim(-18, 18)
    plt.ylim(-18, 18)
    plt.tick_params(labelsize=12)
    labels = ax1.get_xticklabels() + ax1.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    plt.text(x=-2.5,  # 文本x轴坐标
             y=1.3,  # 文本y轴坐标
             s='B1',  # 文本内容
             rotation=1,  # 文字旋转
             ha='left',  # x=2.2是文字的左端位置，可选'center', 'right', 'left'
             va='baseline',  # y=8是文字的低端位置，可选'center', 'top', 'bottom', 'baseline', 'center_baseline'
             fontdict=dict(fontsize=12, color='r',
                           family='Times New Roman',  # 字体,可选'serif', 'sans-serif', 'cursive', 'fantasy', 'monospace'
                           weight='normal',  # 磅值，可选'light', 'normal', 'medium', 'semibold', 'bold', 'heavy', 'black'
                           )  # 字体属性设置
            )
    plt.text(x=-4.5, y=-4, s='B2', rotation=1, ha='left', va='baseline', fontdict=dict(fontsize=12, color='r', family='Times New Roman', weight='normal'))
    plt.text(x=5.5, y=0.75, s='B3', rotation=1, ha='left', va='baseline', fontdict=dict(fontsize=12, color='r', family='Times New Roman', weight='normal'))
    plt.text(x=8.6, y=8.6, s='B4', rotation=1, ha='left', va='baseline', fontdict=dict(fontsize=12, color='r', family='Times New Roman', weight='normal'))
    plt.text(x=3.5, y=12, s='B5', rotation=1, ha='left', va='baseline', fontdict=dict(fontsize=12, color='r', family='Times New Roman', weight='normal'))
    plt.text(x=-7.2, y=9.2, s='B6', rotation=1, ha='left', va='baseline', fontdict=dict(fontsize=12, color='r', family='Times New Roman', weight='normal'))
    plt.text(x=-11.75, y=4.8, s='B7', rotation=1, ha='left', va='baseline', fontdict=dict(fontsize=12, color='r', family='Times New Roman', weight='normal'))
    plt.text(x=-13, y=-3.3, s='B8', rotation=1, ha='left', va='baseline', fontdict=dict(fontsize=12, color='r', family='Times New Roman', weight='normal'))
    plt.text(x=-6.3, y=-14, s='B9', rotation=1, ha='left', va='baseline', fontdict=dict(fontsize=12, color='r', family='Times New Roman', weight='normal'))
    plt.text(x=2.8, y=-12.5, s='B10', rotation=1, ha='left', va='baseline', fontdict=dict(fontsize=12, color='r', family='Times New Roman', weight='normal'))
    plt.text(x=9.1, y=-11.5, s='B11', rotation=1, ha='left', va='baseline', fontdict=dict(fontsize=12, color='r', family='Times New Roman', weight='normal'))
    plt.text(x=14, y=-1.5, s='B12', rotation=1, ha='left', va='baseline', fontdict=dict(fontsize=12, color='r', family='Times New Roman', weight='normal'))
    plt.show()'''