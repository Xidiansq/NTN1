import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.patches import Circle
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

np.random.seed(0)
# 定义波束中心点的经纬度列表
beam_centers_latlon = np.loadtxt('./all_data/center-shy.txt', usecols=(1, 2))

# 定义波束覆盖半径（单位：米）
beam_radius_meters = 440000

# 定义用户的经纬度位置和分布类型（0表示随机分布，1表示高斯分布）
num_users = 12
data = []
for i, center in enumerate(beam_centers_latlon):
    if i in [0, 1,2, 3,4,5]:  # 指定要随机分布的波束序号
        for _ in range(num_users):
            user_coord = np.random.uniform(low=center - 3, high=center + 3, size=2)  # 生成随机坐标
            while np.linalg.norm(user_coord - center) > beam_radius_meters:
                user_coord = np.random.uniform(low=center - 1, high=center + 1, size=2)  # 确保在圆内
            data.append(np.append(user_coord, 0))  # 添加分布类型列
    else:
        for _ in range(num_users):
            user_coord = np.random.normal(loc=center, scale=1, size=2)  # 正态分布坐标
            while np.linalg.norm(user_coord - center) > beam_radius_meters:
                user_coord = np.random.normal(loc=center, scale=1, size=2)  # 确保在圆内
            data.append(np.append(user_coord, 1))  # 添加分布类型列

data = np.array(data)
np.savetxt('./all_data/user-shy.txt', data[:, :2], fmt='%.8f')  # 保存前两列数据
'''with open('user-shy.txt', 'w') as f:
    for row in data:
        f.write(' '.join(map(str, row)) + '\n')'''

# 绘制局部放大地图
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

# 绘制波束覆盖范围和用户位置
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
    #ax.legend(['Random Distribution', 'Gaussian Distribution'])

# 创建图形和子图
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

# 绘制局部放大地图
plot_zoomed_map(ax, (-3.6683936558045714e-16,0.0019151850530515673), zoom_factor=18)  # 根据需要调整中心点和缩放因子

# 绘制波束覆盖范围和用户位置
# 循环处理每个波束中心
for beam_center_latlon in beam_centers_latlon:
    plot_beam_coverage(beam_center_latlon, beam_radius_meters, data, ax)

plt.show()
plt.savefig('./all_data/random.jpg')