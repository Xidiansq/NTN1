import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from GEO_BeamDesign import *

def K_means_user(user_lat_log):
    model = KMeans(max_iter = 500, tolerance = 0.001, n_clusters =19, runs = 100)
    (clusters, data_with_clusters) = model.fit(user_lat_log)
    data_with_clusters = np.array((sorted(data_with_clusters, key=lambda x: x[2])))
    return clusters,data_with_clusters

num_users = 12
data = []
beam_radius_meters = 100000
beam, beam_lat_log = setInitBeamCenterPos(0, [0, 0, 0], type='IRIDIUM')
beam_lat_log = beam_lat_log[:, 1:3]
data=[]
for i, center in enumerate(beam_lat_log):
    if i in [0, 1,2, 3,4,5,6]:  # 指定要随机分布的波束序号
        for _ in range(num_users):
            user_coord = np.random.uniform(low=center -0.3 , high=center + 0.3, size=2)  # 生成随机坐标
            while np.linalg.norm(user_coord - center) > beam_radius_meters:
                user_coord = np.random.uniform(low=center - 0.3, high=center + 0.3, size=2)  # 确保在圆内
            data.append(np.append(user_coord, 0))  # 添加分布类型列
    else:
        for _ in range(num_users):
            user_coord = np.random.normal(loc=center, scale=0.2, size=2)  # 正态分布坐标
            while np.linalg.norm(user_coord - center) > beam_radius_meters:
                user_coord = np.random.normal(loc=center, scale=0.2, size=2)  # 确保在圆内
            data.append(np.append(user_coord, 1))  # 添加分布类型列
user_lat_log = pd.DataFrame(data,columns=['lat','lng','id']).drop(columns='id')

# 绘制聚类前的用户位置图
plt.figure(figsize=(8, 6))
plt.title('Before Clustering')
for center in beam_lat_log:
    plt.scatter(center[0], center[1], color='red')  # 绘制波束中心点
    for _ in range(num_users):
        user_coord = np.random.uniform(low=center - 0.3, high=center + 0.3, size=2)  # 生成随机坐标
        while np.linalg.norm(user_coord - center) > beam_radius_meters:
            user_coord = np.random.uniform(low=center - 0.3, high=center + 0.3, size=2)  # 确保在圆内
        plt.scatter(user_coord[0], user_coord[1], color='blue')  # 绘制用户位置

# 聚类前的图像显示
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.grid(True)
plt.show()
plt.savefig('./all_data/before.jpg')
# 聚类
clusters, data_with_clusters = K_means_user(user_lat_log)
cString = ['#005EAD', '#AF6DE5', '#719FFB', '#1CAC99', '#FE9499', '#4A8FDE', '#F8A13E', 
           '#4DE890', '#2178B8', '#77A2E8', '#F86067', '#FBBF72', '#FA9B97', '#30A02D',
           '#361D32', '#543C52', '#F65A53', '#EED2CB', '#DBD873', '#F1E8E8']
for i, cluster_mean in enumerate(clusters):
    data_cluster_i = data_with_clusters[ data_with_clusters[:, -1] == i ]
    #plt.scatter(data_cluster_i[:, 0], data_cluster_i[:, 1], label ='P'+ str(i),color=cString[i])
    plt.scatter(data_cluster_i[:, 0], data_cluster_i[:, 1],color=cString[i])
    plt.plot(cluster_mean[0], cluster_mean[1], label = 'C' + str(i), marker='*', markersize=15, markeredgecolor="k", markeredgewidth=1,color=cString[i])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.style.use('seaborn')
    plt.legend(bbox_to_anchor=(0.97, 1), loc='upper left')
    plt.savefig('./all_data/beam_kmeans.jpg')
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
plt.savefig('./all_data/after.jpg')