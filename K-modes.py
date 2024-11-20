import numpy as np
from kmodes import kmodes

'''生成互相无交集的离散属性样本集'''
data1 = np.random.randint(1,6,(10000,10))
data2 = np.random.randint(6,12,(10000,10))

data = np.concatenate((data1,data2))

'''进行K-modes聚类'''
km = kmodes.KModes(n_clusters=2)
clusters = km.fit_predict(data)

'''计算正确归类率'''
score = np.sum(clusters[:int(len(clusters)/2)])+(len(clusters)/2-np.sum(clusters[int(len(clusters)/2):]))
score = score/len(clusters)
if score >= 0.5:
    print('正确率：'+ str(score))
else:
    print('正确率：'+ str(1-score))
