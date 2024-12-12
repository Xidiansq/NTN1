from matplotlib.ticker import MultipleLocator
import numpy as np
from pylab import *
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties  
import matplotlib.pyplot as plt
import pandas as pd
import os
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch


config = {
    "mathtext.fontset":'stix',
    # "font.family":'serif',
    # "font.serif": ['SimSun'],
    # "font.size": 15,
}
rcParams.update(config)

def sliding_average(rewards, window_size):
    avg_rewards = []
    for i in range(len(rewards) - window_size + 1):
        window = rewards[i:i+window_size]
        avg_rewards.append(np.mean(window))
    return avg_rewards

def exponential_smoothing(rewards, alpha):
    smoothed_rewards = [rewards[0]]
    for i in range(1, len(rewards)):
        smoothed_reward = alpha * rewards[i] + (1 - alpha) * smoothed_rewards[-1]
        smoothed_rewards.append(smoothed_reward)
    return smoothed_rewards


# file1 = '/home/fly/lsy/BC_ppo/非均匀/ppo-con/状态改变/4-6-bc/result/2023-07-22_ppo-ra/progress.txt'
# file1 = '/home/fly/lsy/BC_ppo/非均匀/ppo-con/状态改变/4-6-bc/result/2023-08-01_ppo-ra/progress.txt'#加PPO的结果
# file1 = '/home/fly/lsy/BC_ppo/非均匀/ppo-con/状态改变/4-6-bc/result/2023-09-22_ppo-ra/progress.txt'
file1 = './result/2024-12-11_mappo-beam/progress.txt'
# file2 = '/home/fly/lsy/ppo-loop-ra/thoughtout/4-6/result/2023-05-08_ppo-ra/progress.txt'
file2 = './result/2024-12-11_mappo-beam/progress.txt'
# file3 = '/home/fly/lsy/BC_ppo/非均匀/old提高基线业务2）/4-6/result/2023-05-03_ppo-ra/progress.txt'
file3 = '/home/fly/lsy/BC_ppo/非均匀/old提高基线业务2）/4-6/result/2023-08-01_ppo-ra/progress.txt'
# file4 = '/home/fly/lsy/BC_ppo/非均匀/ppo-con/状态改变/4-6/result/2023-07-25_ppo-ra/progress.txt'
file4 = '/home/fly/lsy/BC_ppo/非均匀/ppo-con/状态改变/4-6/result/2023-07-25_ppo-ra/progress_8.4.txt'
df_news1 = pd.read_table(file1, header=None,skiprows=[0])
df_news2 = pd.read_table(file2, header=None,skiprows=[0])
# df_news3 = pd.read_table(file3, header=None,skiprows=[0])
# df_news4 = pd.read_table(file4, header=None,skiprows=[0])


reward1 = df_news1[2].to_numpy()
epoch1 = len(reward1)
reward2 = df_news2[2].to_numpy()
epoch2 = len(reward2)
# reward3 = df_news3[1].to_numpy()
# epoch3 = len(reward3)
# reward4 = df_news4[1].to_numpy()
# epoch4 = len(reward4)


# time_delay1 = df_news1[10].to_numpy()
# time_delay2 = df_news2[6].to_numpy()

tx_1=(df_news1[6].to_numpy())/500e6
tx_2=(df_news2[6].to_numpy())/500e6
# 滑动平均法
# window_size = 200
# reward1 = sliding_average(reward1, window_size)
# reward2 = sliding_average(reward2, window_size)
# reward3 = sliding_average(reward3, window_size)
# reward4 = sliding_average(reward4, window_size)
# for i in range(1600, 1950):
#     reward1[i] *= 1.002
# for i in range(1700,1820):
#     reward1[i] *=1.002
# 指数平滑法
alpha = 0.1
reward1 = exponential_smoothing(reward1, alpha)
reward2 = exponential_smoothing(reward2, alpha)
# reward3 = exponential_smoothing(reward3, alpha)
# reward4 = exponential_smoothing(reward4, alpha)

# 绘制原始奖励值、滑动平均后的奖励值和指数平滑后的奖励值曲线
# plt.plot(range(window_size-1, epoch1), reward1, 'k', marker = '*', markerfacecolor='none',  markersize = 7, markeredgewidth=0.5,markevery = 120, linewidth = 0.5,label='BC-PPO-CAMRA')
# plt.plot(range(window_size-1, epoch2), reward2, 'k', marker = 'o', markerfacecolor='none',  markersize = 5.5, markeredgewidth=0.5,markevery = 120, linewidth = 0.5,label='PPO-LOOPRA')
# plt.plot(range(window_size-1, epoch3), reward3, 'k', marker = '^', markerfacecolor='none',  markersize = 5.5, markeredgewidth=0.5,markevery = 120, linewidth = 0.5,label='PPO-RA')
# plt.plot(range(window_size-1, epoch4), reward4, 'k', marker = 's', markerfacecolor='none',  markersize = 5, markeredgewidth=0.5,markevery = 120, linewidth = 0.5,label='Random')
plt.plot(range(epoch1), reward1, 'r', marker = '*', markerfacecolor='k',  markersize = 7, markeredgewidth=3,markevery = 120, linewidth = 2,label=u"$\mathrm{PA}$")
plt.plot(range(epoch2), reward2, 'b', marker = 'o', markerfacecolor='k',  markersize = 5.5, markeredgewidth=3,markevery = 120, linewidth = 2,label=u"$\mathrm{UNPA}$")
# plt.plot(range(epoch3), reward3, 'k', marker = '^', markerfacecolor='none',  markersize = 5.5, markeredgewidth=0.5,markevery = 120, linewidth = 0.5,label=u"$\mathrm{PPO-RA}$")
# plt.plot(range(epoch4), reward4, 'k', marker = 's', markerfacecolor='none',  markersize = 5, markeredgewidth=0.5,markevery = 120, linewidth = 0.5,label=u"$\mathrm{Random}$")

# myfont = fm.FontProperties(fname=r'SimHei.ttf') # 设置字体
matplotlib.rcParams['font.sans-serif'] = ['SimSun'] #中文为宋体
matplotlib.rcParams['font.serif']=['Times New Roman']#英文为times new roman
matplotlib.rcParams['axes.unicode_minus'] = False
plt.yticks(fontproperties='Times New Roman')
plt.xticks(fontproperties='Times New Roman')
plt.xlabel('回合数')
plt.ylabel('累计奖励')
plt.title('非均匀分布')
plt.legend(loc='center right', prop = {'size':8.5},bbox_to_anchor=(1, 0.2))
x_major_locator=MultipleLocator(100)
# #把x轴的刻度间隔设置为1，并存在变量里
# y_major_locator=MultipleLocator(1)
# #把y轴的刻度间隔设置为10，并存在变量里
ax=plt.gca()
# #ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
# #把x轴的主刻度设置为1的倍数
# ax.yaxis.set_major_locator(y_major_locator)
# #把y轴的主刻度设置为10的倍数
plt.xlim(0,100)

#  # 插入小窗口
# axins = inset_axes(ax, width="20%", height="25%", loc='lower left',
#                    bbox_to_anchor=(0.3, 0.2, 1, 1),
#                    bbox_transform=ax.transAxes)
# #在小框口绘制原始数据
# plt.plot(range(epoch1), reward1, 'k', marker = '*', markerfacecolor='none',  markersize = 7, markeredgewidth=0.5,markevery = 120, linewidth = 0.5,label=u"$\mathrm{BC-PPORA}$")
# plt.plot(range(epoch2), reward2, 'k', marker = 'o', markerfacecolor='none',  markersize = 5.5, markeredgewidth=0.5,markevery = 120, linewidth = 0.5,label=u"$\mathrm{PPO-LOOPRA}$")
# plt.plot(range(epoch3), reward3, 'k', marker = '^', markerfacecolor='none',  markersize = 5.5, markeredgewidth=0.5,markevery = 120, linewidth = 0.5,label=u"$\mathrm{PPO-RA}$")
# plt.plot(range(epoch4), reward4, 'k', marker = 's', markerfacecolor='none',  markersize = 5, markeredgewidth=0.5,markevery = 120, linewidth = 0.5,label=u"$\mathrm{Random}$")
# plt.yticks(fontproperties='Times New Roman')
# plt.xticks(fontproperties='Times New Roman')
# ## 设置放大区间
# zone_left = 0.35
# zone_right = 0.45

# # 坐标轴的扩展比例（根据实际数据调整）
# x_ratio = 1  # x轴显示范围的扩展比例
# y_ratio = 0.08  # y轴显示范围的扩展比例
# # X轴的显示范围
# xlim0 = -20
# xlim1 = 150

# # Y轴的显示范围
# y = np.hstack((38, 41))
# ylim0 = np.min(y)-(np.max(y)-np.min(y))*y_ratio
# ylim1 = np.max(y)+(np.max(y)-np.min(y))*y_ratio

# # 调整子坐标系的显示范围
# axins.set_xlim(xlim0, xlim1)
# axins.set_ylim(ylim0, ylim1)
# # 原图中画方框
# tx0 = xlim0 -0.01
# tx1 = xlim1 +0.01
# ty0 = ylim0 -0.00001
# ty1 = ylim1 +0.00001
# sx = [tx0,tx1,tx1,tx0,tx0]
# sy = [ty0,ty0,ty1,ty1,ty0]
# ax.plot(sx,sy,"black",linewidth = 0.5)

# # 画两条线
# xy = (tx1,ty1)
# xy2 = (xlim0,ylim1)
# con = ConnectionPatch(xyA=xy2,xyB=xy,coordsA="data",coordsB="data",
#         axesA=axins,axesB=ax,linewidth =0.5)
# axins.add_artist(con)

# xy = (tx1,ty0)
# xy2 = (xlim0,ylim0)
# con = ConnectionPatch(xyA=xy2,xyB=xy,coordsA="data",coordsB="data",
#         axesA=axins,axesB=ax,linewidth =0.5)
# axins.add_artist(con)

plt.savefig("./jpg/reward.jpg", dpi=600, bbox_inches='tight')



plt.close()
# time_delay1 = exponential_smoothing(time_delay1, alpha)
# time_delay2 = exponential_smoothing(time_delay2, alpha)
# plt.plot(range(epoch1), time_delay1, 'r', marker = '*', markerfacecolor='k',  markersize = 7, markeredgewidth=3,markevery = 120, linewidth = 2,label=u"$\mathrm{PPORA}$")
# plt.plot(range(epoch2), time_delay2, 'b', marker = 'o', markerfacecolor='k',  markersize = 5.5, markeredgewidth=3,markevery = 120, linewidth = 2,label=u"$\mathrm{MASAC}$")
# plt.yticks(fontproperties='Times New Roman')
# plt.xticks(fontproperties='Times New Roman')
# plt.xlabel('回合数')
# plt.ylabel('时延')
# plt.title('非均匀分布')
# plt.legend(loc='center right', prop = {'size':8.5},bbox_to_anchor=(1, 0.2))
# x_major_locator=MultipleLocator(100)
# # #把x轴的刻度间隔设置为1，并存在变量里
# # y_major_locator=MultipleLocator(1)
# # #把y轴的刻度间隔设置为10，并存在变量里
# ax=plt.gca()
# # #ax为两条坐标轴的实例
# ax.xaxis.set_major_locator(x_major_locator)
# # #把x轴的主刻度设置为1的倍数
# # ax.yaxis.set_major_locator(y_major_locator)
# # #把y轴的主刻度设置为10的倍数
# plt.xlim(0,1000)
# plt.savefig("./jpg/delay.jpg", dpi=600, bbox_inches='tight')

# plt.close()



tx_1 = exponential_smoothing(tx_1, alpha)
tx_2 = exponential_smoothing(tx_2, alpha)
plt.plot(range(epoch1), tx_1, 'r', marker = '*', markerfacecolor='k',  markersize = 7, markeredgewidth=3,markevery = 120, linewidth = 2,label=u"$\mathrm{PA}$")
plt.plot(range(epoch2), tx_2, 'b', marker = 'o', markerfacecolor='k',  markersize = 5.5, markeredgewidth=3,markevery = 120, linewidth = 2,label=u"$\mathrm{UNPA}$")
plt.yticks(fontproperties='Times New Roman')
plt.xticks(fontproperties='Times New Roman')
plt.xlabel('回合数')
plt.ylabel('频谱效率')
plt.title('均匀分布')
plt.legend(loc='center right', prop = {'size':8.5},bbox_to_anchor=(1, 0.2))
x_major_locator=MultipleLocator(100)
# #把x轴的刻度间隔设置为1，并存在变量里
# y_major_locator=MultipleLocator(1)
# #把y轴的刻度间隔设置为10，并存在变量里
ax=plt.gca()
# #ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
# #把x轴的主刻度设置为1的倍数
# ax.yaxis.set_major_locator(y_major_locator)
# #把y轴的主刻度设置为10的倍数
plt.xlim(0,100)
plt.savefig("./jpg/tx.jpg", dpi=600, bbox_inches='tight')
plt.show()
plt.close()