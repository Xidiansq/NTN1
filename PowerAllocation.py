import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import Parameters
# 参数设置
U = 6  # 用户数量
C = 6 # 基站数量
P = 27  # 最大功率 (dBm)
Max_iter = 50  # 最大迭代次数
epsilon = 1e-5  # 收敛阈值
B = 500e6  # 带宽 (Hz)

# 功率转换函数 (dBm 转线性功率)
def db2pow(db):
    return 10 ** (db / 10)

# 随机生成信道增益 (假设的信道矩阵，用户需要根据具体情况生成)
np.random.seed(1)
H_gain = np.array([[6.56068032e+00, 4.14082210e-04, 4.14037739e-04, 2.16815721e-02,
        4.14816651e-04, 4.13954782e-04],
       [4.14829294e-04, 6.57251705e+00, 1.47613464e-02, 4.15915248e-04,
        4.14858206e-04, 4.18416520e-04],
       [4.13298993e-04, 1.47084716e-02, 6.54897445e+00, 4.14558044e-04,
        4.13344199e-04, 4.14008107e-04],
       [2.19932608e-02, 4.21134531e-04, 4.21269272e-04, 6.65499497e+00,
        4.21027043e-04, 4.21259586e-04],
       [4.13736426e-04, 4.13032683e-04, 4.13004711e-04, 4.13979385e-04,
        6.54359564e+00, 2.83144994e-03],
       [4.15133180e-04, 4.18851930e-04, 4.15928777e-04, 4.16471689e-04,
        2.84692386e-03, 6.57935650e+00]])
# H_gain = np.array([
#     [0.316384708048302, 0.00155657941674646, 0.000320559598180551, 0.000207379562751292],
#     [0.000744498388245594, 0.931713004950820, 0.00158342528900675, 0.000166163935502793],
#     [5.60281674886471e-05, 6.22297083127522e-05, 0.00605346332054220, 0.000132947707115746],
#     [0.0255567171084806, 7.91360212413198e-05, 0.000137455653973140, 0.00123883885106393]
   
# ])
# 初始化变量
p_temp = np.full(C, db2pow(P) / 6)  # 等功率初始化
A = np.ones((U, C)) - np.eye(U)  # 干扰矩阵
sum_rate = []  # 记录每次迭代的和速率
sum_rate_old = 10000  # 初始和速率
iter_count = 0
p_arr,p_arr_sum=[],[]
p_arr.append(p_temp)
p_arr_sum.append(sum(p_temp))
interference = H_gain * A @ p_temp
y_star = np.sqrt(np.diag(H_gain) * p_temp)/ (interference+1)
sum_rate.append(cp.sum(
        cp.log(
            1 + 2 * cp.multiply(y_star , cp.sqrt(cp.multiply(cp.diag(H_gain) ,p_temp)) ) -cp.multiply((y_star ** 2),cp.multiply(H_gain ,A ) * p_temp + 1)
        )
    ).value)
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
        p >= 1,
        cp.sum(p) <= db2pow(P),
    ]
    problem = cp.Problem(cp.Maximize(objective), constraints)
    problem.solve()

    # 检查求解结果
    if p.value is not None and not np.isnan(problem.value):
        sum_rate.append(problem.value)

    # 更新功率分配
    p_temp = p.value
    p_arr_sum.append(sum(p_temp))
    p_arr.append(p_temp)
    # 判断收敛
    if abs(problem.value - sum_rate_old) / sum_rate_old < epsilon:
        break
    else:
        sum_rate_old = problem.value
    

#绘图
plt.plot(range(len(sum_rate)), np.array(sum_rate) / np.log(2) * B / 1e6, '-b*', linewidth=1)
plt.grid(True)
plt.xlabel('Iteration number')
plt.ylabel('Sum rate (Mbps)')
plt.title('Sum Rate Convergence')
plt.show()

plt.figure(2)
plt.plot(range(len(p_arr)),p_arr_sum,p_arr)
plt.show()
