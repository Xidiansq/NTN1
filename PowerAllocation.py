import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# 参数设置
U = 7  # 用户数量
C = 7  # 基站数量
P = 50  # 最大功率 (dBm)
Max_iter = 50  # 最大迭代次数
epsilon = 1e-5  # 收敛阈值
B = 10e6  # 带宽 (Hz)

# 功率转换函数 (dBm 转线性功率)
def db2pow(db):
    return 10 ** (db / 10)

# 随机生成信道增益 (假设的信道矩阵，用户需要根据具体情况生成)
np.random.seed(1)
H_gain = np.array([
    [0.316384708048302, 0.00155657941674646, 0.000320559598180551, 0.000207379562751292, 0.000257678414436390, 0.0353093407466320, 3.51745209989663e-06],
    [0.000744498388245594, 0.931713004950820, 0.00158342528900675, 0.000166163935502793, 6.73125121407765e-05, 0.000225748907378152, 5.52670929435371e-05],
    [5.60281674886471e-05, 6.22297083127522e-05, 0.00605346332054220, 0.000132947707115746, 6.35260301093142e-05, 2.62925402179796e-06, 2.99332493579648e-06],
    [0.0255567171084806, 7.91360212413198e-05, 0.000137455653973140, 0.00123883885106393, 0.000217197187511500, 7.90374042709428e-05, 1.08745334933822e-05],
    [0.000614789215870454, 0.000702412011106463, 5.47773585263571e-05, 0.000395438929635250, 0.455892818040826, 0.0182173939032595, 1.44162266285057e-05],
    [0.00181635921948911, 0.000159958425848687, 5.22078634129689e-05, 3.67963320299441e-07, 4.77261807243065e-05, 0.0347450371440490, 0.00370316180032659],
    [0.00487307139420793, 0.00166900633824559, 3.48983520098141e-06, 0.000151236606119615, 7.00073883837277e-05, 0.0205160840065022, 0.0153903742676718]
])
v = cp.Parameter((7,7), value=H_gain)
h =np.random.rand(7,7)
# 初始化变量
p_temp = np.full(C, db2pow(P) / 2)  # 等功率初始化
A = np.ones((U, C)) - np.eye(U)  # 干扰矩阵
sum_rate = []  # 记录每次迭代的和速率
sum_rate_old = 100  # 初始和速率
iter_count = 0
p_arr,p_arr_sum=[],[]

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

# 绘图
# plt.plot(range(1, len(sum_rate) + 1), np.array(sum_rate) / np.log(2) * B / 1e6, '-b*', linewidth=1)
# plt.grid(True)
# plt.xlabel('Iteration number')
# plt.ylabel('Sum rate (Mbps)')
# plt.title('Sum Rate Convergence')
# plt.show()

# plt.figure(2)
# plt.plot(range(len(p_arr)),p_arr_sum,p_arr)
# plt.show()
