import numpy as np
from numpy import sqrt
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# 设置图像的嵌入字体（本段代码使用 copilot 辅助生成）
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15

# 设置参数
gamma = 1.4
xi_0 = 0.0                          # 间断点的位置
xi = np.linspace(-1.5, 1.5, 3000)

# 初值条件
rho_L, u_L, p_L = 1.0, 0.0, 1.0
rho_R, u_R, p_R = 0.125, 0.0, 0.1

def ancilla_func(p_star, p, rho, c):
    if p_star > p:
        output = (p_star - p) * np.sqrt((2 / ((gamma + 1) * rho)) / (p_star + (gamma - 1) / (gamma + 1) * p))
    else:
        output = (2 * c / (gamma - 1)) * ((p_star / p) ** ((gamma - 1) / (2 * gamma)) - 1)
    return output

def pressure_eqn(p_star, p_L, p_R, rho_L, rho_R, c_L, c_R):
    return ancilla_func(p_star, p_L, rho_L, c_L) + ancilla_func(p_star, p_R, rho_R, c_R) + u_R - u_L

c_L = sqrt(gamma * p_L / rho_L)     # 左侧声速
c_R = sqrt(gamma * p_R / rho_R)     # 右侧声速
p_guess = 0.5 * (p_L + p_R)         # 初始猜测解

# 数值求解压强 p_star
p_star = fsolve(pressure_eqn, p_guess, args=(p_L, p_R, rho_L, rho_R, c_L, c_R))[0]

# 计算速度 u_star
u_star = 0.5 * (u_L + u_R + ancilla_func(p_star, p_R, rho_R, c_R) - ancilla_func(p_star, p_L, rho_L, c_L))

# 计算接触间断两侧的密度
if p_star > p_L:
    rho_star_L = rho_L * ((p_star / p_L + (gamma - 1) / (gamma + 1)) / ((gamma - 1) / (gamma + 1) * p_star / p_L + 1))
else:
    rho_star_L = rho_L * (p_star / p_L) ** (1 / gamma)

if p_star > p_R:
    rho_star_R = rho_R * ((p_star / p_R + (gamma - 1) / (gamma + 1)) / ((gamma - 1) / (gamma + 1) * p_star / p_R + 1))
else:
    rho_star_R = rho_R * (p_star / p_R) ** (1 / gamma)

# 判断激波或稀疏波
left_is_shock = p_star > p_L
right_is_shock = p_star > p_R

# 计算左侧波头/波尾速度
if left_is_shock:
    xi_head_L = u_L - c_L * np.sqrt((gamma + 1) / (2 * gamma) * (p_star / p_L - 1) + 1)
    xi_tail_L = xi_head_L       # 激波波头与波尾重合
else:
    xi_head_L = u_L - c_L
    xi_tail_L = u_star - c_L * ((p_star / p_L) ** ((gamma - 1) / (2 * gamma)))

# 计算右侧波头/波尾速度
if right_is_shock:
    xi_head_R = u_R + c_R * np.sqrt((gamma + 1) / (2 * gamma) * (p_star / p_R - 1) + 1)
    xi_tail_R = xi_head_R
else:
    xi_head_R = u_R + c_R
    xi_tail_R = u_star + c_R * ((p_star / p_R) ** ((gamma - 1) / (2 * gamma)))
    
# 接触间断速度
xi_contact = u_star


# 初始化各物理量数组
rho = np.zeros_like(xi)
u = np.zeros_like(xi)
p = np.zeros_like(xi)
e = np.zeros_like(xi)

# 填入各区域的物理量
for i, xi_val in enumerate(xi):

    s = xi_val - xi_0

    # 最左侧区域
    if s < xi_head_L:
        rho[i] = rho_L
        u[i] = u_L
        p[i] = p_L

    # 左侧稀疏波区
    elif not left_is_shock and s < xi_tail_L:
        u[i] = (2 / (gamma + 1)) * (c_L + 0.5 * (gamma - 1) * u_L + s)
        c = c_L - 0.5 * (gamma - 1) * (u[i] - u_L)
        rho[i] = rho_L * (c / c_L) ** (2 / (gamma - 1))
        p[i] = p_L * (c / c_L) ** (2 * gamma / (gamma - 1))

    # 左侧激波或稀疏波与接触间断之间区域
    elif s < xi_contact:
        rho[i] = rho_star_L
        u[i] = u_star
        p[i] = p_star

    # 右侧激波或稀疏波与接触间断之间区域
    elif s < xi_tail_R:
        rho[i] = rho_star_R
        u[i] = u_star
        p[i] = p_star

    # 右侧稀疏波区
    elif not right_is_shock and s < xi_head_R:
        u[i] = (2 / (gamma + 1)) * (-c_R + 0.5 * (gamma - 1) * u_R + s)
        c = c_R + 0.5 * (gamma - 1) * (u[i] - u_R)
        rho[i] = rho_R * (c / c_R) ** (2 / (gamma - 1))
        p[i] = p_R * (c / c_R) ** (2 * gamma / (gamma - 1))

    # 最右侧区域
    else:
        rho[i] = rho_R
        u[i] = u_R
        p[i] = p_R

    e[i] = p[i] / (gamma - 1) + 0.5 * rho[i] * u[i] ** 2

# 绘制各物理量的图线并保存数据（本段代码使用 copilot 辅助生成）
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
axs[0, 0].plot(xi, rho)
axs[0, 0].set_title(r'Density $\rho$',fontsize=18)
axs[0, 1].plot(xi, u)
axs[0, 1].set_title(r'Velocity $u$',fontsize=18)
axs[1, 0].plot(xi, p)
axs[1, 0].set_title(r'Pressure $p$',fontsize=18)
axs[1, 1].plot(xi, e)
axs[1, 1].set_title(r'Total Energy Density $e$',fontsize=18)
for ax in axs.flat:
    ax.set_xlabel(r'$\xi=x/t$',fontsize=18)
    ax.grid(True)
plt.tight_layout()
plt.savefig('Sod_Riemann.png', dpi=300)
np.savez('Sod_Riemann_data.npz', xi=xi, rho=rho, u=u, p=p, e=e)