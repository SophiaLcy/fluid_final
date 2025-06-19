import numpy as np
import matplotlib.pyplot as plt

# 设置图像的嵌入字体（本段代码使用 copilot 辅助生成）
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15

# 网格参数
gamma = 1.4                                         # 绝热指数
num_x = 100                                         # 空间网格点数
x = np.linspace(-1.5, 1.5, num_x)                   # 计算域 [-1.5, 1.5]
dx = x[1] - x[0]                                    # 空间步长
CFL = 0.1                                           # CFL 条件数
t_final = 1                                         # 演化终止时间

# 数值格式选择（
FLUX_TYPE = 'TVD'

# 初值条件
def init(x):
    rho = np.where(x < 0, 1.0, 0.125)               # 密度
    u = np.zeros_like(x)                            # 速度
    p = np.where(x < 0, 1.0, 0.1)                   # 压力
    return primitive_to_conserved(rho, u, p)        # 转换为守恒变量

# 守恒变量与原始变量相互转换（本段代码根据 copilot 的建议调整）
def primitive_to_conserved(rho, u, p):
    e = p / (gamma - 1) + 0.5 * rho * u ** 2        # 总能量密度
    return np.array([rho, rho * u, e])

def conserved_to_primitive(U):
    rho = U[0]
    u = np.where(rho > 1e-12, U[1] / rho, 0.0)      # 防止分母为零
    e = U[2]
    p = (gamma - 1) * (e - 0.5 * rho * u ** 2)
    
    # 防止负密度或负压强
    rho = np.maximum(1e-9, rho)
    p = np.maximum(1e-9, p)
    return rho, u, p

# 计算通量函数
def calc_flux(U):
    rho, u, p = conserved_to_primitive(U)
    flux = np.zeros_like(U)
    flux[0] = rho * u
    flux[1] = rho * u ** 2 + p
    flux[2] = u * (U[2] + p)
    return flux

# Minmod 限制器 (TVD 格式)
def minmod(a, b):
    return np.where(a * b > 0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0)

# TVD 格式 (二阶 MUSCL + Lax-Friedrichs 通量)
def tvd_flux(U):

    # 计算原始变量 W (采取 E.F. Toro 的符号记法)
    W = np.array(conserved_to_primitive(U))
    rho, u, p = W
    c = np.sqrt(gamma * p / rho)                    # 声速
    alpha = np.max(np.abs(u) + c)                   # Jacobi矩阵的谱半径，即最大波速

    # 计算原始变量的斜率，并使用 minmod 限制器
    dW_forward  = W[:, 2:] - W[:, 1:-1]             # 前向差分
    dW_backward = W[:, 1:-1] - W[:, :-2]            # 后向差分
    dW = minmod(dW_forward, dW_backward)            # 通过限制器约束斜率

    # 对 i+1/2 界面进行线性插值
    W_L = W[:, 1:-1] + 0.5 * dW                     # i+1/2 界面左侧值
    W_R = W[:, 2:]   - 0.5 * dW                     # i+1/2 界面右侧值
    
    # 转换为守恒变量
    U_L = primitive_to_conserved(W_L[0], W_L[1], W_L[2])
    U_R = primitive_to_conserved(W_R[0], W_R[1], W_R[2])

    # 计算中心格式界面通量
    flux_L = calc_flux(U_L)
    flux_R = calc_flux(U_R)
    flux_interface = 0.5 * (flux_L + flux_R - alpha * (U_R - U_L))
    
    # 为了与其他格式的输出维度对齐，在两端填充零（本段代码根据 copilot 的建议添加）
    padded_flux = np.zeros((3, num_x))
    padded_flux[:, 1:-1] = flux_interface
    return padded_flux

# 五阶 WENO 算法
def weno5_reconstruct(f):
    # 初始化
    eps = 1e-6                                      # 防止分母为零
    num_var, num_x = f.shape
    flux_L = np.zeros((num_var, num_x))             # 存储界面左侧的重构值
    flux_R = np.zeros((num_var, num_x))             # 存储界面右侧的重构值

    for m in range(num_var):                        # 对每个守恒分量独立进行重构
        
        # 定义重构所需的五个模板点
        # - 此处根据 copilot 建议修改为 np.roll 以加速运算
        # - 为了消除边界污染，采取对其他格式两端补零，并在外围增加虚拟单元的方式处理
        # - 处理方法同样来自 copilot 的建议
        flux = f[m]
        flux_m2 = np.roll(flux, 2)
        flux_m1 = np.roll(flux, 1)
        flux_p1 = np.roll(flux, -1)
        flux_p2 = np.roll(flux, -2)
        
        # 计算光滑指示器 beta
        beta_0 = (13/12) * (flux_m2 - 2 * flux_m1 + flux)**2 + (1/4) * (flux_m2 - 4 * flux_m1 + 3 * flux)**2
        beta_1 = (13/12) * (flux_m1 - 2 * flux + flux_p1)**2 + (1/4) * (flux_m1 - flux_p1)**2
        beta_2 = (13/12) * (flux - 2 * flux_p1 + flux_p2)**2 + (1/4) * (3 * flux - 4 * flux_p1 + flux_p2)**2
        
        # 计算非权重系数 w
        alpha_0 = 0.1 / (eps + beta_0)**2
        alpha_1 = 0.6 / (eps + beta_1)**2
        alpha_2 = 0.3 / (eps + beta_2)**2
        w_sum = alpha_0 + alpha_1 + alpha_2
        w_0 = alpha_0 / w_sum
        w_1 = alpha_1 / w_sum
        w_2 = alpha_2 / w_sum
        
        # 计算左侧重构值
        f0 = (2 * flux_m2 - 7 * flux_m1 + 11 * flux)/6
        f1 = (-flux_m1 + 5 * flux + 2 * flux_p1)/6
        f2 = (2 * flux + 5 * flux_p1 - flux_p2)/6
        flux_L[m] = w_0 * f0 + w_1 * f1 + w_2 * f2
        
        # 计算右侧重构值
        f0 = (-flux_m2 + 5 * flux_m1 + 2 * flux)/6
        f1 = (2 * flux_m1 + 5 * flux - flux_p1)/6
        f2 = (11 * flux - 7 * flux_p1 + 2 * flux_p2)/6
        flux_R[m] = w_0 * f2 + w_1 * f1 + w_2 * f0
        
    return flux_L, flux_R

# FVS 方法 (WENO5 + Lax-Friedrichs 通量分裂)
def weno_lf_flux(U):

    # 计算通量函数和最大波速
    rho, u, p = conserved_to_primitive(U)
    flux = calc_flux(U)
    c = np.sqrt(gamma * p / rho)
    alpha = np.max(np.abs(u) + c)

    # 将通量函数 f 分裂为正负通量 f+ 和 f-
    flux_plus  = 0.5 * (flux + alpha * U)
    flux_minus = 0.5 * (flux - alpha * U)

    # 对 f+ 和 f- 分别进行 WENO5 重构
    flux_plus_L, _  = weno5_reconstruct(flux_plus)
    _, flux_minus_R = weno5_reconstruct(flux_minus)
    
    # 组合重构后的通量得到界面通量
    # - 其中 f^{-}_{r} 是在 i+1 单元上重构得到的，需要向左移动一位
    flux_mines_R_shifted = np.roll(flux_minus_R, -1, axis=1)
    flux_interface = flux_plus_L + flux_mines_R_shifted
    return flux_interface

# FVS 方法 (WENO5 + Van Leer 通量分裂)
def weno_vl_flux(U):

    # 计算原始变量和马赫数 mach
    rho, u, p = conserved_to_primitive(U)
    c = np.sqrt(gamma * p / rho)
    mach = np.where(c != 0, u / c, 0.0)             # 防止分母为零
    flux = calc_flux(U)

    flux_plus  = np.zeros_like(U)
    flux_minus = np.zeros_like(U)

    # 根据马赫数判断流动状态（生成掩码）
    subsonic = np.abs(mach) < 1                     # 亚音速
    supersonic_pos = mach >= 1                      # 超音速（正向）
    supersonic_neg = mach <= -1                     # 超音速（负向）
    
    # 计算超音速区域的正负通量
    flux_plus[:, supersonic_pos] = flux[:, supersonic_pos]
    flux_minus[:, supersonic_pos] = 0.0
    flux_plus[:, supersonic_neg] = 0.0
    flux_minus[:, supersonic_neg] = flux[:, supersonic_neg]

    # 利用掩码截取亚音速区域
    rho_sub, u_sub, c_sub, M_sub = rho[subsonic], u[subsonic], c[subsonic], mach[subsonic]
    
    # 计算亚音速区域的分裂通量 f+
    f_mass_p = rho_sub * c_sub * (M_sub + 1)**2 / 4.0
    f_mom_p = f_mass_p * ((gamma - 1) * u_sub + 2 * c_sub) / gamma
    f_eng_p = f_mass_p * (((gamma - 1) * u_sub + 2 * c_sub)**2 / (2 * (gamma**2 - 1)))
    flux_plus[0, subsonic], flux_plus[1, subsonic], flux_plus[2, subsonic] = f_mass_p, f_mom_p, f_eng_p
    
    # 计算亚音速区域的分裂通量 f-
    f_mass_m = -rho_sub * c_sub * (M_sub - 1)**2 / 4.0
    f_mom_m = f_mass_m * ((gamma - 1) * u_sub - 2 * c_sub) / gamma
    f_eng_m = f_mass_m * (((gamma - 1) * u_sub - 2 * c_sub)**2 / (2 * (gamma**2 - 1)))
    flux_minus[0, subsonic], flux_minus[1, subsonic], flux_minus[2, subsonic] = f_mass_m, f_mom_m, f_eng_m

    # 对 f+ 和 f- 分别进行 WENO5 重构
    flux_plus_L, _  = weno5_reconstruct(flux_plus)
    _, flux_minus_R = weno5_reconstruct(flux_minus)
    
    # 组合重构后的通量得到界面通量
    # - 其中 f^{-}_{r} 是在 i+1 单元上重构得到的，需要向左移动一位
    flux_mines_R_shifted = np.roll(flux_minus_R, -1, axis=1)
    flux_interface = flux_plus_L + flux_mines_R_shifted
    return flux_interface

# 间断指示器 (混合格式)
def shock_sensor(U, threshold=0.03):
    # 中心差分方法计算密度梯度
    rho, _, _ = conserved_to_primitive(U)
    grad_rho = np.abs(np.roll(rho, -1) - np.roll(rho, 1))
    
    # 搜索密度梯度的最大值作为参考值
    max_grad = np.max(grad_rho[2:-2])               # 忽略边界效应
    if max_grad < 1e-9:
        return np.zeros_like(rho, dtype=bool)       # 防止分母为零
    
    return (grad_rho / max_grad) > threshold

# 混合通量计算 (WENO_LF/TVD 混合格式)
def hybrid_lf_flux(U):

    # 分别计算两种格式的数值通量
    flux_weno = weno_lf_flux(U)
    flux_tvd = tvd_flux(U)

    # 判断是否位于间断区域
    shock_indicator = shock_sensor(U)
    
    # 创建掩码（本段代码使用 copilot 辅助生成）
    shock_interface_raw = shock_indicator[:-1] | shock_indicator[1:]
    shock_interface = np.zeros(num_x, dtype=bool)
    shock_interface[:-1] = shock_interface_raw
    shock_mask = np.broadcast_to(shock_interface[:, np.newaxis], (num_x, 3)).T
    
    # 根据掩码选择通量格式：在间断处使用 TVD，在光滑区域使用 WENO
    flux_interface = np.where(shock_mask, flux_weno, flux_tvd)
    return flux_interface

# 混合通量计算 (WENO_VL/TVD 混合格式)
def hybrid_vl_flux(U):

    # 分别计算两种格式的数值通量
    flux_weno = weno_vl_flux(U)
    flux_tvd = tvd_flux(U)

    # 判断是否位于间断区域
    shock_indicator = shock_sensor(U)
    
    # 创建掩码（本段代码使用 copilot 辅助生成）
    shock_interface_raw = shock_indicator[:-1] | shock_indicator[1:]
    shock_interface = np.zeros(num_x, dtype=bool)
    shock_interface[:-1] = shock_interface_raw
    shock_mask = np.broadcast_to(shock_interface[:, np.newaxis], (num_x, 3)).T
    
    # 根据掩码选择通量格式：在间断处使用 TVD，在光滑区域使用 WENO
    flux_interface = np.where(shock_mask, flux_weno, flux_tvd)
    return flux_interface

# 设置边界条件（零阶外推以设置2层虚拟节点）
def apply_boundary(U):
    U[:,0] = U[:,2]
    U[:,1] = U[:,2]
    U[:,-1] = U[:,-3]
    U[:,-2] = U[:,-3]
    return U

# TVD Runge-Kutta 时间推进格式（三阶精度）
def rk3_step(U, dt, flux_func):
    u_0 = U.copy()
    
    # 第一步
    flux = flux_func(u_0)
    rhs = -(flux[:, 2:-2] - flux[:, 1:-3]) / dx
    u_1 = u_0.copy()
    u_1[:, 2:-2] = u_0[:, 2:-2] + dt * rhs
    u_1 = apply_boundary(u_1)

    # 第二步
    flux = flux_func(u_1)
    rhs = -(flux[:, 2:-2] - flux[:, 1:-3]) / dx
    u_2 = u_0.copy()
    u_2[:, 2:-2] = 0.75 * u_0[:, 2:-2] + 0.25 * (u_1[:, 2:-2] + dt * rhs)
    u_2 = apply_boundary(u_2)

    # 第三步
    flux = flux_func(u_2)
    rhs = -(flux[:, 2:-2] - flux[:, 1:-3]) / dx
    u_new = u_0.copy()
    u_new[:, 2:-2] = (1.0/3.0) * u_0[:, 2:-2] + (2.0/3.0) * (u_2[:, 2:-2] + dt * rhs)
    u_new = apply_boundary(u_new)
    
    return u_new

############（以下代码使用 copilot 辅助生成）############

# 主循环
U = init(x)                                     # 初值条件
U = apply_boundary(U)                           # 边界条件
time = 0.0                                      # 初始化时间

# 根据全局设置选择通量计算函数
if FLUX_TYPE == 'WENO_LF':
    flux_function = weno_lf_flux
    print("使用 FVS (WENO + Lax-Friedrichs) 方法")
elif FLUX_TYPE == 'WENO_VL':
    flux_function = weno_vl_flux
    print("使用 FVS (WENO + Van Leer) 方法")
elif FLUX_TYPE == 'TVD':
    flux_function = tvd_flux
    print("使用 TVD (MUSCL-Hancock) 方法")
elif FLUX_TYPE == 'HBD_LF':
    flux_function = hybrid_lf_flux
    print("使用 Hybrid (WENO_LF/TVD Hybrid) 方法")
elif FLUX_TYPE == 'HBD_VL':
    flux_function = hybrid_vl_flux
    print("使用 Hybrid (WENO_VL/TVD Hybrid) 方法")
else:
    raise ValueError(f"未知的通量类型: {FLUX_TYPE}")

# 时间推进循环
while time < t_final:
    # 根据 CFL 条件计算当前时间步长 dt
    rho, u, p = conserved_to_primitive(U)
    c = np.sqrt(gamma * p / rho)
    max_wave_speed = np.nanmax(np.abs(u) + c)
    if max_wave_speed == 0: break               # 防止除以零
    dt = CFL * dx / max_wave_speed
    dt = min(dt, t_final - time)                # 确保不会超出最终时间
    
    # 调用 RK3 函数推进一个时间步
    U = rk3_step(U, dt, flux_function)
    time += dt
    print(f"Time: {time:.4f}/{t_final}", end="\r")

# 加载Riemann严格解
try:
    exact_data = np.load('Sod_Riemann_data.npz')
    x_exact = exact_data['xi']
    rho_exact = exact_data['rho']
    u_exact = exact_data['u']
    p_exact = exact_data['p']
    has_exact_solution = True
except FileNotFoundError:
    print("\n警告: 未找到严格解文件 'Sod_Riemann_data.npz'，仅绘制数值解。")
    has_exact_solution = False

# 获取最终时刻的数值解原始变量
rho_num, u_num, p_num = conserved_to_primitive(U)

# 设置图像的嵌入字体
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['legend.fontsize'] = 10

# 创建图表，包含三个子图
fig, axs = plt.subplots(3, 1, figsize=(6, 15))
fig.suptitle(rf'Scheme: {FLUX_TYPE} at $t={t_final:.2f}$\quad($N={num_x}$)', fontsize=18)

# 密度图
axs[0].plot(x, rho_num, 'b-', label='Numerical')
if has_exact_solution:
    axs[0].plot(x_exact, rho_exact, 'r--', label='Exact')
axs[0].set_title(r'Density $\rho$',fontsize=18)
axs[0].set_xlabel(r'Coordinate $x$',fontsize=18)
axs[0].legend(fontsize=15)
axs[0].grid(True)

# 速度图
axs[1].plot(x, u_num, 'b-', label='Numerical')
if has_exact_solution:
    axs[1].plot(x_exact, u_exact, 'r--', label='Exact')
axs[1].set_title(r'Velocity $u$',fontsize=18)
axs[1].set_xlabel(r'Coordinate $x$',fontsize=18)
axs[1].legend(fontsize=15)
axs[1].grid(True)

# 压力图
axs[2].plot(x, p_num, 'b-', label='Numerical')
if has_exact_solution:
    axs[2].plot(x_exact, p_exact, 'r--', label='Exact')
axs[2].set_title(r'Pressure $p$',fontsize=18)
axs[2].set_xlabel(r'Coordinate $x$',fontsize=18)
axs[2].legend(fontsize=15)
axs[2].grid(True)

# 调整布局并保存图像
plt.tight_layout(rect=[0, 0.05, 1, 0.98])
plt.savefig(f'{FLUX_TYPE}_N={num_x}.png', dpi=300)
np.savez(f'{FLUX_TYPE}_N={num_x}_data.npz', x=x, rho=rho, u=u, p=p)