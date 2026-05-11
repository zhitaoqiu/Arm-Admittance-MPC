import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from scipy.optimize import minimize


# ============================================================
# 1. MuJoCo 模型路径
# ============================================================
MODEL_PATH = Path(__file__).parent / "models" / "two_link_arm.xml"


# ============================================================
# 2. MPC 控制器
# ============================================================
class JointTorqueMPC:
    """
    最小版关节力矩 MPC 控制器。

    控制目标：
        让关节角度 q 跟踪目标 q_des。

    控制输入：
        关节力矩 tau。

    约束：
        tau_min <= tau <= tau_max

    当前这一版是简化 MPC：
        1. 用当前 MuJoCo 计算出的质量矩阵 M
        2. 在 MPC 预测窗口内暂时认为 M 和 bias 不变
        3. 预测未来 N 步状态
        4. 通过优化求出未来 N 步的 tau
        5. 只执行第 1 步 tau
        6. 下一次循环重新优化
    """

    def __init__(self, model, horizon=12, dt_mpc=0.03):
        """
        model:
            MuJoCo 模型

        horizon:
            MPC 预测步数 N

        dt_mpc:
            MPC 预测模型使用的时间步长
        """

        self.model = model

        # 预测步数
        self.N = horizon

        # MPC 内部预测步长
        self.dt = dt_mpc

        # 关节数量
        self.nq = 2

        # 控制量数量，也就是两个关节力矩
        self.nu = 2

        # --------------------------------------------------------
        # 力矩约束
        # --------------------------------------------------------
        # 这里和 XML 里的 motor ctrlrange="-20 20" 对齐。
        self.tau_min = -20.0
        self.tau_max = 20.0

        # --------------------------------------------------------
        # 代价函数权重
        # --------------------------------------------------------
        # Qq 越大，MPC 越重视关节位置误差
        self.Qq = np.array([120.0, 80.0])

        # Qdq 越大，MPC 越希望速度小、不要剧烈运动
        self.Qdq = np.array([3.0, 2.0])

        # R 越大，MPC 越不愿意使用大力矩
        self.R = np.array([0.01, 0.01])

        # 终端位置误差权重
        self.Qq_terminal = np.array([300.0, 200.0])

        # --------------------------------------------------------
        # 保存上一轮优化结果，用来作为下一轮初值
        # 这样优化更稳定，也更快。
        # --------------------------------------------------------
        self.previous_u_sequence = np.zeros((self.N, self.nu))

    def get_frozen_dynamics(self, model, data):
        """
        从 MuJoCo 当前状态中提取一个简化动力学模型。

        MuJoCo 真实动力学是：

            M(q) * qdd + bias(q, dq) = tau

        所以：

            qdd = inv(M) * (tau - bias)

        这里为了让 MPC 简单可跑，
        我们在一个预测窗口内认为 M 和 bias 暂时不变。
        """

        # 确保 MuJoCo 内部数据是最新的
        mujoco.mj_forward(model, data)

        # 读取质量矩阵 M
        M = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, M, data.qM)

        # 读取 bias 项
        # qfrc_bias 包括科氏力、离心力、重力等广义偏置力
        # qfrc_passive 包括关节阻尼等被动力
        bias = data.qfrc_bias.copy() + data.qfrc_passive.copy()

        return M, bias

    def rollout_cost(self, u_flat, q0, dq0, q_des, M, bias):
        """
        MPC 代价函数。

        输入：
            u_flat : 展平后的未来 N 步力矩序列
            q0     : 当前关节角度
            dq0    : 当前关节角速度
            q_des  : 目标关节角度
            M      : 当前质量矩阵
            bias   : 当前偏置力

        输出：
            cost   : 这个力矩序列的总代价

        MPC 会不断尝试不同的 u_flat，
        找到让 cost 最小的那一个。
        """

        # 把一维向量恢复成 N x 2 的力矩序列
        u_sequence = u_flat.reshape(self.N, self.nu)

        # 当前预测状态
        q = q0.copy()
        dq = dq0.copy()

        # 质量矩阵求逆
        M_inv = np.linalg.inv(M)

        cost = 0.0

        # --------------------------------------------------------
        # 逐步预测未来状态
        # --------------------------------------------------------
        for k in range(self.N):
            tau = u_sequence[k]

            # 简化动力学：
            # qdd = inv(M) * (tau - bias)
            qdd = M_inv @ (tau - bias)

            # 欧拉积分更新速度
            dq = dq + qdd * self.dt

            # 欧拉积分更新位置
            q = q + dq * self.dt

            # 位置误差
            q_error = q - q_des

            # 速度误差
            dq_error = dq

            # 当前步代价
            position_cost = np.sum(self.Qq * q_error ** 2)
            velocity_cost = np.sum(self.Qdq * dq_error ** 2)
            effort_cost = np.sum(self.R * tau ** 2)

            cost += position_cost + velocity_cost + effort_cost

        # --------------------------------------------------------
        # 终端代价
        # --------------------------------------------------------
        # 希望预测窗口最后一步尽量接近目标
        terminal_error = q - q_des
        cost += np.sum(self.Qq_terminal * terminal_error ** 2)

        return cost

    def solve(self, model, data, q_des):
        """
        求解 MPC 优化问题。

        输入：
            model : MuJoCo 模型
            data  : MuJoCo 当前状态
            q_des : 目标关节角度

        输出：
            tau_cmd : 当前时刻要执行的关节力矩
            info    : 调试信息
        """

        # 当前状态
        q0 = data.qpos.copy()
        dq0 = data.qvel.copy()

        # 获取简化动力学参数
        M, bias = self.get_frozen_dynamics(model, data)

        # --------------------------------------------------------
        # 初始猜测
        # --------------------------------------------------------
        # 用上一轮优化结果右移一格作为本轮初值。
        # 这叫 warm start。
        initial_guess_sequence = np.vstack([
            self.previous_u_sequence[1:],
            self.previous_u_sequence[-1:]
        ])

        u0 = initial_guess_sequence.reshape(-1)

        # --------------------------------------------------------
        # 力矩边界
        # --------------------------------------------------------
        # 每个时间步有两个力矩，每个都限制在 [-20, 20]。
        bounds = [
            (self.tau_min, self.tau_max)
            for _ in range(self.N * self.nu)
        ]

        # --------------------------------------------------------
        # 调用 scipy 优化器
        # --------------------------------------------------------
        result = minimize(
            fun=self.rollout_cost,
            x0=u0,
            args=(q0, dq0, q_des, M, bias),
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": 30,
                "ftol": 1e-4,
            }
        )

        # 如果优化成功，就使用优化结果
        if result.success:
            optimal_sequence = result.x.reshape(self.N, self.nu)
        else:
            # 如果优化失败，也先使用 result.x。
            # 很多时候即使 success=False，result.x 仍然比初值好。
            optimal_sequence = result.x.reshape(self.N, self.nu)

        # 保存本轮结果，给下一轮 warm start
        self.previous_u_sequence = optimal_sequence.copy()

        # MPC 只执行第一个控制输入
        tau_cmd = optimal_sequence[0]

        info = {
            "success": result.success,
            "cost": result.fun,
            "message": result.message,
        }

        return tau_cmd, info


# ============================================================
# 3. 画图函数
# ============================================================
def plot_results(log, q_des):
    """
    画出 MPC 关节跟踪结果。
    """

    time_array = np.array(log["time"])
    q_array = np.array(log["q"])
    dq_array = np.array(log["dq"])
    tau_array = np.array(log["tau"])
    cost_array = np.array(log["cost"])

    # ------------------------------------------------------------
    # 图 1：关节角度 q
    # ------------------------------------------------------------
    plt.figure()
    plt.plot(time_array, q_array[:, 0], label="q1")
    plt.plot(time_array, q_array[:, 1], label="q2")
    plt.plot(time_array, np.ones_like(time_array) * q_des[0], "--", label="q1_des")
    plt.plot(time_array, np.ones_like(time_array) * q_des[1], "--", label="q2_des")
    plt.xlabel("Time [s]")
    plt.ylabel("Joint position [rad]")
    plt.title("MPC Joint Position Tracking")
    plt.grid(True)
    plt.legend()

    # ------------------------------------------------------------
    # 图 2：关节速度 dq
    # ------------------------------------------------------------
    plt.figure()
    plt.plot(time_array, dq_array[:, 0], label="dq1")
    plt.plot(time_array, dq_array[:, 1], label="dq2")
    plt.xlabel("Time [s]")
    plt.ylabel("Joint velocity [rad/s]")
    plt.title("Joint Velocity")
    plt.grid(True)
    plt.legend()

    # ------------------------------------------------------------
    # 图 3：关节力矩 tau
    # ------------------------------------------------------------
    plt.figure()
    plt.plot(time_array, tau_array[:, 0], label="tau1")
    plt.plot(time_array, tau_array[:, 1], label="tau2")
    plt.axhline(20.0, linestyle="--", label="torque upper limit")
    plt.axhline(-20.0, linestyle="--", label="torque lower limit")
    plt.xlabel("Time [s]")
    plt.ylabel("Joint torque [N.m]")
    plt.title("MPC Joint Torque")
    plt.grid(True)
    plt.legend()

    # ------------------------------------------------------------
    # 图 4：MPC cost
    # ------------------------------------------------------------
    plt.figure()
    plt.plot(time_array, cost_array, label="MPC cost")
    plt.xlabel("Time [s]")
    plt.ylabel("Cost")
    plt.title("MPC Optimization Cost")
    plt.grid(True)
    plt.legend()

    plt.show()


# ============================================================
# 4. 主程序
# ============================================================
def main():
    # ------------------------------------------------------------
    # 4.1 加载 MuJoCo 模型
    # ------------------------------------------------------------
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    # ------------------------------------------------------------
    # 4.2 设置初始状态
    # ------------------------------------------------------------
    data.qpos[:] = np.array([0.0, 0.0])
    data.qvel[:] = np.array([0.0, 0.0])

    mujoco.mj_forward(model, data)

    # ------------------------------------------------------------
    # 4.3 设置目标关节角度
    # ------------------------------------------------------------
    # 和最开始 PD 控制一样：
    # joint1 目标 45 度
    # joint2 目标 -90 度
    q_des = np.array([np.pi / 4.0, -np.pi / 2.0])

    # 创建 MPC 控制器
    mpc = JointTorqueMPC(
        model=model,
        horizon=12,
        dt_mpc=0.03
    )

    print("Model loaded successfully.")
    print("Model path:", MODEL_PATH)
    print("Target q_des:", q_des)
    print("MPC horizon:", mpc.N)
    print("MPC dt:", mpc.dt)
    print("Torque limit:", mpc.tau_min, mpc.tau_max)

    # ------------------------------------------------------------
    # 4.4 日志
    # ------------------------------------------------------------
    log = {
        "time": [],
        "q": [],
        "dq": [],
        "tau": [],
        "cost": [],
    }

    # 仿真总时长
    sim_duration = 5.0

    # MuJoCo 仿真步长是 0.002 s
    sim_dt = model.opt.timestep

    # MPC 不需要每 0.002 秒都求解一次，会太慢
    # 这里设置每 10 个 MuJoCo 步求解一次 MPC
    control_interval_steps = 10

    step_count = 0

    # 当前保持的控制力矩
    tau_cmd = np.array([0.0, 0.0])

    # ------------------------------------------------------------
    # 4.5 打开 viewer
    # ------------------------------------------------------------
    with mujoco.viewer.launch_passive(model, data) as viewer:

        while viewer.is_running() and data.time < sim_duration:
            step_start = time.time()

            # ----------------------------------------------------
            # 每隔 control_interval_steps 步求解一次 MPC
            # ----------------------------------------------------
            if step_count % control_interval_steps == 0:
                tau_cmd, info = mpc.solve(
                    model=model,
                    data=data,
                    q_des=q_des
                )

                # 限制力矩
                tau_cmd = np.clip(tau_cmd, mpc.tau_min, mpc.tau_max)

            # ----------------------------------------------------
            # 输入当前力矩
            # ----------------------------------------------------
            data.ctrl[:] = tau_cmd

            # 推进 MuJoCo 仿真一步
            mujoco.mj_step(model, data)

            # 更新 viewer
            viewer.sync()

            # ----------------------------------------------------
            # 记录数据
            # ----------------------------------------------------
            log["time"].append(data.time)
            log["q"].append(data.qpos.copy())
            log["dq"].append(data.qvel.copy())
            log["tau"].append(tau_cmd.copy())

            # 如果这一步没有重新求解 MPC，就沿用上一轮 cost
            if step_count % control_interval_steps == 0:
                current_cost = info["cost"]
            else:
                current_cost = log["cost"][-1] if len(log["cost"]) > 0 else 0.0

            log["cost"].append(current_cost)

            # ----------------------------------------------------
            # 打印简要信息
            # ----------------------------------------------------
            if step_count % 250 == 0:
                q = data.qpos.copy()
                dq = data.qvel.copy()
                q_error = q_des - q

                print(
                    f"t = {data.time:.2f} s, "
                    f"q = [{q[0]:.3f}, {q[1]:.3f}], "
                    f"q_error = [{q_error[0]:.3f}, {q_error[1]:.3f}], "
                    f"tau = [{tau_cmd[0]:.3f}, {tau_cmd[1]:.3f}], "
                    f"cost = {current_cost:.3f}"
                )

            step_count += 1

            # ----------------------------------------------------
            # 控制仿真速度接近真实时间
            # ----------------------------------------------------
            elapsed = time.time() - step_start

            if elapsed < sim_dt:
                time.sleep(sim_dt - elapsed)

    # ------------------------------------------------------------
    # 4.6 仿真结束后画图
    # ------------------------------------------------------------
    print("Simulation finished. Plotting results...")
    plot_results(log, q_des)


# ============================================================
# 5. 程序入口
# ============================================================
if __name__ == "__main__":
    main()