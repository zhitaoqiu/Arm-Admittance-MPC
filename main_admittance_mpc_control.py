import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from scipy.optimize import minimize

from kinematics import forward_kinematics, jacobian


# ============================================================
# 1. MuJoCo 模型路径
# ============================================================
MODEL_PATH = Path(__file__).parent / "models" / "two_link_arm.xml"


# ============================================================
# 2. 导纳控制器
# ============================================================
class AdmittanceController:
    """
    二维平面导纳控制器。

    输入：
        外力 F_ext

    输出：
        动态参考位置 x_ref
        动态参考速度 dx_ref

    导纳模型：
        M * ddx_ref + D * dx_ref + K * (x_ref - x0) = F_ext

    直观理解：
        外力越大，x_ref 偏移越多；
        K 越大，系统越硬，偏移越小；
        D 越大，系统越不容易振荡；
        M 越大，响应越慢。
    """

    def __init__(self, x0):
        # 原始平衡位置，也就是绿色目标点
        self.x0 = x0.copy()

        # 当前导纳参考位置，一开始等于 x0
        self.x_ref = x0.copy()

        # 当前导纳参考速度
        self.dx_ref = np.array([0.0, 0.0])

        # 虚拟质量
        self.M = np.array([1.0, 1.0])

        # 虚拟阻尼
        self.D = np.array([15.0, 15.0])

        # 虚拟刚度
        self.K = np.array([60.0, 60.0])

    def update(self, F_ext, dt):
        """
        根据外力 F_ext 更新导纳模型状态。
        """

        # 导纳方程：
        # M * ddx_ref + D * dx_ref + K * (x_ref - x0) = F_ext
        #
        # 移项得到：
        # ddx_ref = (F_ext - D * dx_ref - K * (x_ref - x0)) / M
        ddx_ref = (
            F_ext
            - self.D * self.dx_ref
            - self.K * (self.x_ref - self.x0)
        ) / self.M

        # 积分得到速度
        self.dx_ref = self.dx_ref + ddx_ref * dt

        # 积分得到位置
        self.x_ref = self.x_ref + self.dx_ref * dt

        return self.x_ref.copy(), self.dx_ref.copy(), ddx_ref.copy()


# ============================================================
# 3. 末端空间力矩 MPC 控制器
# ============================================================
class TaskSpaceTorqueMPC:
    """
    末端空间力矩 MPC 控制器。

    控制目标：
        让机械臂真实末端 x 跟踪导纳生成的 x_ref。

    控制输入：
        两个关节力矩 tau。

    约束：
        tau_min <= tau <= tau_max

    这一版和普通末端 MPC 的区别：
        目标 x_ref 是动态变化的。
        x_ref 来自导纳控制器。
    """

    def __init__(self, model, horizon=12, dt_mpc=0.03):
        self.model = model

        # MPC 向未来预测多少步
        self.N = horizon

        # MPC 内部预测步长
        self.dt = dt_mpc

        # 两个关节
        self.nq = 2

        # 两个控制输入，也就是两个关节力矩
        self.nu = 2

        # 力矩限制
        self.tau_min = -20.0
        self.tau_max = 20.0

        # --------------------------------------------------------
        # 代价函数权重
        # --------------------------------------------------------
        # 末端位置跟踪误差权重
        self.Qx = np.array([600.0, 600.0])

        # 末端速度跟踪误差权重
        self.Qdx = np.array([20.0, 20.0])

        # 关节速度惩罚，防止关节转太快
        self.Qdq = np.array([1.0, 1.0])

        # 力矩惩罚，防止力矩太大
        self.R = np.array([0.01, 0.01])

        # 终端末端位置误差权重
        self.Qx_terminal = np.array([1500.0, 1500.0])

        # warm start：保存上一轮优化出来的未来力矩序列
        self.previous_u_sequence = np.zeros((self.N, self.nu))

    def get_frozen_dynamics(self, model, data):
        """
        从 MuJoCo 当前状态里提取一个简化动力学模型。

        MuJoCo 动力学近似写成：

            M(q) * qdd + bias(q, dq) = tau

        所以：

            qdd = inv(M) * (tau - bias)

        为了让 MPC 先跑起来，这里在一个预测窗口内固定 M 和 bias。
        """

        mujoco.mj_forward(model, data)

        # 质量矩阵 M
        M = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, M, data.qM)

        # 偏置力项
        bias = data.qfrc_bias.copy() + data.qfrc_passive.copy()

        return M, bias

    def rollout_cost(self, u_flat, q0, dq0, x_ref, dx_ref, M, bias):
        """
        MPC 代价函数。

        输入：
            u_flat : 未来 N 步力矩序列，展开成一维
            q0     : 当前关节角度
            dq0    : 当前关节速度
            x_ref  : 当前导纳生成的参考末端位置
            dx_ref : 当前导纳生成的参考末端速度
            M      : 简化质量矩阵
            bias   : 简化偏置项

        输出：
            cost   : 这串未来力矩的总代价

        作用：
            给一串未来力矩打分。
            cost 越小，说明这串力矩越好。
        """

        # 优化器给的是一维数组，这里恢复成 N x 2
        u_sequence = u_flat.reshape(self.N, self.nu)

        # 从当前状态开始预测
        q = q0.copy()
        dq = dq0.copy()

        M_inv = np.linalg.inv(M)

        cost = 0.0

        for k in range(self.N):
            tau = u_sequence[k]

            # ----------------------------------------------------
            # 预测下一步关节状态
            # ----------------------------------------------------
            qdd = M_inv @ (tau - bias)

            dq = dq + qdd * self.dt
            q = q + dq * self.dt

            # ----------------------------------------------------
            # 根据预测出来的 q, dq 计算预测末端位置和速度
            # ----------------------------------------------------
            x = forward_kinematics(q)
            J = jacobian(q)
            dx = J @ dq

            # ----------------------------------------------------
            # 误差
            # ----------------------------------------------------
            x_error = x - x_ref
            dx_error = dx - dx_ref

            # ----------------------------------------------------
            # 当前预测步代价
            # ----------------------------------------------------
            position_cost = np.sum(self.Qx * x_error ** 2)
            velocity_cost = np.sum(self.Qdx * dx_error ** 2)
            joint_velocity_cost = np.sum(self.Qdq * dq ** 2)
            effort_cost = np.sum(self.R * tau ** 2)

            cost += (
                position_cost
                + velocity_cost
                + joint_velocity_cost
                + effort_cost
            )

        # --------------------------------------------------------
        # 终端代价
        # --------------------------------------------------------
        x_terminal = forward_kinematics(q)
        terminal_error = x_terminal - x_ref

        cost += np.sum(self.Qx_terminal * terminal_error ** 2)

        return cost

    def solve(self, model, data, x_ref, dx_ref):
        """
        求解 MPC。

        输入：
            x_ref  : 导纳控制器生成的参考末端位置
            dx_ref : 导纳控制器生成的参考末端速度

        输出：
            tau_cmd : 当前时刻要执行的关节力矩
            info    : 优化信息
        """

        q0 = data.qpos.copy()
        dq0 = data.qvel.copy()

        M, bias = self.get_frozen_dynamics(model, data)

        # --------------------------------------------------------
        # warm start
        # --------------------------------------------------------
        # 上次优化得到：
        # [u0, u1, u2, ..., u11]
        #
        # 当前已经执行过 u0，所以这次初值用：
        # [u1, u2, ..., u11, u11]
        initial_guess_sequence = np.vstack([
            self.previous_u_sequence[1:],
            self.previous_u_sequence[-1:]
        ])

        u0 = initial_guess_sequence.reshape(-1)

        # 每个未来力矩都限制在 [-20, 20]
        bounds = [
            (self.tau_min, self.tau_max)
            for _ in range(self.N * self.nu)
        ]

        result = minimize(
            fun=self.rollout_cost,
            x0=u0,
            args=(q0, dq0, x_ref, dx_ref, M, bias),
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": 35,
                "ftol": 1e-4,
            }
        )

        optimal_sequence = result.x.reshape(self.N, self.nu)

        # 保存本轮优化结果，供下一轮 warm start
        self.previous_u_sequence = optimal_sequence.copy()

        # MPC 只执行第一步力矩
        tau_cmd = optimal_sequence[0]

        info = {
            "success": result.success,
            "cost": result.fun,
            "message": result.message,
        }

        return tau_cmd, info


# ============================================================
# 4. 外力输入
# ============================================================
def external_force_schedule(sim_time):
    """
    模拟外力输入。

    在 2 秒到 6 秒之间：
        给导纳控制器输入 +x 方向 5N 外力。

    其他时间：
        外力为 0。

    注意：
        这里的外力不直接加到 MuJoCo 机械臂上。
        它只进入导纳控制器，用来生成 x_ref。
    """

    if 2.0 <= sim_time <= 6.0:
        F_ext = np.array([5.0, 0.0])
    else:
        F_ext = np.array([0.0, 0.0])

    return F_ext


# ============================================================
# 5. 更新黄色导纳参考点
# ============================================================
def update_admittance_ref_site(model, ref_site_id, x_ref):
    """
    把黄色点 admittance_ref_site 移动到当前 x_ref。

    x_ref 是二维坐标 [x, y]。
    MuJoCo site 需要三维坐标 [x, y, z]。
    """

    model.site_pos[ref_site_id] = np.array([
        x_ref[0],
        x_ref[1],
        0.09,
    ])


# ============================================================
# 6. 画图函数
# ============================================================
def plot_results(log):
    time_array = np.array(log["time"])
    x_array = np.array(log["x"])
    x_ref_array = np.array(log["x_ref"])
    x0_array = np.array(log["x0"])
    F_ext_array = np.array(log["F_ext"])
    tau_array = np.array(log["tau"])
    cost_array = np.array(log["cost"])

    tracking_error = x_ref_array - x_array
    tracking_error_norm = np.linalg.norm(tracking_error, axis=1)

    # ------------------------------------------------------------
    # 图 1：x 方向位置
    # ------------------------------------------------------------
    plt.figure()
    plt.plot(time_array, x_array[:, 0], label="actual x")
    plt.plot(time_array, x_ref_array[:, 0], "--", label="admittance reference x_ref")
    plt.plot(time_array, x0_array[:, 0], ":", label="original target x0")
    plt.xlabel("Time [s]")
    plt.ylabel("X position [m]")
    plt.title("Admittance + MPC: X Position")
    plt.grid(True)
    plt.legend()

    # ------------------------------------------------------------
    # 图 2：y 方向位置
    # ------------------------------------------------------------
    plt.figure()
    plt.plot(time_array, x_array[:, 1], label="actual y")
    plt.plot(time_array, x_ref_array[:, 1], "--", label="admittance reference y_ref")
    plt.plot(time_array, x0_array[:, 1], ":", label="original target y0")
    plt.xlabel("Time [s]")
    plt.ylabel("Y position [m]")
    plt.title("Admittance + MPC: Y Position")
    plt.grid(True)
    plt.legend()

    # ------------------------------------------------------------
    # 图 3：外力输入
    # ------------------------------------------------------------
    plt.figure()
    plt.plot(time_array, F_ext_array[:, 0], label="external force Fx")
    plt.plot(time_array, F_ext_array[:, 1], label="external force Fy")
    plt.xlabel("Time [s]")
    plt.ylabel("External force [N]")
    plt.title("External Force Input")
    plt.grid(True)
    plt.legend()

    # ------------------------------------------------------------
    # 图 4：MPC 跟踪误差
    # ------------------------------------------------------------
    plt.figure()
    plt.plot(time_array, tracking_error[:, 0], label="x_ref - x")
    plt.plot(time_array, tracking_error[:, 1], label="y_ref - y")
    plt.plot(time_array, tracking_error_norm, "--", label="tracking error norm")
    plt.xlabel("Time [s]")
    plt.ylabel("Tracking error [m]")
    plt.title("MPC Tracking Error")
    plt.grid(True)
    plt.legend()

    # ------------------------------------------------------------
    # 图 5：关节力矩
    # ------------------------------------------------------------
    plt.figure()
    plt.plot(time_array, tau_array[:, 0], label="tau1")
    plt.plot(time_array, tau_array[:, 1], label="tau2")
    plt.axhline(20.0, linestyle="--", label="torque upper limit")
    plt.axhline(-20.0, linestyle="--", label="torque lower limit")
    plt.xlabel("Time [s]")
    plt.ylabel("Joint torque [N.m]")
    plt.title("Admittance + MPC: Joint Torque")
    plt.grid(True)
    plt.legend()

    # ------------------------------------------------------------
    # 图 6：MPC cost
    # ------------------------------------------------------------
    plt.figure()
    plt.plot(time_array, cost_array, label="MPC cost")
    plt.xlabel("Time [s]")
    plt.ylabel("Cost")
    plt.title("Admittance + MPC: Optimization Cost")
    plt.grid(True)
    plt.legend()

    plt.show()


# ============================================================
# 7. 主程序
# ============================================================
def main():
    # ------------------------------------------------------------
    # 7.1 加载模型
    # ------------------------------------------------------------
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    # ------------------------------------------------------------
    # 7.2 设置初始状态
    # ------------------------------------------------------------
    data.qpos[:] = np.array([0.3, -0.9])
    data.qvel[:] = np.array([0.0, 0.0])

    mujoco.mj_forward(model, data)

    # ------------------------------------------------------------
    # 7.3 获取 site id
    # ------------------------------------------------------------
    ee_site_id = model.site("ee_site").id
    target_site_id = model.site("target_site").id
    ref_site_id = model.site("admittance_ref_site").id

    # 绿色点位置，作为导纳平衡位置 x0
    target_pos_3d = data.site_xpos[target_site_id].copy()
    x0 = target_pos_3d[:2]

    # 创建导纳控制器
    admittance_controller = AdmittanceController(x0=x0)

    # 创建 MPC 控制器
    mpc = TaskSpaceTorqueMPC(
        model=model,
        horizon=12,
        dt_mpc=0.03
    )

    # 黄色参考点一开始放在绿色目标点
    update_admittance_ref_site(
        model=model,
        ref_site_id=ref_site_id,
        x_ref=x0
    )

    mujoco.mj_forward(model, data)

    print("Model loaded successfully.")
    print("Model path:", MODEL_PATH)
    print("Green fixed target x0:", x0)
    print("Control structure: F_ext -> Admittance -> x_ref -> MPC -> tau")
    print("External force: 5 N in +x direction from t=2s to t=6s")
    print("MPC horizon:", mpc.N)
    print("MPC dt:", mpc.dt)
    print("Torque limit:", mpc.tau_min, mpc.tau_max)
    print("\nViewer meaning:")
    print("  red point    = real end-effector ee_site")
    print("  green point  = original fixed target x0")
    print("  yellow point = admittance reference x_ref")

    # ------------------------------------------------------------
    # 7.4 日志
    # ------------------------------------------------------------
    log = {
        "time": [],
        "x": [],
        "x_ref": [],
        "x0": [],
        "F_ext": [],
        "tau": [],
        "cost": [],
    }

    sim_duration = 10.0
    sim_dt = model.opt.timestep

    # 每 10 个 MuJoCo 步求解一次 MPC
    control_interval_steps = 10

    step_count = 0
    tau_cmd = np.array([0.0, 0.0])
    current_cost = 0.0

    # ------------------------------------------------------------
    # 7.5 打开 viewer 并运行仿真
    # ------------------------------------------------------------
    with mujoco.viewer.launch_passive(model, data) as viewer:

        while viewer.is_running() and data.time < sim_duration:
            step_start = time.time()

            sim_time = data.time
            dt = model.opt.timestep

            # 当前关节状态
            q = data.qpos.copy()
            dq = data.qvel.copy()

            # ----------------------------------------------------
            # 7.5.1 外力输入
            # ----------------------------------------------------
            F_ext = external_force_schedule(sim_time)

            # ----------------------------------------------------
            # 7.5.2 导纳控制器生成动态参考 x_ref
            # ----------------------------------------------------
            x_ref, dx_ref, ddx_ref = admittance_controller.update(
                F_ext=F_ext,
                dt=dt
            )

            # 更新黄色参考点
            update_admittance_ref_site(
                model=model,
                ref_site_id=ref_site_id,
                x_ref=x_ref
            )

            # ----------------------------------------------------
            # 7.5.3 MPC 跟踪 x_ref
            # ----------------------------------------------------
            if step_count % control_interval_steps == 0:
                tau_cmd, info = mpc.solve(
                    model=model,
                    data=data,
                    x_ref=x_ref,
                    dx_ref=dx_ref
                )

                tau_cmd = np.clip(tau_cmd, mpc.tau_min, mpc.tau_max)
                current_cost = info["cost"]

            # 输入关节力矩
            data.ctrl[:] = tau_cmd

            # 推进仿真
            mujoco.mj_step(model, data)

            # 更新 viewer
            viewer.sync()

            # ----------------------------------------------------
            # 7.5.4 记录当前真实末端状态
            # ----------------------------------------------------
            q_after = data.qpos.copy()
            dq_after = data.qvel.copy()

            x = forward_kinematics(q_after)
            J = jacobian(q_after)
            dx = J @ dq_after

            ee_pos = data.site_xpos[ee_site_id].copy()
            ref_pos = data.site_xpos[ref_site_id].copy()

            log["time"].append(data.time)
            log["x"].append(x.copy())
            log["x_ref"].append(x_ref.copy())
            log["x0"].append(x0.copy())
            log["F_ext"].append(F_ext.copy())
            log["tau"].append(tau_cmd.copy())
            log["cost"].append(current_cost)

            # ----------------------------------------------------
            # 7.5.5 打印信息
            # ----------------------------------------------------
            if step_count % 500 == 0:
                tracking_error = x_ref - x

                print(
                    f"t = {data.time:.2f} s, "
                    f"F_ext = [{F_ext[0]:.1f}, {F_ext[1]:.1f}], "
                    f"x_ref = [{x_ref[0]:.3f}, {x_ref[1]:.3f}], "
                    f"x = [{x[0]:.3f}, {x[1]:.3f}], "
                    f"err = [{tracking_error[0]:.3f}, {tracking_error[1]:.3f}], "
                    f"tau = [{tau_cmd[0]:.3f}, {tau_cmd[1]:.3f}], "
                    f"cost = {current_cost:.3f}, "
                    f"yellow = [{ref_pos[0]:.3f}, {ref_pos[1]:.3f}], "
                    f"red = [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}]"
                )

            step_count += 1

            # 控制仿真接近真实时间
            elapsed = time.time() - step_start

            if elapsed < sim_dt:
                time.sleep(sim_dt - elapsed)

    # ------------------------------------------------------------
    # 7.6 仿真结束后画图
    # ------------------------------------------------------------
    print("Simulation finished. Plotting results...")
    plot_results(log)


# ============================================================
# 8. 程序入口
# ============================================================
if __name__ == "__main__":
    main()