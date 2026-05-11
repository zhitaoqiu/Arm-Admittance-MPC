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
# 2. 末端空间 MPC 控制器
# ============================================================
class TaskSpaceTorqueMPC:
    """
    末端空间力矩 MPC 控制器。

    控制目标：
        让机械臂末端位置 x 接近目标位置 x_des。

    控制输入：
        两个关节力矩 tau。

    约束：
        tau_min <= tau <= tau_max

    和上一版关节 MPC 的区别：
        上一版 cost 惩罚的是 q - q_des。
        这一版 cost 惩罚的是 forward_kinematics(q) - x_des。
    """

    def __init__(self, model, horizon=12, dt_mpc=0.03):
        self.model = model

        # MPC 预测步数
        self.N = horizon

        # MPC 预测步长
        self.dt = dt_mpc

        # 两个关节
        self.nq = 2

        # 两个力矩输入
        self.nu = 2

        # 力矩约束
        self.tau_min = -20.0
        self.tau_max = 20.0

        # --------------------------------------------------------
        # 代价函数权重
        # --------------------------------------------------------
        # 末端位置误差权重
        # 越大，MPC 越努力让末端靠近目标点
        self.Qx = np.array([500.0, 500.0])

        # 末端速度权重
        # 越大，末端运动越平稳
        self.Qdx = np.array([10.0, 10.0])

        # 关节速度权重
        # 防止关节转得太快
        self.Qdq = np.array([1.0, 1.0])

        # 力矩使用权重
        # 越大，MPC 越不愿意用大力矩
        self.R = np.array([0.01, 0.01])

        # 终端末端位置误差权重
        self.Qx_terminal = np.array([1200.0, 1200.0])

        # 上一轮优化结果，用于 warm start
        self.previous_u_sequence = np.zeros((self.N, self.nu))

    def get_frozen_dynamics(self, model, data):
        """
        从 MuJoCo 当前状态中提取一个简化动力学模型。

        MuJoCo 真实动力学可以近似写成：

            M(q) * qdd + bias(q, dq) = tau

        所以：

            qdd = inv(M) * (tau - bias)

        为了让 MPC 计算简单，这里在一个预测窗口内固定 M 和 bias。
        """

        mujoco.mj_forward(model, data)

        M = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, M, data.qM)

        bias = data.qfrc_bias.copy() + data.qfrc_passive.copy()

        return M, bias

    def rollout_cost(self, u_flat, q0, dq0, x_des, M, bias):
        """
        MPC 代价函数。

        输入一串未来力矩 u_sequence，
        预测未来末端位置 x，
        然后计算这串力矩好不好。

        cost 越小，说明这串未来力矩越好。
        """

        # 优化器给的是一维数组，先恢复成 N x 2 的力矩序列
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
            # 根据预测出来的 q，计算预测末端位置和速度
            # ----------------------------------------------------
            x = forward_kinematics(q)
            J = jacobian(q)
            dx = J @ dq

            # ----------------------------------------------------
            # 计算误差
            # ----------------------------------------------------
            x_error = x - x_des

            # ----------------------------------------------------
            # 当前预测步代价
            # ----------------------------------------------------
            position_cost = np.sum(self.Qx * x_error ** 2)
            ee_velocity_cost = np.sum(self.Qdx * dx ** 2)
            joint_velocity_cost = np.sum(self.Qdq * dq ** 2)
            effort_cost = np.sum(self.R * tau ** 2)

            cost += (
                position_cost
                + ee_velocity_cost
                + joint_velocity_cost
                + effort_cost
            )

        # --------------------------------------------------------
        # 终端代价
        # --------------------------------------------------------
        # 希望预测窗口最后一步的末端位置尽量靠近目标点
        x_terminal = forward_kinematics(q)
        terminal_error = x_terminal - x_des

        cost += np.sum(self.Qx_terminal * terminal_error ** 2)

        return cost

    def solve(self, model, data, x_des):
        """
        求解 MPC。

        输入：
            当前 MuJoCo model/data
            目标末端位置 x_des

        输出：
            tau_cmd 当前要执行的关节力矩
            info    优化信息
        """

        q0 = data.qpos.copy()
        dq0 = data.qvel.copy()

        M, bias = self.get_frozen_dynamics(model, data)

        # warm start：
        # 上一次的 [u0, u1, ..., u11]
        # 这一次用 [u1, ..., u11, u11] 作为初值
        initial_guess_sequence = np.vstack([
            self.previous_u_sequence[1:],
            self.previous_u_sequence[-1:]
        ])

        u0 = initial_guess_sequence.reshape(-1)

        # 每一个未来力矩都加边界约束
        bounds = [
            (self.tau_min, self.tau_max)
            for _ in range(self.N * self.nu)
        ]

        result = minimize(
            fun=self.rollout_cost,
            x0=u0,
            args=(q0, dq0, x_des, M, bias),
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": 35,
                "ftol": 1e-4,
            }
        )

        optimal_sequence = result.x.reshape(self.N, self.nu)

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
# 3. 画图函数
# ============================================================
def plot_results(log, x_des):
    time_array = np.array(log["time"])
    x_array = np.array(log["x"])
    dx_array = np.array(log["dx"])
    q_array = np.array(log["q"])
    tau_array = np.array(log["tau"])
    cost_array = np.array(log["cost"])

    # 图 1：末端 x 位置
    plt.figure()
    plt.plot(time_array, x_array[:, 0], label="actual x")
    plt.plot(time_array, np.ones_like(time_array) * x_des[0], "--", label="target x")
    plt.xlabel("Time [s]")
    plt.ylabel("X position [m]")
    plt.title("Task-space MPC: End-effector X Position")
    plt.grid(True)
    plt.legend()

    # 图 2：末端 y 位置
    plt.figure()
    plt.plot(time_array, x_array[:, 1], label="actual y")
    plt.plot(time_array, np.ones_like(time_array) * x_des[1], "--", label="target y")
    plt.xlabel("Time [s]")
    plt.ylabel("Y position [m]")
    plt.title("Task-space MPC: End-effector Y Position")
    plt.grid(True)
    plt.legend()

    # 图 3：末端速度
    plt.figure()
    plt.plot(time_array, dx_array[:, 0], label="dx")
    plt.plot(time_array, dx_array[:, 1], label="dy")
    plt.xlabel("Time [s]")
    plt.ylabel("End-effector velocity [m/s]")
    plt.title("Task-space MPC: End-effector Velocity")
    plt.grid(True)
    plt.legend()

    # 图 4：关节角度
    plt.figure()
    plt.plot(time_array, q_array[:, 0], label="q1")
    plt.plot(time_array, q_array[:, 1], label="q2")
    plt.xlabel("Time [s]")
    plt.ylabel("Joint position [rad]")
    plt.title("Task-space MPC: Joint Position")
    plt.grid(True)
    plt.legend()

    # 图 5：关节力矩
    plt.figure()
    plt.plot(time_array, tau_array[:, 0], label="tau1")
    plt.plot(time_array, tau_array[:, 1], label="tau2")
    plt.axhline(20.0, linestyle="--", label="torque upper limit")
    plt.axhline(-20.0, linestyle="--", label="torque lower limit")
    plt.xlabel("Time [s]")
    plt.ylabel("Joint torque [N.m]")
    plt.title("Task-space MPC: Joint Torque")
    plt.grid(True)
    plt.legend()

    # 图 6：MPC cost
    plt.figure()
    plt.plot(time_array, cost_array, label="MPC cost")
    plt.xlabel("Time [s]")
    plt.ylabel("Cost")
    plt.title("Task-space MPC: Optimization Cost")
    plt.grid(True)
    plt.legend()

    plt.show()


# ============================================================
# 4. 主程序
# ============================================================
def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    # ------------------------------------------------------------
    # 初始关节状态
    # ------------------------------------------------------------
    # 不从 [0, 0] 开始，是因为完全伸直姿态比较接近奇异位置。
    # 这里给一个弯曲初始姿态。
    data.qpos[:] = np.array([0.3, -0.9])
    data.qvel[:] = np.array([0.0, 0.0])

    mujoco.mj_forward(model, data)

    # ------------------------------------------------------------
    # 从绿色目标点读取 x_des
    # ------------------------------------------------------------
    target_site_id = model.site("target_site").id
    target_pos_3d = data.site_xpos[target_site_id].copy()

    x_des = target_pos_3d[:2]

    # 红色末端点 id
    ee_site_id = model.site("ee_site").id

    # 创建末端空间 MPC
    mpc = TaskSpaceTorqueMPC(
        model=model,
        horizon=12,
        dt_mpc=0.03
    )

    print("Model loaded successfully.")
    print("Model path:", MODEL_PATH)
    print("Target end-effector position x_des:", x_des)
    print("MPC horizon:", mpc.N)
    print("MPC dt:", mpc.dt)
    print("Torque limit:", mpc.tau_min, mpc.tau_max)

    log = {
        "time": [],
        "x": [],
        "dx": [],
        "q": [],
        "dq": [],
        "tau": [],
        "cost": [],
    }

    sim_duration = 5.0
    sim_dt = model.opt.timestep

    # 每 10 个 MuJoCo 步求解一次 MPC
    control_interval_steps = 10

    step_count = 0
    tau_cmd = np.array([0.0, 0.0])
    current_cost = 0.0

    with mujoco.viewer.launch_passive(model, data) as viewer:

        while viewer.is_running() and data.time < sim_duration:
            step_start = time.time()

            # ----------------------------------------------------
            # 每隔一段时间求解一次 MPC
            # ----------------------------------------------------
            if step_count % control_interval_steps == 0:
                tau_cmd, info = mpc.solve(
                    model=model,
                    data=data,
                    x_des=x_des
                )

                tau_cmd = np.clip(tau_cmd, mpc.tau_min, mpc.tau_max)
                current_cost = info["cost"]

            # ----------------------------------------------------
            # 输入关节力矩
            # ----------------------------------------------------
            data.ctrl[:] = tau_cmd

            # 推进仿真
            mujoco.mj_step(model, data)

            # 更新 viewer
            viewer.sync()

            # ----------------------------------------------------
            # 当前真实末端状态
            # ----------------------------------------------------
            q = data.qpos.copy()
            dq = data.qvel.copy()
            x = forward_kinematics(q)
            J = jacobian(q)
            dx = J @ dq

            ee_pos = data.site_xpos[ee_site_id].copy()

            # ----------------------------------------------------
            # 记录数据
            # ----------------------------------------------------
            log["time"].append(data.time)
            log["x"].append(x.copy())
            log["dx"].append(dx.copy())
            log["q"].append(q.copy())
            log["dq"].append(dq.copy())
            log["tau"].append(tau_cmd.copy())
            log["cost"].append(current_cost)

            # ----------------------------------------------------
            # 打印简要信息
            # ----------------------------------------------------
            if step_count % 250 == 0:
                x_error = x_des - x

                print(
                    f"t = {data.time:.2f} s, "
                    f"x = [{x[0]:.3f}, {x[1]:.3f}], "
                    f"x_error = [{x_error[0]:.3f}, {x_error[1]:.3f}], "
                    f"ee_site = [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}], "
                    f"tau = [{tau_cmd[0]:.3f}, {tau_cmd[1]:.3f}], "
                    f"cost = {current_cost:.3f}"
                )

            step_count += 1

            elapsed = time.time() - step_start

            if elapsed < sim_dt:
                time.sleep(sim_dt - elapsed)

    print("Simulation finished. Plotting results...")
    plot_results(log, x_des)


# ============================================================
# 5. 程序入口
# ============================================================
if __name__ == "__main__":
    main()