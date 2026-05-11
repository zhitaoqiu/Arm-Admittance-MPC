import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

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
        新的末端参考位置 x_ref
        新的末端参考速度 dx_ref

    导纳模型：
        M * ddx_ref + D * dx_ref + K * (x_ref - x0) = F_ext

    直观理解：
        外力越大，x_ref 偏移越多；
        K 越大，系统越硬，偏移越小；
        D 越大，系统越不容易振荡；
        M 越大，响应越慢。
    """

    def __init__(self, x0):
        """
        x0:
            原始平衡位置，也就是绿色目标点位置。
        """

        # 平衡位置
        self.x0 = x0.copy()

        # 当前导纳模型输出的参考位置
        self.x_ref = x0.copy()

        # 当前导纳模型输出的参考速度
        self.dx_ref = np.array([0.0, 0.0])

        # --------------------------------------------------------
        # 导纳参数
        # --------------------------------------------------------
        # 虚拟质量
        self.M = np.array([1.0, 1.0])

        # 虚拟阻尼
        self.D = np.array([15.0, 15.0])

        # 虚拟刚度
        self.K = np.array([60.0, 60.0])

    def update(self, F_ext, dt):
        """
        根据外力 F_ext 更新导纳模型状态。

        输入：
            F_ext : 外力，二维向量 [Fx, Fy]
            dt    : 仿真步长

        输出：
            x_ref   : 新的参考位置
            dx_ref  : 新的参考速度
            ddx_ref : 新的参考加速度
        """

        # --------------------------------------------------------
        # 由导纳模型计算参考加速度
        # --------------------------------------------------------
        # M * ddx_ref + D * dx_ref + K * (x_ref - x0) = F_ext
        #
        # 移项得到：
        # ddx_ref = (F_ext - D * dx_ref - K * (x_ref - x0)) / M
        ddx_ref = (
            F_ext
            - self.D * self.dx_ref
            - self.K * (self.x_ref - self.x0)
        ) / self.M

        # --------------------------------------------------------
        # 数值积分，更新参考速度和参考位置
        # --------------------------------------------------------
        self.dx_ref = self.dx_ref + ddx_ref * dt
        self.x_ref = self.x_ref + self.dx_ref * dt

        return self.x_ref.copy(), self.dx_ref.copy(), ddx_ref.copy()


# ============================================================
# 3. 末端空间轨迹跟踪控制器
# ============================================================
def task_space_tracking_control(q, dq, x_ref, dx_ref):
    """
    末端空间轨迹跟踪控制器。

    作用：
        让真实机械臂末端 x 去跟踪导纳控制器生成的 x_ref。

    输入：
        q      : 当前关节角度
        dq     : 当前关节角速度
        x_ref  : 导纳控制器生成的参考末端位置
        dx_ref : 导纳控制器生成的参考末端速度

    输出：
        tau     : 关节力矩
        x       : 当前真实末端位置
        dx      : 当前真实末端速度
        F_track : 轨迹跟踪产生的末端虚拟力
        J       : 雅可比矩阵
    """

    # 当前末端位置
    x = forward_kinematics(q)

    # 雅可比矩阵
    J = jacobian(q)

    # 当前末端速度
    dx = J @ dq

    # ------------------------------------------------------------
    # 跟踪控制参数
    # ------------------------------------------------------------
    # 这里的 Kp_track / Kd_track 是内环跟踪参数。
    # 它们负责让真实末端位置 x 跟上导纳模型输出的 x_ref。
    Kp_track = np.array([120.0, 120.0])
    Kd_track = np.array([20.0, 20.0])

    # 位置误差
    position_error = x_ref - x

    # 速度误差
    velocity_error = dx_ref - dx

    # 末端跟踪虚拟力
    F_track = Kp_track * position_error + Kd_track * velocity_error

    # 映射成关节力矩
    tau = J.T @ F_track

    return tau, x, dx, F_track, J


# ============================================================
# 4. 外力输入函数
# ============================================================
def external_force_schedule(sim_time):
    """
    模拟外力输入。

    在 2 秒到 4 秒之间，给导纳控制器输入一个 +x 方向 5N 的外力。
    其他时间外力为 0。

    注意：
        在导纳控制里，这个外力不是直接加到 data.ctrl 上。
        它是作为导纳模型的输入，用来生成 x_ref。
    """

    if 2.0 <= sim_time <= 4.0:
        F_ext = np.array([5.0, 0.0])
    else:
        F_ext = np.array([0.0, 0.0])

    return F_ext


# ============================================================
# 5. 画图函数
# ============================================================
def plot_results(log):
    """
    画出导纳控制实验结果。
    """

    time_array = np.array(log["time"])
    x_array = np.array(log["x"])
    x_ref_array = np.array(log["x_ref"])
    x0_array = np.array(log["x0"])
    F_ext_array = np.array(log["F_ext"])
    tau_array = np.array(log["tau"])

    # ------------------------------------------------------------
    # 图 1：末端 x 方向位置
    # ------------------------------------------------------------
    plt.figure()
    plt.plot(time_array, x_array[:, 0], label="actual x")
    plt.plot(time_array, x_ref_array[:, 0], "--", label="admittance reference x_ref")
    plt.plot(time_array, x0_array[:, 0], ":", label="original target x0")
    plt.xlabel("Time [s]")
    plt.ylabel("X position [m]")
    plt.title("Admittance Control: X Position")
    plt.grid(True)
    plt.legend()

    # ------------------------------------------------------------
    # 图 2：末端 y 方向位置
    # ------------------------------------------------------------
    plt.figure()
    plt.plot(time_array, x_array[:, 1], label="actual y")
    plt.plot(time_array, x_ref_array[:, 1], "--", label="admittance reference y_ref")
    plt.plot(time_array, x0_array[:, 1], ":", label="original target y0")
    plt.xlabel("Time [s]")
    plt.ylabel("Y position [m]")
    plt.title("Admittance Control: Y Position")
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
    # 图 4：关节力矩
    # ------------------------------------------------------------
    plt.figure()
    plt.plot(time_array, tau_array[:, 0], label="tau1")
    plt.plot(time_array, tau_array[:, 1], label="tau2")
    plt.xlabel("Time [s]")
    plt.ylabel("Joint torque [N.m]")
    plt.title("Joint Torque")
    plt.grid(True)
    plt.legend()

    plt.show()


# ============================================================
# 6. 主程序
# ============================================================
def main():
    # ------------------------------------------------------------
    # 6.1 加载 MuJoCo 模型
    # ------------------------------------------------------------
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    # ------------------------------------------------------------
    # 6.2 设置初始状态
    # ------------------------------------------------------------
    data.qpos[:] = np.array([0.3, -0.9])
    data.qvel[:] = np.array([0.0, 0.0])

    # 更新 MuJoCo 内部状态
    mujoco.mj_forward(model, data)

    # ------------------------------------------------------------
    # 6.3 读取绿色目标点
    # ------------------------------------------------------------
    # 要求 XML 中已经有：
    # <site name="target_site" .../>
    target_site_id = model.site("target_site").id
    target_pos_3d = data.site_xpos[target_site_id].copy()

    # 原始平衡位置 x0，也就是绿色目标点的 xy 坐标
    x0 = target_pos_3d[:2]

    # 创建导纳控制器
    admittance_controller = AdmittanceController(x0=x0)

    print("Model loaded successfully.")
    print("Model path:", MODEL_PATH)
    print("Original target position x0:", x0)
    print("Simulation duration: 6 seconds")
    print("External force: 5 N in +x direction from t=2s to t=4s")

    # ------------------------------------------------------------
    # 6.4 日志
    # ------------------------------------------------------------
    log = {
        "time": [],
        "x": [],
        "x_ref": [],
        "x0": [],
        "F_ext": [],
        "tau": [],
    }

    sim_duration = 6.0
    step_count = 0

    # ------------------------------------------------------------
    # 6.5 打开 viewer 运行仿真
    # ------------------------------------------------------------
    with mujoco.viewer.launch_passive(model, data) as viewer:

        while viewer.is_running() and data.time < sim_duration:
            step_start = time.time()

            # 当前仿真时间
            sim_time = data.time

            # 当前仿真步长
            dt = model.opt.timestep

            # 当前关节状态
            q = data.qpos.copy()
            dq = data.qvel.copy()

            # ----------------------------------------------------
            # 6.5.1 外力输入
            # ----------------------------------------------------
            F_ext = external_force_schedule(sim_time)

            # ----------------------------------------------------
            # 6.5.2 导纳模型更新参考轨迹
            # ----------------------------------------------------
            x_ref, dx_ref, ddx_ref = admittance_controller.update(
                F_ext=F_ext,
                dt=dt
            )

            # ----------------------------------------------------
            # 6.5.3 内环末端空间跟踪控制
            # ----------------------------------------------------
            tau, x, dx, F_track, J = task_space_tracking_control(
                q=q,
                dq=dq,
                x_ref=x_ref,
                dx_ref=dx_ref
            )

            # 限制力矩
            tau = np.clip(tau, -20.0, 20.0)

            # 输入 MuJoCo
            data.ctrl[:] = tau

            # 推进仿真
            mujoco.mj_step(model, data)

            # 更新画面
            viewer.sync()

            # ----------------------------------------------------
            # 6.5.4 记录数据
            # ----------------------------------------------------
            log["time"].append(sim_time)
            log["x"].append(x.copy())
            log["x_ref"].append(x_ref.copy())
            log["x0"].append(x0.copy())
            log["F_ext"].append(F_ext.copy())
            log["tau"].append(tau.copy())

            # ----------------------------------------------------
            # 6.5.5 打印简要信息
            # ----------------------------------------------------
            step_count += 1

            if step_count % 500 == 0:
                print(
                    f"t = {sim_time:.2f} s, "
                    f"x = [{x[0]:.3f}, {x[1]:.3f}], "
                    f"x_ref = [{x_ref[0]:.3f}, {x_ref[1]:.3f}], "
                    f"F_ext = [{F_ext[0]:.1f}, {F_ext[1]:.1f}], "
                    f"tau = [{tau[0]:.3f}, {tau[1]:.3f}]"
                )

            # ----------------------------------------------------
            # 6.5.6 控制仿真速度接近真实时间
            # ----------------------------------------------------
            elapsed = time.time() - step_start

            if elapsed < dt:
                time.sleep(dt - elapsed)

    # ------------------------------------------------------------
    # 6.6 仿真结束后画图
    # ------------------------------------------------------------
    print("Simulation finished. Plotting results...")
    plot_results(log)


# ============================================================
# 7. 程序入口
# ============================================================
if __name__ == "__main__":
    main()