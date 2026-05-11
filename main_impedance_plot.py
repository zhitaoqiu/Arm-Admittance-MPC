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
# 2. 阻抗控制器
# ============================================================
def impedance_control(q, dq, x_des, dx_des):
    """
    末端阻抗控制器。

    输入：
        q      : 当前关节角度
        dq     : 当前关节角速度
        x_des  : 目标末端位置
        dx_des : 目标末端速度

    输出：
        tau_imp : 阻抗控制产生的关节力矩
        x       : 当前末端位置
        dx      : 当前末端速度
        F_imp   : 阻抗控制产生的末端虚拟力
        J       : 雅可比矩阵
    """

    # 当前末端位置
    x = forward_kinematics(q)

    # 当前雅可比矩阵
    J = jacobian(q)

    # 当前末端速度
    dx = J @ dq

    # ------------------------------------------------------------
    # 阻抗参数
    # ------------------------------------------------------------
    # K 越大，机械臂末端越“硬”
    # K 越小，机械臂末端越“软”
    K = np.array([60.0, 60.0])

    # D 是阻尼，用于减少振荡
    D = np.array([12.0, 12.0])

    # 位置误差
    position_error = x_des - x

    # 速度误差
    velocity_error = dx_des - dx

    # 末端虚拟弹簧-阻尼力
    F_imp = K * position_error + D * velocity_error

    # 末端力映射成关节力矩
    tau_imp = J.T @ F_imp

    return tau_imp, x, dx, F_imp, J


# ============================================================
# 3. 外力扰动函数
# ============================================================
def external_force_schedule(sim_time):
    """
    模拟外力。

    在 2 秒到 4 秒之间，给末端一个 +x 方向 5N 的推力。
    其他时间外力为 0。
    """

    if 2.0 <= sim_time <= 4.0:
        F_ext = np.array([5.0, 0.0])
    else:
        F_ext = np.array([0.0, 0.0])

    return F_ext


# ============================================================
# 4. 画图函数
# ============================================================
def plot_results(log):
    """
    根据仿真记录画图。

    log 是一个字典，里面保存了每个时刻的：
        time
        x
        x_des
        dx
        F_ext
        F_imp
        tau
    """

    time_array = np.array(log["time"])
    x_array = np.array(log["x"])
    x_des_array = np.array(log["x_des"])
    dx_array = np.array(log["dx"])
    F_ext_array = np.array(log["F_ext"])
    F_imp_array = np.array(log["F_imp"])
    tau_array = np.array(log["tau"])

    # ------------------------------------------------------------
    # 图 1：末端 x 方向位置
    # ------------------------------------------------------------
    plt.figure()
    plt.plot(time_array, x_array[:, 0], label="actual x")
    plt.plot(time_array, x_des_array[:, 0], "--", label="target x")
    plt.xlabel("Time [s]")
    plt.ylabel("End-effector x position [m]")
    plt.title("End-effector X Position")
    plt.grid(True)
    plt.legend()

    # ------------------------------------------------------------
    # 图 2：末端 y 方向位置
    # ------------------------------------------------------------
    plt.figure()
    plt.plot(time_array, x_array[:, 1], label="actual y")
    plt.plot(time_array, x_des_array[:, 1], "--", label="target y")
    plt.xlabel("Time [s]")
    plt.ylabel("End-effector y position [m]")
    plt.title("End-effector Y Position")
    plt.grid(True)
    plt.legend()

    # ------------------------------------------------------------
    # 图 3：外力
    # ------------------------------------------------------------
    plt.figure()
    plt.plot(time_array, F_ext_array[:, 0], label="external force Fx")
    plt.plot(time_array, F_ext_array[:, 1], label="external force Fy")
    plt.xlabel("Time [s]")
    plt.ylabel("External force [N]")
    plt.title("External Force")
    plt.grid(True)
    plt.legend()

    # ------------------------------------------------------------
    # 图 4：阻抗控制产生的末端力
    # ------------------------------------------------------------
    plt.figure()
    plt.plot(time_array, F_imp_array[:, 0], label="impedance force Fx")
    plt.plot(time_array, F_imp_array[:, 1], label="impedance force Fy")
    plt.xlabel("Time [s]")
    plt.ylabel("Impedance force [N]")
    plt.title("Impedance Control Force")
    plt.grid(True)
    plt.legend()

    # ------------------------------------------------------------
    # 图 5：关节力矩
    # ------------------------------------------------------------
    plt.figure()
    plt.plot(time_array, tau_array[:, 0], label="tau1")
    plt.plot(time_array, tau_array[:, 1], label="tau2")
    plt.xlabel("Time [s]")
    plt.ylabel("Joint torque [N.m]")
    plt.title("Joint Torque")
    plt.grid(True)
    plt.legend()

    # ------------------------------------------------------------
    # 图 6：末端速度
    # ------------------------------------------------------------
    plt.figure()
    plt.plot(time_array, dx_array[:, 0], label="dx")
    plt.plot(time_array, dx_array[:, 1], label="dy")
    plt.xlabel("Time [s]")
    plt.ylabel("End-effector velocity [m/s]")
    plt.title("End-effector Velocity")
    plt.grid(True)
    plt.legend()

    plt.show()


# ============================================================
# 5. 主程序
# ============================================================
def main():
    # ------------------------------------------------------------
    # 5.1 加载模型
    # ------------------------------------------------------------
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    # ------------------------------------------------------------
    # 5.2 设置初始状态
    # ------------------------------------------------------------
    data.qpos[:] = np.array([0.3, -0.9])
    data.qvel[:] = np.array([0.0, 0.0])

    mujoco.mj_forward(model, data)

    # ------------------------------------------------------------
    # 5.3 读取目标点
    # ------------------------------------------------------------
    # 这里要求 XML 里面已经有 target_site。
    target_site_id = model.site("target_site").id
    target_pos_3d = data.site_xpos[target_site_id].copy()

    # 只控制 xy 平面
    x_des = target_pos_3d[:2]
    dx_des = np.array([0.0, 0.0])

    print("Model loaded successfully.")
    print("Model path:", MODEL_PATH)
    print("Target end-effector position:", x_des)
    print("Simulation duration: 6 seconds")
    print("External force: 5 N in +x direction from t=2s to t=4s")

    # ------------------------------------------------------------
    # 5.4 创建日志字典
    # ------------------------------------------------------------
    # 这个字典用来保存仿真过程中的数据。
    log = {
        "time": [],
        "x": [],
        "x_des": [],
        "dx": [],
        "F_ext": [],
        "F_imp": [],
        "tau": [],
    }

    # 仿真总时长
    sim_duration = 6.0

    # 计数器
    step_count = 0

    # ------------------------------------------------------------
    # 5.5 打开 viewer 并运行仿真
    # ------------------------------------------------------------
    with mujoco.viewer.launch_passive(model, data) as viewer:

        while viewer.is_running() and data.time < sim_duration:
            step_start = time.time()

            # 当前仿真时间
            sim_time = data.time

            # 当前关节状态
            q = data.qpos.copy()
            dq = data.qvel.copy()

            # 阻抗控制
            tau_imp, x, dx, F_imp, J = impedance_control(
                q=q,
                dq=dq,
                x_des=x_des,
                dx_des=dx_des
            )

            # 外力
            F_ext = external_force_schedule(sim_time)

            # 把末端外力映射成关节力矩
            tau_ext = J.T @ F_ext

            # 总力矩
            tau_total = tau_imp + tau_ext

            # 限制力矩
            tau_total = np.clip(tau_total, -20.0, 20.0)

            # 输入给 MuJoCo
            data.ctrl[:] = tau_total

            # 推进仿真
            mujoco.mj_step(model, data)

            # 更新画面
            viewer.sync()

            # ----------------------------------------------------
            # 保存数据
            # ----------------------------------------------------
            log["time"].append(sim_time)
            log["x"].append(x.copy())
            log["x_des"].append(x_des.copy())
            log["dx"].append(dx.copy())
            log["F_ext"].append(F_ext.copy())
            log["F_imp"].append(F_imp.copy())
            log["tau"].append(tau_total.copy())

            # ----------------------------------------------------
            # 每隔 500 步打印一次简要信息
            # ----------------------------------------------------
            step_count += 1

            if step_count % 500 == 0:
                print(
                    f"t = {sim_time:.2f} s, "
                    f"x = [{x[0]:.3f}, {x[1]:.3f}], "
                    f"F_ext = [{F_ext[0]:.1f}, {F_ext[1]:.1f}], "
                    f"tau = [{tau_total[0]:.3f}, {tau_total[1]:.3f}]"
                )

            # 控制仿真速度接近真实时间
            dt = model.opt.timestep
            elapsed = time.time() - step_start

            if elapsed < dt:
                time.sleep(dt - elapsed)

    # ------------------------------------------------------------
    # 5.6 仿真结束后画图
    # ------------------------------------------------------------
    print("Simulation finished. Plotting results...")
    plot_results(log)


# ============================================================
# 6. 程序入口
# ============================================================
if __name__ == "__main__":
    main()