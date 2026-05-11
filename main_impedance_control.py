import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

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

    核心思想：
        让机械臂末端表现得像一个弹簧-阻尼系统。
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
    # K 表示末端虚拟弹簧刚度。
    # K 越大，末端越“硬”，越不容易被推开。
    K = np.array([60.0, 60.0])

    # D 表示末端虚拟阻尼。
    # D 越大，末端越不容易振荡。
    D = np.array([12.0, 12.0])

    # 末端位置误差
    position_error = x_des - x

    # 末端速度误差
    velocity_error = dx_des - dx

    # ------------------------------------------------------------
    # 末端阻抗力
    # ------------------------------------------------------------
    # 这就是虚拟弹簧-阻尼模型：
    #
    # F = K * 位置误差 + D * 速度误差
    #
    # 如果末端偏离目标，弹簧项会把它拉回来。
    # 如果末端速度太大，阻尼项会让它慢下来。
    F_imp = K * position_error + D * velocity_error

    # ------------------------------------------------------------
    # 末端力映射成关节力矩
    # ------------------------------------------------------------
    tau_imp = J.T @ F_imp

    return tau_imp, x, dx, F_imp, J


# ============================================================
# 3. 外力扰动函数
# ============================================================
def external_force_schedule(sim_time):
    """
    模拟一个末端外力。

    在 2 秒到 4 秒之间，给末端一个 x 方向的推力。
    其他时间没有外力。

    注意：
        这里的 F_ext 是我们人为设计的虚拟外力。
        后面做真实接触墙时，会换成 MuJoCo 接触力。
    """

    if 2.0 <= sim_time <= 4.0:
        # x 方向 5 N 推力，y 方向 0 N
        F_ext = np.array([5.0, 0.0])
    else:
        F_ext = np.array([0.0, 0.0])

    return F_ext


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
    data.qpos[:] = np.array([0.3, -0.9])
    data.qvel[:] = np.array([0.0, 0.0])

    # 更新一次 MuJoCo 内部状态
    mujoco.mj_forward(model, data)

    # ------------------------------------------------------------
    # 4.3 读取目标点
    # ------------------------------------------------------------
    # 这里要求 XML 里面已经有 target_site。
    # target_site 是绿色目标点。
    target_site_id = model.site("target_site").id
    ee_site_id = model.site("ee_site").id

    # 从 MuJoCo 里读取绿色目标点位置
    target_pos_3d = data.site_xpos[target_site_id].copy()

    # 控制器只控制 xy 平面，所以取前两个分量
    x_des = target_pos_3d[:2]

    # 目标末端速度为 0，希望最后停住
    dx_des = np.array([0.0, 0.0])

    print("Model loaded successfully.")
    print("Model path:", MODEL_PATH)
    print("Target end-effector position x_des:", x_des)
    print("External force: 5 N in +x direction from t=2s to t=4s")

    step_count = 0

    # ------------------------------------------------------------
    # 4.4 打开 MuJoCo viewer
    # ------------------------------------------------------------
    with mujoco.viewer.launch_passive(model, data) as viewer:

        while viewer.is_running():
            step_start = time.time()

            # 当前仿真时间
            sim_time = data.time

            # ----------------------------------------------------
            # 读取当前关节状态
            # ----------------------------------------------------
            q = data.qpos.copy()
            dq = data.qvel.copy()

            # ----------------------------------------------------
            # 计算阻抗控制力矩
            # ----------------------------------------------------
            tau_imp, x, dx, F_imp, J = impedance_control(
                q=q,
                dq=dq,
                x_des=x_des,
                dx_des=dx_des
            )

            # ----------------------------------------------------
            # 计算外力扰动
            # ----------------------------------------------------
            # F_ext 是作用在末端的外力。
            F_ext = external_force_schedule(sim_time)

            # 把末端外力等效成关节力矩
            tau_ext = J.T @ F_ext

            # ----------------------------------------------------
            # 总关节力矩
            # ----------------------------------------------------
            # tau_imp 是控制器给的力矩
            # tau_ext 是外力扰动等效到关节上的力矩
            tau_total = tau_imp + tau_ext

            # 限制关节力矩范围
            tau_total = np.clip(tau_total, -20.0, 20.0)

            # 输入给 MuJoCo
            data.ctrl[:] = tau_total

            # 推进仿真
            mujoco.mj_step(model, data)

            # 更新 viewer
            viewer.sync()

            # ----------------------------------------------------
            # 打印调试信息
            # ----------------------------------------------------
            step_count += 1

            if step_count % 500 == 0:
                ee_pos_mujoco = data.site_xpos[ee_site_id].copy()

                print("\n================ Impedance Control Check ================")
                print("Simulation time:", sim_time)

                print("\nTarget position x_des:")
                print(x_des)

                print("\nCurrent end-effector position x:")
                print(x)

                print("\nMuJoCo end-effector position:")
                print(ee_pos_mujoco[:2])

                print("\nPosition error x_des - x:")
                print(x_des - x)

                print("\nEnd-effector velocity dx:")
                print(dx)

                print("\nImpedance force F_imp:")
                print(F_imp)

                print("\nExternal force F_ext:")
                print(F_ext)

                print("\nImpedance torque tau_imp:")
                print(tau_imp)

                print("\nExternal torque tau_ext:")
                print(tau_ext)

                print("\nTotal torque tau_total:")
                print(tau_total)
                print("=========================================================")

            # 控制仿真速度
            dt = model.opt.timestep
            elapsed = time.time() - step_start

            if elapsed < dt:
                time.sleep(dt - elapsed)


# ============================================================
# 5. 程序入口
# ============================================================
if __name__ == "__main__":
    main()