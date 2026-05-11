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
# 2. PD 力矩控制器
# ============================================================
def pd_torque_control(q, dq, q_des, dq_des):
    """
    关节空间 PD 力矩控制器。

    输入：
        q      : 当前关节角度
        dq     : 当前关节角速度
        q_des  : 目标关节角度
        dq_des : 目标关节角速度

    输出：
        tau    : 两个关节需要施加的力矩
    """

    # Kp 越大，机械臂越想快速靠近目标角度
    kp = np.array([40.0, 25.0])

    # Kd 越大，机械臂运动越不容易振荡
    kd = np.array([4.0, 3.0])

    # 位置误差
    position_error = q_des - q

    # 速度误差
    velocity_error = dq_des - dq

    # PD 控制律
    tau = kp * position_error + kd * velocity_error

    return tau


# ============================================================
# 3. 主程序
# ============================================================
def main():
    # ------------------------------------------------------------
    # 3.1 读取 MuJoCo 模型
    # ------------------------------------------------------------
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    # ------------------------------------------------------------
    # 3.2 设置初始状态
    # ------------------------------------------------------------
    data.qpos[:] = np.array([0.0, 0.0])
    data.qvel[:] = np.array([0.0, 0.0])

    # ------------------------------------------------------------
    # 3.3 设置目标关节角度
    # ------------------------------------------------------------
    # joint1 目标：45 度
    # joint2 目标：-90 度
    q_des = np.array([np.pi / 4.0, -np.pi / 2.0])
    dq_des = np.array([0.0, 0.0])

    # ------------------------------------------------------------
    # 3.4 更新 MuJoCo 内部状态
    # ------------------------------------------------------------
    mujoco.mj_forward(model, data)

    # ------------------------------------------------------------
    # 3.5 找到末端 site 的编号
    # ------------------------------------------------------------
    # XML 里我们写了：
    # <site name="ee_site" pos="0.4 0 0" .../>
    #
    # 这里就是通过名字 "ee_site" 找到它在 MuJoCo 里的 id。
    # 后面可以通过 data.site_xpos[ee_site_id] 读取末端位置。
    ee_site_id = model.site("ee_site").id

    # 用来控制打印频率
    step_count = 0

    print("Model loaded successfully.")
    print("Model path:", MODEL_PATH)
    print("Number of joints:", model.njnt)
    print("Number of actuators:", model.nu)
    print("End-effector site id:", ee_site_id)
    print("Target q_des:", q_des)

    # ------------------------------------------------------------
    # 3.6 打开 MuJoCo viewer
    # ------------------------------------------------------------
    with mujoco.viewer.launch_passive(model, data) as viewer:

        while viewer.is_running():
            step_start = time.time()

            # ----------------------------------------------------
            # 读取当前关节状态
            # ----------------------------------------------------
            q = data.qpos.copy()
            dq = data.qvel.copy()

            # ----------------------------------------------------
            # 计算 PD 控制力矩
            # ----------------------------------------------------
            tau = pd_torque_control(q, dq, q_des, dq_des)

            # 限制力矩范围
            tau = np.clip(tau, -20.0, 20.0)

            # 把力矩输入给 MuJoCo
            data.ctrl[:] = tau

            # ----------------------------------------------------
            # 推进仿真一步
            # ----------------------------------------------------
            mujoco.mj_step(model, data)

            # 更新画面
            viewer.sync()

            # ----------------------------------------------------
            # 每隔 500 步打印一次运动学信息
            # ----------------------------------------------------
            step_count += 1

            if step_count % 500 == 0:
                # MuJoCo 计算出来的末端位置
                # data.site_xpos[ee_site_id] 是三维坐标 [x, y, z]
                ee_pos_mujoco_3d = data.site_xpos[ee_site_id].copy()

                # 我们自己用公式算出来的末端位置
                # 这里是二维坐标 [x, y]
                ee_pos_kinematics_2d = forward_kinematics(q)

                # 当前雅可比矩阵
                J = jacobian(q)

                print("\n================ Kinematics Check ================")
                print("Current q:", q)
                print("Current dq:", dq)
                print("Torque tau:", tau)

                print("\nEnd-effector position from our kinematics:")
                print(ee_pos_kinematics_2d)

                print("\nEnd-effector position from MuJoCo:")
                print(ee_pos_mujoco_3d[:2])

                print("\nDifference:")
                print(ee_pos_kinematics_2d - ee_pos_mujoco_3d[:2])

                print("\nJacobian J:")
                print(J)
                print("==================================================")

            # ----------------------------------------------------
            # 控制仿真速度
            # ----------------------------------------------------
            dt = model.opt.timestep
            elapsed = time.time() - step_start

            if elapsed < dt:
                time.sleep(dt - elapsed)


# ============================================================
# 4. 程序入口
# ============================================================
if __name__ == "__main__":
    main()