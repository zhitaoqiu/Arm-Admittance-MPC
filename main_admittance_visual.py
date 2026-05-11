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
# 2. 导纳控制器
# ============================================================
class AdmittanceController:
    """
    二维导纳控制器。

    作用：
        把外力 F_ext 转换成一个新的参考位置 x_ref。

    导纳模型：
        M * ddx_ref + D * dx_ref + K * (x_ref - x0) = F_ext

    直观理解：
        外力越大，x_ref 偏移越多；
        K 越大，系统越硬，x_ref 偏移越小；
        D 越大，系统越不容易振荡；
        M 越大，响应越慢。
    """

    def __init__(self, x0):
        # 原始平衡位置，也就是绿色目标点
        self.x0 = x0.copy()

        # 导纳模型当前输出的参考位置
        self.x_ref = x0.copy()

        # 导纳模型当前输出的参考速度
        self.dx_ref = np.array([0.0, 0.0])

        # 虚拟质量
        self.M = np.array([1.0, 1.0])

        # 虚拟阻尼
        self.D = np.array([15.0, 15.0])

        # 虚拟刚度
        self.K = np.array([60.0, 60.0])

    def update(self, F_ext, dt):
        """
        根据外力更新导纳模型。

        输入：
            F_ext : 外力 [Fx, Fy]
            dt    : 仿真步长

        输出：
            x_ref   : 导纳生成的新参考位置
            dx_ref  : 导纳生成的新参考速度
            ddx_ref : 导纳生成的新参考加速度
        """

        # 由导纳方程移项得到加速度：
        #
        # M * ddx_ref + D * dx_ref + K * (x_ref - x0) = F_ext
        #
        # 所以：
        #
        # ddx_ref = (F_ext - D * dx_ref - K * (x_ref - x0)) / M
        ddx_ref = (
            F_ext
            - self.D * self.dx_ref
            - self.K * (self.x_ref - self.x0)
        ) / self.M

        # 用加速度积分得到速度
        self.dx_ref = self.dx_ref + ddx_ref * dt

        # 用速度积分得到位置
        self.x_ref = self.x_ref + self.dx_ref * dt

        return self.x_ref.copy(), self.dx_ref.copy(), ddx_ref.copy()


# ============================================================
# 3. 末端空间跟踪控制器
# ============================================================
def task_space_tracking_control(q, dq, x_ref, dx_ref):
    """
    末端空间跟踪控制器。

    作用：
        让真实机械臂末端 x 跟踪导纳生成的参考位置 x_ref。

    输入：
        q      : 当前关节角度
        dq     : 当前关节角速度
        x_ref  : 导纳生成的参考末端位置
        dx_ref : 导纳生成的参考末端速度

    输出：
        tau     : 关节力矩
        x       : 当前真实末端位置
        dx      : 当前真实末端速度
        F_track : 跟踪控制生成的末端虚拟力
        J       : 雅可比矩阵
    """

    # 当前真实末端位置
    x = forward_kinematics(q)

    # 当前雅可比矩阵
    J = jacobian(q)

    # 当前真实末端速度
    dx = J @ dq

    # 跟踪控制参数
    # 这两个参数负责让红色末端点跟住黄色参考点
    Kp_track = np.array([120.0, 120.0])
    Kd_track = np.array([20.0, 20.0])

    # 位置误差：黄色参考点 - 红色真实末端点
    position_error = x_ref - x

    # 速度误差：参考速度 - 真实末端速度
    velocity_error = dx_ref - dx

    # 末端跟踪虚拟力
    F_track = Kp_track * position_error + Kd_track * velocity_error

    # 用雅可比转置把末端力转换成关节力矩
    tau = J.T @ F_track

    return tau, x, dx, F_track, J


# ============================================================
# 4. 外力输入
# ============================================================
def external_force_schedule(sim_time):
    """
    模拟外力输入。

    2 秒到 4 秒之间：
        给导纳控制器输入 +x 方向 5N 外力。

    其他时间：
        外力为 0。
    """

    if 2.0 <= sim_time <= 4.0:
        F_ext = np.array([5.0, 0.0])
    else:
        F_ext = np.array([0.0, 0.0])

    return F_ext


# ============================================================
# 5. 更新 MuJoCo 中黄色参考点的位置
# ============================================================
def update_admittance_ref_site(model, ref_site_id, x_ref):
    """
    把 MuJoCo 里的黄色点 admittance_ref_site 移动到当前 x_ref 位置。

    注意：
        x_ref 是二维坐标 [x, y]。
        MuJoCo site 需要三维坐标 [x, y, z]。
        所以这里人为设置 z = 0.09，让黄色点稍微高一点，方便观察。
    """

    model.site_pos[ref_site_id] = np.array([
        x_ref[0],
        x_ref[1],
        0.09,
    ])


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
    # 6.2 设置初始关节状态
    # ------------------------------------------------------------
    data.qpos[:] = np.array([0.3, -0.9])
    data.qvel[:] = np.array([0.0, 0.0])

    # 更新一次状态，保证 site 位置是最新的
    mujoco.mj_forward(model, data)

    # ------------------------------------------------------------
    # 6.3 获取三个 site 的 id
    # ------------------------------------------------------------
    # 红色点：机械臂末端
    ee_site_id = model.site("ee_site").id

    # 绿色点：原始目标点，不动
    target_site_id = model.site("target_site").id

    # 黄色点：导纳生成的动态参考点，会动
    ref_site_id = model.site("admittance_ref_site").id

    # ------------------------------------------------------------
    # 6.4 读取绿色目标点位置，作为导纳平衡位置 x0
    # ------------------------------------------------------------
    target_pos_3d = data.site_xpos[target_site_id].copy()

    # 只取 x, y
    x0 = target_pos_3d[:2]

    # 创建导纳控制器
    admittance_controller = AdmittanceController(x0=x0)

    # 一开始黄色点和绿色点重合
    update_admittance_ref_site(
        model=model,
        ref_site_id=ref_site_id,
        x_ref=x0
    )

    mujoco.mj_forward(model, data)

    print("Model loaded successfully.")
    print("Model path:", MODEL_PATH)
    print("Green fixed target x0:", x0)
    print("Yellow admittance reference x_ref starts from:", x0)
    print("External force: 5 N in +x direction from t=2s to t=4s")
    print("\nViewer meaning:")
    print("  red point    = real end-effector ee_site")
    print("  green point  = original fixed target x0")
    print("  yellow point = admittance reference x_ref")

    step_count = 0
    sim_duration = 8.0

    # ------------------------------------------------------------
    # 6.5 打开 viewer
    # ------------------------------------------------------------
    with mujoco.viewer.launch_passive(model, data) as viewer:

        while viewer.is_running() and data.time < sim_duration:
            step_start = time.time()

            # 当前时间和步长
            sim_time = data.time
            dt = model.opt.timestep

            # ----------------------------------------------------
            # 读取当前关节状态
            # ----------------------------------------------------
            q = data.qpos.copy()
            dq = data.qvel.copy()

            # ----------------------------------------------------
            # 计算外力输入
            # ----------------------------------------------------
            F_ext = external_force_schedule(sim_time)

            # ----------------------------------------------------
            # 导纳控制器根据外力生成 x_ref
            # ----------------------------------------------------
            x_ref, dx_ref, ddx_ref = admittance_controller.update(
                F_ext=F_ext,
                dt=dt
            )

            # ----------------------------------------------------
            # 更新黄色点的位置
            # ----------------------------------------------------
            update_admittance_ref_site(
                model=model,
                ref_site_id=ref_site_id,
                x_ref=x_ref
            )

            # ----------------------------------------------------
            # 真实机械臂末端跟踪黄色点
            # ----------------------------------------------------
            tau, x, dx, F_track, J = task_space_tracking_control(
                q=q,
                dq=dq,
                x_ref=x_ref,
                dx_ref=dx_ref
            )

            # 限制关节力矩
            tau = np.clip(tau, -20.0, 20.0)

            # 输入给 MuJoCo
            data.ctrl[:] = tau

            # 推进仿真
            mujoco.mj_step(model, data)

            # 更新画面
            viewer.sync()

            # ----------------------------------------------------
            # 打印简要信息
            # ----------------------------------------------------
            step_count += 1

            if step_count % 500 == 0:
                ee_pos = data.site_xpos[ee_site_id].copy()
                ref_pos = data.site_xpos[ref_site_id].copy()

                print(
                    f"t = {sim_time:.2f} s, "
                    f"F_ext = [{F_ext[0]:.1f}, {F_ext[1]:.1f}], "
                    f"x_ref = [{x_ref[0]:.3f}, {x_ref[1]:.3f}], "
                    f"x = [{x[0]:.3f}, {x[1]:.3f}], "
                    f"ee_site = [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}], "
                    f"yellow_site = [{ref_pos[0]:.3f}, {ref_pos[1]:.3f}]"
                )

            # ----------------------------------------------------
            # 控制仿真接近真实时间
            # ----------------------------------------------------
            elapsed = time.time() - step_start

            if elapsed < dt:
                time.sleep(dt - elapsed)

    print("Simulation finished.")


# ============================================================
# 7. 程序入口
# ============================================================
if __name__ == "__main__":
    main()