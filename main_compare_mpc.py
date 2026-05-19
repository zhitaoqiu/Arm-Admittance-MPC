"""
SciPy L-BFGS-B MPC vs OSQP MPC 对比实验。

运行方式:
    python main_compare_mpc.py

对比指标:
    - 末端轨迹 RMSE
    - 最大 / 平均力矩
    - 平均 / 最大求解时间
    - 求解成功率
"""
import csv
import time
from pathlib import Path

import numpy as np
import mujoco

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# ---- 导入两个 MPC 控制器 ----
from main_admittance_mpc_control import (
    AdmittanceController as AdmittanceController1,
    TaskSpaceTorqueMPC as SciPyMPC,
    external_force_schedule as force_schedule1,
)
from main_admittance_mpc_osqp_control import (
    AdmittanceController as AdmittanceController2,
    TaskSpaceOSQPMPC as OSQPMPC,
    external_force_schedule as force_schedule2,
)

from kinematics import forward_kinematics, jacobian

MODEL_PATH = Path(__file__).parent / "models" / "two_link_arm.xml"


def run_simulation(mpc_type, sim_duration=10.0):
    """
    运行一次仿真，返回 log 和统计。

    mpc_type: "scipy" 或 "osqp"
    """
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    data.qpos[:] = np.array([0.3, -0.9])
    data.qvel[:] = np.array([0.0, 0.0])
    mujoco.mj_forward(model, data)

    target_site_id = model.site("target_site").id
    target_pos_3d = data.site_xpos[target_site_id].copy()
    x0 = target_pos_3d[:2]

    if mpc_type == "scipy":
        admittance = AdmittanceController1(x0=x0)
        mpc = SciPyMPC(model=model, horizon=12, dt_mpc=0.03)
        force_fn = force_schedule1
    else:
        admittance = AdmittanceController2(x0=x0)
        mpc = OSQPMPC(model=model, horizon=12, dt_mpc=0.03)
        force_fn = force_schedule2

    log = {
        "time": [], "x": [], "x_ref": [], "F_ext": [],
        "tau": [], "cost": [],
    }
    solve_stats = {
        "solve_time_ms": [], "status_ok": [],
        "fallback_count": 0,
    }

    sim_dt = model.opt.timestep
    control_interval_steps = 10
    step_count = 0
    tau_cmd = np.array([0.0, 0.0])
    mpc_info = {
        "cost": None, "solve_time_ms": 0.0,
        "status": 0, "iters": 0, "status_ok": True, "fallback": False,
    }

    t_start = time.perf_counter()

    while data.time < sim_duration:
        sim_time = data.time
        dt = model.opt.timestep

        q = data.qpos.copy()
        dq = data.qvel.copy()

        F_ext = force_fn(sim_time)
        x_ref, dx_ref, _ddx_ref = admittance.update(F_ext=F_ext, dt=dt)

        if step_count % control_interval_steps == 0:
            if mpc_type == "scipy":
                t_s = time.perf_counter()
                tau_cmd, info = mpc.solve(
                    model=model, data=data, x_ref=x_ref, dx_ref=dx_ref
                )
                t_ms = (time.perf_counter() - t_s) * 1000.0
                tau_cmd = np.clip(tau_cmd, -20.0, 20.0)
                status_ok = info["success"]
                fallback = False
                cost_val = info["cost"]
            else:
                t_s = time.perf_counter()
                tau_cmd, info = mpc.solve(
                    model=model, data=data, x_ref=x_ref, dx_ref=dx_ref
                )
                t_ms = info["solve_time_ms"]
                status_ok = info["status_ok"]
                fallback = info["fallback"]
                cost_val = info["cost"] if info["cost"] is not None else 0.0
                if fallback:
                    solve_stats["fallback_count"] += 1

            solve_stats["solve_time_ms"].append(t_ms)
            solve_stats["status_ok"].append(status_ok)

            mpc_info = {
                "cost": cost_val,
                "solve_time_ms": t_ms,
                "status_ok": status_ok,
                "fallback": fallback,
            }

        data.ctrl[:] = tau_cmd
        mujoco.mj_step(model, data)

        q_after = data.qpos.copy()
        dq_after = data.qvel.copy()
        x = forward_kinematics(q_after)

        log["time"].append(data.time)
        log["x"].append(x.copy())
        log["x_ref"].append(x_ref.copy())
        log["F_ext"].append(F_ext.copy())
        log["tau"].append(tau_cmd.copy())
        log["cost"].append(
            mpc_info["cost"] if mpc_info["cost"] is not None else 0.0
        )

        step_count += 1

    wall_time = time.perf_counter() - t_start

    return log, solve_stats, wall_time


def compute_metrics(log, solve_stats, label):
    """从 log 和统计中计算对比指标。"""
    t_arr = np.array(log["time"])
    x_arr = np.array(log["x"])
    x_ref_arr = np.array(log["x_ref"])
    tau_arr = np.array(log["tau"])
    F_ext_arr = np.array(log["F_ext"])
    solvetimes = np.array(solve_stats["solve_time_ms"])
    status_ok_arr = np.array(solve_stats["status_ok"])

    # 末端跟踪 RMSE
    tracking_error = x_ref_arr - x_arr
    rmse_x = np.sqrt(np.mean(tracking_error[:, 0] ** 2))
    rmse_y = np.sqrt(np.mean(tracking_error[:, 1] ** 2))
    rmse_overall = np.sqrt(np.mean(np.sum(tracking_error ** 2, axis=1)))

    # 力矩统计
    max_tau = np.max(np.abs(tau_arr))
    mean_tau = np.mean(np.abs(tau_arr))

    # 求解时间
    avg_solve = np.mean(solvetimes)
    max_solve = np.max(solvetimes)
    min_solve = np.min(solvetimes)

    # 成功率
    success_rate = np.mean(status_ok_arr.astype(float)) * 100.0
    fallback_count = solve_stats.get("fallback_count", 0)

    return {
        "label": label,
        "rmse_x": rmse_x, "rmse_y": rmse_y, "rmse_overall": rmse_overall,
        "max_tau": max_tau, "mean_tau": mean_tau,
        "avg_solve_ms": avg_solve, "max_solve_ms": max_solve,
        "min_solve_ms": min_solve,
        "success_rate": success_rate,
        "fallback_count": fallback_count,
        "total_solves": len(solvetimes),
    }


def plot_comparison(log_scipy, log_osqp, metric_scipy, metric_osqp):
    """并排对比图。"""
    t_sci = np.array(log_scipy["time"])
    t_osqp = np.array(log_osqp["time"])
    x_sci = np.array(log_scipy["x"])
    x_osqp = np.array(log_osqp["x"])
    x_ref_sci = np.array(log_scipy["x_ref"])
    x_ref_osqp = np.array(log_osqp["x_ref"])
    tau_sci = np.array(log_scipy["tau"])
    tau_osqp = np.array(log_osqp["tau"])
    F_ext_sci = np.array(log_scipy["F_ext"])

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # (0,0) X tracking - SciPy
    ax = axes[0, 0]
    ax.plot(t_sci, x_sci[:, 0], label="actual x")
    ax.plot(t_sci, x_ref_sci[:, 0], "--", label="x_ref")
    ax.set_title(f"SciPy MPC — X Tracking (RMSE={metric_scipy['rmse_overall']*1000:.1f}mm)")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("X [m]")
    ax.grid(True); ax.legend()

    # (0,1) X tracking - OSQP
    ax = axes[0, 1]
    ax.plot(t_osqp, x_osqp[:, 0], label="actual x")
    ax.plot(t_osqp, x_ref_osqp[:, 0], "--", label="x_ref")
    ax.set_title(f"OSQP MPC — X Tracking (RMSE={metric_osqp['rmse_overall']*1000:.1f}mm)")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("X [m]")
    ax.grid(True); ax.legend()

    # (1,0) Torque - SciPy
    ax = axes[1, 0]
    ax.plot(t_sci, tau_sci[:, 0], label="tau1")
    ax.plot(t_sci, tau_sci[:, 1], label="tau2")
    ax.axhline(20, ls="--", color="gray"); ax.axhline(-20, ls="--", color="gray")
    ax.set_title(f"SciPy MPC — Torque (max|τ|={metric_scipy['max_tau']:.2f})")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Torque [N.m]")
    ax.grid(True); ax.legend()

    # (1,1) Torque - OSQP
    ax = axes[1, 1]
    ax.plot(t_osqp, tau_osqp[:, 0], label="tau1")
    ax.plot(t_osqp, tau_osqp[:, 1], label="tau2")
    ax.axhline(20, ls="--", color="gray"); ax.axhline(-20, ls="--", color="gray")
    ax.set_title(f"OSQP MPC — Torque (max|τ|={metric_osqp['max_tau']:.2f})")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Torque [N.m]")
    ax.grid(True); ax.legend()

    # (2,0) External force
    ax = axes[2, 0]
    ax.plot(t_sci, F_ext_sci[:, 0], label="Fx")
    ax.plot(t_sci, F_ext_sci[:, 1], label="Fy")
    ax.set_title("External Force Input")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Force [N]")
    ax.grid(True); ax.legend()

    # (2,1) Solve time comparison
    ax = axes[2, 1]
    ax.bar(
        [0, 1],
        [metric_scipy["avg_solve_ms"], metric_osqp["avg_solve_ms"]],
        yerr=[metric_scipy["max_solve_ms"] - metric_scipy["avg_solve_ms"],
              metric_osqp["max_solve_ms"] - metric_osqp["avg_solve_ms"]],
        color=["#1f77b4", "#ff7f0e"],
        capsize=8,
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels([
        f"SciPy\navg={metric_scipy['avg_solve_ms']:.1f}ms\nmax={metric_scipy['max_solve_ms']:.1f}ms",
        f"OSQP\navg={metric_osqp['avg_solve_ms']:.1f}ms\nmax={metric_osqp['max_solve_ms']:.1f}ms",
    ])
    ax.set_ylabel("Solve time [ms]")
    ax.set_title("MPC Solve Time Comparison")
    ax.grid(True, axis="y")

    plt.tight_layout()
    plt.show()


def print_comparison_table(m1, m2):
    """终端对比表。"""
    print("\n" + "=" * 70)
    print("  SciPy L-BFGS-B  vs  OSQP MPC  Comparison")
    print("=" * 70)
    rows = [
        ("End-effector RMSE X [mm]",        m1["rmse_x"]*1000,    m2["rmse_x"]*1000,    ".1f"),
        ("End-effector RMSE Y [mm]",        m1["rmse_y"]*1000,    m2["rmse_y"]*1000,    ".1f"),
        ("End-effector RMSE overall [mm]",  m1["rmse_overall"]*1000, m2["rmse_overall"]*1000, ".1f"),
        ("Max |torque| [N.m]",              m1["max_tau"],        m2["max_tau"],        ".2f"),
        ("Mean |torque| [N.m]",             m1["mean_tau"],       m2["mean_tau"],       ".3f"),
        ("Avg solve time [ms]",             m1["avg_solve_ms"],   m2["avg_solve_ms"],   ".3f"),
        ("Max solve time [ms]",             m1["max_solve_ms"],   m2["max_solve_ms"],   ".3f"),
        ("Min solve time [ms]",             m1["min_solve_ms"],   m2["min_solve_ms"],   ".3f"),
        ("Success rate [%]",                m1["success_rate"],   m2["success_rate"],   ".1f"),
        ("Fallback count",                  m1["fallback_count"], m2["fallback_count"], "d"),
        ("Total solves",                    m1["total_solves"],   m2["total_solves"],   "d"),
    ]
    header = f"  {'Metric':<34s} {'SciPy':>12s} {'OSQP':>12s}"
    print(header)
    print("  " + "-" * 60)
    for name, v1, v2, fmt in rows:
        print(f"  {name:<34s} {v1:{fmt}} {v2:{fmt}}")
    print("=" * 70)


def main():
    print("=" * 60)
    print("  Running SciPy L-BFGS-B MPC ...")
    print("=" * 60)
    log_scipy, stats_scipy, wall_scipy = run_simulation("scipy", sim_duration=10.0)
    metric_scipy = compute_metrics(log_scipy, stats_scipy, "SciPy")

    print("\n" + "=" * 60)
    print("  Running OSQP MPC ...")
    print("=" * 60)
    log_osqp, stats_osqp, wall_osqp = run_simulation("osqp", sim_duration=10.0)
    metric_osqp = compute_metrics(log_osqp, stats_osqp, "OSQP")

    print(f"\nSciPy wall time: {wall_scipy:.1f}s")
    print(f"OSQP  wall time: {wall_osqp:.1f}s")

    print_comparison_table(metric_scipy, metric_osqp)

    # 保存对比 CSV
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "compare_scipy_vs_osqp.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "SciPy", "OSQP"])
        for name, v1, v2, fmt in [
            ("RMSE X [mm]", metric_scipy["rmse_x"]*1000, metric_osqp["rmse_x"]*1000),
            ("RMSE Y [mm]", metric_scipy["rmse_y"]*1000, metric_osqp["rmse_y"]*1000),
            ("RMSE overall [mm]", metric_scipy["rmse_overall"]*1000, metric_osqp["rmse_overall"]*1000),
            ("Max torque [N.m]", metric_scipy["max_tau"], metric_osqp["max_tau"]),
            ("Mean torque [N.m]", metric_scipy["mean_tau"], metric_osqp["mean_tau"]),
            ("Avg solve time [ms]", metric_scipy["avg_solve_ms"], metric_osqp["avg_solve_ms"]),
            ("Max solve time [ms]", metric_scipy["max_solve_ms"], metric_osqp["max_solve_ms"]),
            ("Min solve time [ms]", metric_scipy["min_solve_ms"], metric_osqp["min_solve_ms"]),
            ("Success rate [%]", metric_scipy["success_rate"], metric_osqp["success_rate"]),
            ("Fallback count", metric_scipy["fallback_count"], metric_osqp["fallback_count"]),
            ("Total solves", metric_scipy["total_solves"], metric_osqp["total_solves"]),
            ("Wall time [s]", wall_scipy, wall_osqp),
        ]:
            writer.writerow([name, f"{v1:.4f}", f"{v2:.4f}"])
    print(f"\nComparison CSV saved to: {csv_path}")

    print("\nPlotting comparison...")
    plot_comparison(log_scipy, log_osqp, metric_scipy, metric_osqp)


if __name__ == "__main__":
    main()
