import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import mujoco
import numpy as np
import torch

from export_mujoco_video import get_gravity_orientation, get_heading, load_config, pd_control
from legged_gym import LEGGED_GYM_ROOT_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate teacher or distilled goal command policy in Mujoco.")
    parser.add_argument("--model", default="deploy/pre_train/g1/motion.pt")
    parser.add_argument("--goal-policy", help="TorchScript goal command policy. If omitted, use the hand-written teacher.")
    parser.add_argument("--config", default="g1.yaml")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--success-radius", type=float, default=0.45)
    parser.add_argument("--fall-height", type=float, default=0.45)
    parser.add_argument("--output", default="reports/g1_goal_student_grid25_eval.csv")
    return parser.parse_args()


def get_command(goal_vec, goal_dist, heading, goal_policy, success_radius):
    goal_dir = goal_vec / (goal_dist + 1e-6)
    goal_heading = np.arctan2(goal_dir[1], goal_dir[0])
    heading_error = (goal_heading - heading + np.pi) % (2 * np.pi) - np.pi

    if goal_policy is None:
        cmd = np.zeros(3, dtype=np.float32)
        cmd[0] = np.clip(goal_dist * 0.45, 0.0, 0.5)
        if goal_dist < success_radius:
            cmd[0] = 0.0
        cmd[1] = np.clip(goal_dir[1] * 0.15, -0.15, 0.15)
        cmd[2] = np.clip(heading_error * 0.8, -0.5, 0.5)
        return cmd

    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    body_goal_x = cos_h * goal_vec[0] + sin_h * goal_vec[1]
    body_goal_y = -sin_h * goal_vec[0] + cos_h * goal_vec[1]
    body_goal_dist = np.linalg.norm([body_goal_x, body_goal_y]) + 1e-6
    goal_obs = np.array(
        [
            np.clip(goal_dist / 4.5, 0.0, 1.5),
            body_goal_x / body_goal_dist,
            body_goal_y / body_goal_dist,
            heading_error / np.pi,
        ],
        dtype=np.float32,
    )
    with torch.no_grad():
        cmd = goal_policy(torch.from_numpy(goal_obs).unsqueeze(0)).numpy().squeeze().astype(np.float32)
    if goal_dist < success_radius:
        cmd[0] = 0.0
    return cmd


def evaluate_one(goal, args, config, gait_policy, goal_policy):
    xml_path = Path(config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR))
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    simulation_dt = config["simulation_dt"]
    model.opt.timestep = simulation_dt
    control_decimation = config["control_decimation"]
    kps = np.array(config["kps"], dtype=np.float32)
    kds = np.array(config["kds"], dtype=np.float32)
    default_angles = np.array(config["default_angles"], dtype=np.float32)
    cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
    num_actions = config["num_actions"]

    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(config["num_obs"], dtype=np.float32)
    start_xy = data.qpos[:2].copy()
    goal_world = start_xy + goal

    min_dist = float("inf")
    reach_time = ""
    fell = False

    for step in range(int(args.duration / simulation_dt)):
        torques = pd_control(
            target_dof_pos,
            data.qpos[7:],
            kps,
            np.zeros_like(kds),
            data.qvel[6:],
            kds,
        )
        data.ctrl[:] = torques
        mujoco.mj_step(model, data)

        goal_vec = goal_world - data.qpos[:2]
        goal_dist = float(np.linalg.norm(goal_vec))
        min_dist = min(min_dist, goal_dist)
        if reach_time == "" and goal_dist < args.success_radius:
            reach_time = f"{(step + 1) * simulation_dt:.3f}"
        if data.qpos[2] < args.fall_height:
            fell = True
            break

        counter = step + 1
        if counter % control_decimation != 0:
            continue

        qj = (data.qpos[7:] - default_angles) * config["dof_pos_scale"]
        dqj = data.qvel[6:] * config["dof_vel_scale"]
        omega = data.qvel[3:6] * config["ang_vel_scale"]
        gravity = get_gravity_orientation(data.qpos[3:7].copy())
        heading = get_heading(data.qpos[3:7].copy())
        cmd = get_command(goal_vec, goal_dist, heading, goal_policy, args.success_radius)

        phase = (counter * simulation_dt) % 0.8 / 0.8
        obs[:3] = omega
        obs[3:6] = gravity
        obs[6:9] = cmd * cmd_scale
        obs[9 : 9 + num_actions] = qj
        obs[9 + num_actions : 9 + 2 * num_actions] = dqj
        obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action
        obs[9 + 3 * num_actions : 9 + 3 * num_actions + 2] = np.array(
            [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)], dtype=np.float32
        )

        with torch.no_grad():
            action = gait_policy(torch.from_numpy(obs).unsqueeze(0)).numpy().squeeze()
        target_dof_pos = action * config["action_scale"] + default_angles

    final_dist = float(np.linalg.norm(goal_world - data.qpos[:2]))
    return {
        "goal_x": goal[0],
        "goal_y": goal[1],
        "final_dist": final_dist,
        "min_dist": min_dist,
        "reached": reach_time != "",
        "reach_time": reach_time,
        "fell": fell,
        "final_x": data.qpos[0] - start_xy[0],
        "final_y": data.qpos[1] - start_xy[1],
    }


def main():
    args = parse_args()
    config = load_config(args.config)
    gait_policy = torch.jit.load(str(Path(args.model).expanduser().resolve()))
    goal_policy = torch.jit.load(str(Path(args.goal_policy).expanduser().resolve())) if args.goal_policy else None

    goals = np.array(
        [(x, y) for x in [1.0, 1.75, 2.5, 3.25, 4.0] for y in [-1.0, -0.5, 0.0, 0.5, 1.0]],
        dtype=np.float32,
    )
    rows = []
    for idx, goal in enumerate(goals, start=1):
        row = evaluate_one(goal, args, config, gait_policy, goal_policy)
        row["index"] = idx
        rows.append(row)
        print(
            f"{idx:02d}/{len(goals)} goal=({row['goal_x']:.2f},{row['goal_y']:.2f}) "
            f"final={row['final_dist']:.3f} reached={row['reached']} fell={row['fell']}"
        )

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    final_dist = np.array([row["final_dist"] for row in rows], dtype=np.float32)
    reached = np.array([row["reached"] for row in rows])
    fell = np.array([row["fell"] for row in rows])
    reach_times = np.array([float(row["reach_time"]) for row in rows if row["reach_time"]])
    print(f"Saved CSV to: {output_path}")
    print(f"success={int(reached.sum())}/{len(rows)} rate={reached.mean():.3f}")
    print(f"falls={int(fell.sum())}/{len(rows)}")
    print(
        f"final_dist mean={final_dist.mean():.4f} median={np.median(final_dist):.4f} "
        f"p90={np.percentile(final_dist, 90):.4f} max={final_dist.max():.4f}"
    )
    if len(reach_times) > 0:
        print(
            f"reach_time mean={reach_times.mean():.4f} median={np.median(reach_times):.4f} "
            f"p90={np.percentile(reach_times, 90):.4f} max={reach_times.max():.4f}"
        )


if __name__ == "__main__":
    main()
