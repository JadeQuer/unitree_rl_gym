import argparse
import os
from pathlib import Path
from typing import Optional

os.environ.setdefault("MUJOCO_GL", "osmesa")

import imageio
import mujoco
import numpy as np
import torch
import yaml

from legged_gym import LEGGED_GYM_ROOT_DIR


def get_gravity_orientation(quaternion: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = quaternion
    gravity_orientation = np.zeros(3, dtype=np.float32)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation


def pd_control(
    target_q: np.ndarray,
    q: np.ndarray,
    kp: np.ndarray,
    target_dq: np.ndarray,
    dq: np.ndarray,
    kd: np.ndarray,
) -> np.ndarray:
    return (target_q - q) * kp + (target_dq - dq) * kd


def get_heading(quaternion: np.ndarray) -> float:
    qw, qx, qy, qz = quaternion
    return np.arctan2(
        2 * (qw * qz + qx * qy),
        1 - 2 * (qy * qy + qz * qz),
    )


def load_config(config_name: str) -> dict:
    config_path = Path(LEGGED_GYM_ROOT_DIR) / "deploy" / "deploy_mujoco" / "configs" / config_name
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def resolve_output_path(output: Optional[str], model_path: Path, suffix: str) -> Path:
    if output:
        return Path(output).expanduser().resolve()
    stem = model_path.stem
    return Path(LEGGED_GYM_ROOT_DIR) / "reports" / f"{stem}{suffix}"


def add_goal_marker(renderer: mujoco.Renderer, goal_xyz: np.ndarray) -> None:
    geom = renderer.scene.geoms[renderer.scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([0.06, 0.06, 0.06], dtype=np.float32),
        goal_xyz.astype(np.float32),
        np.eye(3, dtype=np.float32).reshape(-1),
        np.array([1.0, 0.2, 0.2, 1.0], dtype=np.float32),
    )
    geom.emission = 0.8
    geom.specular = 0.3
    geom.shininess = 0.8
    geom.segid = -1
    renderer.scene.ngeom = min(renderer.scene.ngeom + 1, len(renderer.scene.geoms))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a headless Mujoco walking video for a policy model."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to a torchscript policy model, e.g. deploy/pre_train/g1/motion.pt",
    )
    parser.add_argument(
        "--config",
        default="g1.yaml",
        help="Deploy config name under deploy/deploy_mujoco/configs/ (default: g1.yaml)",
    )
    parser.add_argument(
        "--output",
        help="Output media path. Defaults to reports/<model_stem>_walk.gif",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=4.0,
        help="Simulation duration in seconds (default: 4.0)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Output frames per second (default: 15)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=320,
        help="Output width in pixels (default: 320)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=180,
        help="Output height in pixels (default: 180)",
    )
    parser.add_argument(
        "--format",
        choices=("gif", "mp4"),
        default="mp4",
        help="Output format (default: gif)",
    )
    parser.add_argument(
        "--goal",
        type=float,
        nargs=2,
        default=(4.0, 0.0),
        metavar=("X", "Y"),
        help="Local goal position in the world frame relative to start (default: 4.0 0.0)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    config = load_config(args.config)
    xml_path = Path(
        config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
    )
    output_suffix = "_walk.gif" if args.format == "gif" else "_walk.mp4"
    output_path = resolve_output_path(args.output, model_path, output_suffix)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    simulation_dt = config["simulation_dt"]
    control_decimation = config["control_decimation"]
    kps = np.array(config["kps"], dtype=np.float32)
    kds = np.array(config["kds"], dtype=np.float32)
    default_angles = np.array(config["default_angles"], dtype=np.float32)
    ang_vel_scale = config["ang_vel_scale"]
    dof_pos_scale = config["dof_pos_scale"]
    dof_vel_scale = config["dof_vel_scale"]
    action_scale = config["action_scale"]
    cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
    num_actions = config["num_actions"]
    num_obs = config["num_obs"]
    cmd = np.array(config["cmd_init"], dtype=np.float32)
    goal = np.array(args.goal, dtype=np.float32)

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    model.opt.timestep = simulation_dt
    policy = torch.jit.load(str(model_path))
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 4.0
    camera.azimuth = 140
    camera.elevation = -20
    camera.lookat = np.array([0.0, 0.0, 0.8])

    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    steps = int(args.duration / simulation_dt)
    render_every = max(1, int((1 / args.fps) / simulation_dt))
    frames = []
    start_xy = None

    for step in range(steps):
        if start_xy is None:
            start_xy = data.qpos[:2].copy()

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

        counter = step + 1
        quat = data.qpos[3:7].copy()
        gravity = get_gravity_orientation(quat)

        if counter % control_decimation == 0:
            qj = (data.qpos[7:] - default_angles) * dof_pos_scale
            dqj = data.qvel[6:] * dof_vel_scale
            omega = data.qvel[3:6] * ang_vel_scale

            period = 0.8
            elapsed = counter * simulation_dt
            phase = elapsed % period / period
            sin_phase = np.sin(2 * np.pi * phase)
            cos_phase = np.cos(2 * np.pi * phase)

            goal_vec = goal + start_xy - data.qpos[:2]
            goal_dist = np.linalg.norm(goal_vec) + 1e-6
            goal_dir = goal_vec / goal_dist
            heading = get_heading(data.qpos[3:7].copy())
            goal_heading = np.arctan2(goal_dir[1], goal_dir[0])
            heading_error = (goal_heading - heading + np.pi) % (2 * np.pi) - np.pi

            cmd[0] = np.clip(goal_dist * 0.45, 0.0, 0.5)
            if goal_dist < 0.45:
                cmd[0] = 0.0
            cmd[1] = np.clip(goal_dir[1] * 0.15, -0.15, 0.15)
            cmd[2] = np.clip(heading_error * 0.8, -0.5, 0.5)

            obs[:3] = omega
            obs[3:6] = gravity
            obs[6:9] = cmd * cmd_scale
            obs[9 : 9 + num_actions] = qj
            obs[9 + num_actions : 9 + 2 * num_actions] = dqj
            obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action
            obs[9 + 3 * num_actions : 9 + 3 * num_actions + 2] = np.array(
                [sin_phase, cos_phase], dtype=np.float32
            )

            with torch.no_grad():
                action = policy(torch.from_numpy(obs).unsqueeze(0)).numpy().squeeze()
            target_dof_pos = action * action_scale + default_angles

        if step % render_every == 0:
            camera.lookat[:] = np.array([data.qpos[0], data.qpos[1], 0.8])
            renderer.update_scene(data, camera=camera)
            goal_world = np.array([start_xy[0] + goal[0], start_xy[1] + goal[1], 0.8], dtype=np.float32)
            add_goal_marker(renderer, goal_world)
            frames.append(renderer.render().copy())

    if args.format == "gif":
        imageio.mimsave(output_path, frames, duration=1 / args.fps)
    else:
        with imageio.get_writer(output_path, fps=args.fps, codec="libx264", quality=7) as writer:
            for frame in frames:
                writer.append_data(frame)

    print(f"Saved video to: {output_path}")
    print(f"Frames: {len(frames)}")


if __name__ == "__main__":
    main()
