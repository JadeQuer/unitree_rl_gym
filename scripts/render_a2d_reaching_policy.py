import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch

from a2d_reaching_ik_teacher import IK_JOINTS, joint_info, named_id, sample_targets
from load_a2d_urdf import convert_package_mesh_paths
from train_a2d_reaching_delta_policy import clamp_q, get_joint_arrays, make_obs


def add_marker(renderer, position, rgba, radius):
    geom = renderer.scene.geoms[renderer.scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, radius, radius], dtype=np.float32),
        position.astype(np.float32),
        np.eye(3, dtype=np.float32).reshape(-1),
        np.array(rgba, dtype=np.float32),
    )
    geom.segid = -1
    renderer.scene.ngeom += 1


def parse_args():
    parser = argparse.ArgumentParser(description="Render A2D reaching policy rollout.")
    parser.add_argument("--urdf", default="/root/autodl-tmp/A2D_dh/A2D_dh/A2D.urdf")
    parser.add_argument("--converted", default="reports/a2d_mujoco/A2D_mujoco.urdf")
    parser.add_argument("--policy", default="logs/a2d_reaching/reaching_delta_policy_gpu.pt")
    parser.add_argument("--ee-body", default="Link7_r")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--target-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=70)
    parser.add_argument("--policy-step-scale", type=float, default=0.8)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--output", default="reports/a2d_mujoco/a2d_reaching_policy_demo.mp4")
    parser.add_argument("--x-offset", type=float, nargs=2, default=(0.10, 0.38))
    parser.add_argument("--y-offset", type=float, nargs=2, default=(-0.02, 0.22))
    parser.add_argument("--z-offset", type=float, nargs=2, default=(-0.16, 0.16))
    return parser.parse_args()


def main():
    args = parse_args()
    converted = Path(args.converted).expanduser().resolve()
    convert_package_mesh_paths(Path(args.urdf).expanduser().resolve(), converted)

    model = mujoco.MjModel.from_xml_path(str(converted))
    data = mujoco.MjData(model)
    ee_body_id = named_id(model, mujoco.mjtObj.mjOBJ_BODY, args.ee_body)
    joints = joint_info(model, IK_JOINTS)
    qadr, _dadr, lower, upper, center, half = get_joint_arrays(model, joints)
    mujoco.mj_forward(model, data)
    default_ee_pos = data.xpos[ee_body_id].copy()

    target_args = argparse.Namespace(x_offset=args.x_offset, y_offset=args.y_offset, z_offset=args.z_offset)
    targets = sample_targets(default_ee_pos, max(args.target_index + 1, 8), args.seed, target_args)
    target = targets[args.target_index]
    policy = torch.jit.load(str(Path(args.policy).expanduser().resolve()))

    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat = np.array([0.18, -0.55, 0.82])
    camera.distance = 2.1
    camera.azimuth = 120
    camera.elevation = -18

    frames = []
    q = np.zeros(len(qadr), dtype=np.float64)
    best_err = float("inf")
    for _ in range(args.steps):
        data.qpos[:] = 0.0
        data.qpos[qadr] = q
        mujoco.mj_forward(model, data)
        ee_pos = data.xpos[ee_body_id].copy()
        err_vec = target - ee_pos
        best_err = min(best_err, float(np.linalg.norm(err_vec)))

        renderer.update_scene(data, camera=camera)
        add_marker(renderer, target, [1.0, 0.12, 0.08, 1.0], 0.045)
        add_marker(renderer, ee_pos, [0.1, 0.65, 1.0, 1.0], 0.032)
        frames.append(renderer.render().copy())

        obs = torch.as_tensor(make_obs(q, err_vec, center, half), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            dq = policy(obs).numpy().squeeze()
        q = clamp_q(q + args.policy_step_scale * dq, lower, upper)

    data.qpos[:] = 0.0
    data.qpos[qadr] = q
    mujoco.mj_forward(model, data)
    final_err = float(np.linalg.norm(target - data.xpos[ee_body_id].copy()))
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output, fps=args.fps, codec="libx264", quality=7) as writer:
        for frame in frames:
            writer.append_data(frame)
    renderer.close()
    print(f"target={np.array2string(target, precision=4)}")
    print(f"best_error={best_err:.4f} final_error={final_err:.4f}")
    print(f"saved_video={output}")


if __name__ == "__main__":
    main()
