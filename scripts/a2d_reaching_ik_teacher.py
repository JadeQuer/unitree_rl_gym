import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import imageio.v2 as imageio
import mujoco
import numpy as np

from load_a2d_urdf import convert_package_mesh_paths


IK_JOINTS = [
    "joint_lift_body",
    "joint_body_pitch",
    "right_arm_joint1",
    "right_arm_joint2",
    "right_arm_joint3",
    "right_arm_joint4",
    "right_arm_joint5",
    "right_arm_joint6",
    "right_arm_joint7",
]


def parse_args():
    parser = argparse.ArgumentParser(description="A2D right-arm FK/IK teacher and small reaching validation.")
    parser.add_argument("--urdf", default="/root/autodl-tmp/A2D_dh/A2D_dh/A2D.urdf")
    parser.add_argument("--converted", default="reports/a2d_mujoco/A2D_mujoco.urdf")
    parser.add_argument("--ee-body", default="Link7_r")
    parser.add_argument("--num-targets", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-iters", type=int, default=120)
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--tol", type=float, default=0.05)
    parser.add_argument("--damping", type=float, default=0.04)
    parser.add_argument("--step-scale", type=float, default=0.65)
    parser.add_argument("--csv", default="reports/a2d_mujoco/a2d_reaching_ik_teacher.csv")
    parser.add_argument("--image", default="reports/a2d_mujoco/a2d_reaching_ik_teacher.png")
    parser.add_argument("--x-offset", type=float, nargs=2, default=(0.10, 0.38))
    parser.add_argument("--y-offset", type=float, nargs=2, default=(-0.02, 0.22))
    parser.add_argument("--z-offset", type=float, nargs=2, default=(-0.16, 0.16))
    return parser.parse_args()


def named_id(model, obj_type, name):
    idx = mujoco.mj_name2id(model, obj_type, name)
    if idx < 0:
        raise ValueError(f"Missing {obj_type} named {name}")
    return idx


def joint_info(model, joint_names):
    items = []
    for name in joint_names:
        jid = named_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qadr = int(model.jnt_qposadr[jid])
        dadr = int(model.jnt_dofadr[jid])
        limited = bool(model.jnt_limited[jid])
        lower, upper = model.jnt_range[jid]
        items.append(
            {
                "name": name,
                "jid": jid,
                "qadr": qadr,
                "dadr": dadr,
                "limited": limited,
                "lower": float(lower),
                "upper": float(upper),
            }
        )
    return items


def clamp_qpos(model, data, joints):
    for item in joints:
        if item["limited"]:
            qadr = item["qadr"]
            data.qpos[qadr] = np.clip(data.qpos[qadr], item["lower"], item["upper"])


def set_initial_qpos(data, joints, restart, rng):
    data.qpos[:] = 0.0
    if restart == 0:
        return
    for item in joints:
        qadr = item["qadr"]
        if item["limited"]:
            lower = item["lower"]
            upper = item["upper"]
            data.qpos[qadr] = rng.uniform(lower, upper)
        else:
            data.qpos[qadr] = rng.uniform(-0.5, 0.5)


def solve_ik_once(model, data, ee_body_id, joints, target, args, restart, rng):
    set_initial_qpos(data, joints, restart, rng)
    mujoco.mj_forward(model, data)
    initial_pos = data.xpos[ee_body_id].copy()
    best_qpos = data.qpos.copy()
    best_err = float(np.linalg.norm(target - initial_pos))
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    dof_ids = np.array([item["dadr"] for item in joints], dtype=np.int32)

    for iteration in range(args.max_iters):
        mujoco.mj_forward(model, data)
        ee_pos = data.xpos[ee_body_id].copy()
        err_vec = target - ee_pos
        err = float(np.linalg.norm(err_vec))
        if err < best_err:
            best_err = err
            best_qpos = data.qpos.copy()
        if err < args.tol:
            return best_qpos, best_err, iteration + 1, True, initial_pos

        mujoco.mj_jacBody(model, data, jacp, jacr, ee_body_id)
        j_task = jacp[:, dof_ids]
        lhs = j_task @ j_task.T + (args.damping**2) * np.eye(3)
        dq = j_task.T @ np.linalg.solve(lhs, err_vec)
        dq = np.clip(dq, -0.12, 0.12)

        for item, delta in zip(joints, dq):
            data.qpos[item["qadr"]] += args.step_scale * delta
        clamp_qpos(model, data, joints)

    data.qpos[:] = best_qpos
    mujoco.mj_forward(model, data)
    return best_qpos, best_err, args.max_iters, best_err < args.tol, initial_pos


def solve_ik(model, data, ee_body_id, joints, target, args, rng):
    best = None
    for restart in range(args.restarts):
        result = solve_ik_once(model, data, ee_body_id, joints, target, args, restart, rng)
        if best is None or result[1] < best[1]:
            best = result
        if result[3]:
            return result
    return best


def sample_targets(default_ee_pos, num_targets, seed, args):
    rng = np.random.default_rng(seed)
    targets = np.zeros((num_targets, 3), dtype=np.float64)
    # A conservative reachable box around the default right wrist position.
    targets[:, 0] = rng.uniform(default_ee_pos[0] + args.x_offset[0], default_ee_pos[0] + args.x_offset[1], size=num_targets)
    targets[:, 1] = rng.uniform(default_ee_pos[1] + args.y_offset[0], default_ee_pos[1] + args.y_offset[1], size=num_targets)
    targets[:, 2] = rng.uniform(default_ee_pos[2] + args.z_offset[0], default_ee_pos[2] + args.z_offset[1], size=num_targets)
    return targets


def add_marker(renderer, position, rgba, radius=0.035):
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


def render_solution(model, data, target, ee_body_id, image_path):
    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat = np.array([0.15, -0.45, 0.75])
    camera.distance = 2.2
    camera.azimuth = 120
    camera.elevation = -18
    renderer.update_scene(data, camera=camera)
    add_marker(renderer, target, [1.0, 0.15, 0.1, 1.0], radius=0.045)
    add_marker(renderer, data.xpos[ee_body_id].copy(), [0.1, 0.8, 1.0, 1.0], radius=0.035)
    image = renderer.render()
    image_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(image_path, image)
    renderer.close()


def main():
    args = parse_args()
    converted = Path(args.converted).expanduser().resolve()
    convert_package_mesh_paths(Path(args.urdf).expanduser().resolve(), converted)

    model = mujoco.MjModel.from_xml_path(str(converted))
    data = mujoco.MjData(model)
    ee_body_id = named_id(model, mujoco.mjtObj.mjOBJ_BODY, args.ee_body)
    joints = joint_info(model, IK_JOINTS)

    mujoco.mj_forward(model, data)
    default_ee_pos = data.xpos[ee_body_id].copy()
    targets = sample_targets(default_ee_pos, args.num_targets, args.seed, args)
    rng = np.random.default_rng(args.seed + 1000)

    rows = []
    best_success = None
    best_success_target = None

    print(f"model={converted}")
    print(f"ee_body={args.ee_body} default_ee_pos={np.array2string(default_ee_pos, precision=4)}")
    print("ik_joints=" + ", ".join(item["name"] for item in joints))

    for idx, target in enumerate(targets, start=1):
        qpos, final_err, iters, success, initial_pos = solve_ik(model, data, ee_body_id, joints, target, args, rng)
        final_pos = data.xpos[ee_body_id].copy()
        row = {
            "index": idx,
            "target_x": target[0],
            "target_y": target[1],
            "target_z": target[2],
            "initial_x": initial_pos[0],
            "initial_y": initial_pos[1],
            "initial_z": initial_pos[2],
            "final_x": final_pos[0],
            "final_y": final_pos[1],
            "final_z": final_pos[2],
            "final_error": final_err,
            "success": success,
            "iterations": iters,
        }
        for item in joints:
            row[item["name"]] = qpos[item["qadr"]]
        rows.append(row)
        if success and best_success is None:
            best_success = qpos.copy()
            best_success_target = target.copy()
        print(
            f"{idx:02d}/{len(targets)} target={np.array2string(target, precision=3)} "
            f"err={final_err:.4f} success={success} iters={iters}"
        )

    csv_path = Path(args.csv).expanduser().resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    errors = np.array([row["final_error"] for row in rows], dtype=np.float64)
    successes = np.array([row["success"] for row in rows], dtype=np.bool_)
    print(f"csv={csv_path}")
    print(f"success={int(successes.sum())}/{len(successes)} rate={successes.mean():.3f}")
    print(
        f"error_mean={errors.mean():.4f} error_median={np.median(errors):.4f} "
        f"error_p90={np.percentile(errors, 90):.4f} error_max={errors.max():.4f}"
    )

    if best_success is not None:
        data.qpos[:] = best_success
        mujoco.mj_forward(model, data)
        image_path = Path(args.image).expanduser().resolve()
        render_solution(model, data, best_success_target, ee_body_id, image_path)
        print(f"image={image_path}")


if __name__ == "__main__":
    main()
