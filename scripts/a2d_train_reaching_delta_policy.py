import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import mujoco
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a2d_reaching_ik_teacher import IK_JOINTS, add_marker, joint_info, named_id, sample_targets
from load_a2d_urdf import convert_package_mesh_paths


class DeltaReachingPolicy(nn.Module):
    def __init__(self, obs_mean, obs_std):
        super().__init__()
        self.register_buffer("obs_mean", torch.as_tensor(obs_mean, dtype=torch.float32))
        self.register_buffer("obs_std", torch.as_tensor(obs_std, dtype=torch.float32))
        self.net = nn.Sequential(
            nn.Linear(3 + len(IK_JOINTS), 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, len(IK_JOINTS)),
        )

    def forward(self, obs):
        x = (obs - self.obs_mean) / self.obs_std
        return 0.12 * torch.tanh(self.net(x))


def parse_args():
    parser = argparse.ArgumentParser(description="Train a closed-loop A2D reaching delta policy from Jacobian IK teacher.")
    parser.add_argument("--urdf", default="/root/autodl-tmp/A2D_dh/A2D_dh/A2D.urdf")
    parser.add_argument("--converted", default="reports/a2d_mujoco/A2D_mujoco.urdf")
    parser.add_argument("--ee-body", default="Link7_r")
    parser.add_argument("--teacher-targets", type=int, default=900)
    parser.add_argument("--eval-targets", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--tol", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=70)
    parser.add_argument("--damping", type=float, default=0.04)
    parser.add_argument("--step-scale", type=float, default=0.65)
    parser.add_argument("--x-offset", type=float, nargs=2, default=(0.10, 0.38))
    parser.add_argument("--y-offset", type=float, nargs=2, default=(-0.02, 0.22))
    parser.add_argument("--z-offset", type=float, nargs=2, default=(-0.16, 0.16))
    parser.add_argument("--policy", default="logs/a2d_reaching/a2d_reaching_delta_policy.pt")
    parser.add_argument("--dataset", default="reports/a2d_mujoco/a2d_reaching_delta_dataset.npz")
    parser.add_argument("--eval-csv", default="reports/a2d_mujoco/a2d_reaching_delta_policy_eval.csv")
    parser.add_argument("--image", default="reports/a2d_mujoco/a2d_reaching_delta_policy_eval.png")
    return parser.parse_args()


def build_model(args):
    converted = Path(args.converted).expanduser().resolve()
    convert_package_mesh_paths(Path(args.urdf).expanduser().resolve(), converted)
    model = mujoco.MjModel.from_xml_path(str(converted))
    data = mujoco.MjData(model)
    ee_body_id = named_id(model, mujoco.mjtObj.mjOBJ_BODY, args.ee_body)
    joints = joint_info(model, IK_JOINTS)
    mujoco.mj_forward(model, data)
    default_ee_pos = data.xpos[ee_body_id].copy()
    return model, data, ee_body_id, joints, default_ee_pos


def clamp_qpos(data, joints):
    for item in joints:
        if item["limited"]:
            data.qpos[item["qadr"]] = np.clip(data.qpos[item["qadr"]], item["lower"], item["upper"])


def selected_q(data, joints):
    return np.array([data.qpos[item["qadr"]] for item in joints], dtype=np.float32)


def teacher_delta(model, data, ee_body_id, joints, err_vec, args):
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    dof_ids = np.array([item["dadr"] for item in joints], dtype=np.int32)
    mujoco.mj_jacBody(model, data, jacp, jacr, ee_body_id)
    j_task = jacp[:, dof_ids]
    lhs = j_task @ j_task.T + (args.damping**2) * np.eye(3)
    dq = j_task.T @ np.linalg.solve(lhs, err_vec)
    return np.clip(dq, -0.12, 0.12).astype(np.float32)


def make_dataset(args, model, data, ee_body_id, joints, default_ee_pos):
    targets = sample_targets(default_ee_pos, args.teacher_targets, args.seed, args)
    obs_rows = []
    action_rows = []

    for tidx, target in enumerate(targets, start=1):
        data.qpos[:] = 0.0
        mujoco.mj_forward(model, data)
        for _ in range(args.max_steps):
            ee_pos = data.xpos[ee_body_id].copy()
            err_vec = target - ee_pos
            if np.linalg.norm(err_vec) < args.tol:
                break
            obs_rows.append(np.concatenate([err_vec.astype(np.float32), selected_q(data, joints)]))
            dq = teacher_delta(model, data, ee_body_id, joints, err_vec, args)
            action_rows.append(dq)
            for item, delta in zip(joints, dq):
                data.qpos[item["qadr"]] += args.step_scale * float(delta)
            clamp_qpos(data, joints)
            mujoco.mj_forward(model, data)
        if tidx % 100 == 0:
            print(f"teacher_traj {tidx}/{len(targets)} samples={len(obs_rows)}")

    obs = np.stack(obs_rows).astype(np.float32)
    actions = np.stack(action_rows).astype(np.float32)
    dataset_path = Path(args.dataset).expanduser().resolve()
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dataset_path, obs=obs, actions=actions)
    print(f"dataset={dataset_path} samples={len(obs)}")
    return obs, actions


def train(args, obs, actions):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = len(obs)
    perm_np = np.random.default_rng(args.seed).permutation(n)
    split = int(n * 0.9)
    train_idx = torch.from_numpy(perm_np[:split]).long().to(device)
    val_idx = torch.from_numpy(perm_np[split:]).long().to(device)
    x = torch.from_numpy(obs).float().to(device)
    y = torch.from_numpy(actions).float().to(device)
    obs_mean = obs[perm_np[:split]].mean(axis=0).astype(np.float32)
    obs_std = np.maximum(obs[perm_np[:split]].std(axis=0).astype(np.float32), 1e-3)
    policy = DeltaReachingPolicy(obs_mean, obs_std).to(device)
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(1, args.epochs + 1):
        order = train_idx[torch.randperm(len(train_idx), device=device)]
        losses = []
        for start in range(0, len(order), args.batch_size):
            idx = order[start : start + args.batch_size]
            pred = policy(x[idx])
            loss = F.mse_loss(pred, y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        if epoch == 1 or epoch % 20 == 0 or epoch == args.epochs:
            with torch.no_grad():
                pred = policy(x[val_idx])
                mae = torch.mean(torch.abs(pred - y[val_idx])).item()
                rmse = torch.sqrt(F.mse_loss(pred, y[val_idx])).item()
            print(f"epoch={epoch:03d} train_mse={np.mean(losses):.7f} val_mae={mae:.5f} val_rmse={rmse:.5f}")

    policy_path = Path(args.policy).expanduser().resolve()
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    scripted = torch.jit.script(policy.cpu())
    scripted.save(str(policy_path))
    print(f"policy={policy_path}")
    return scripted


def render_eval(model, data, ee_body_id, target, image_path):
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
    image_path.parent.mkdir(parents=True, exist_ok=True)
    import imageio.v2 as imageio

    imageio.imwrite(image_path, renderer.render())
    renderer.close()


def evaluate(args, policy, model, data, ee_body_id, joints, default_ee_pos):
    targets = sample_targets(default_ee_pos, args.eval_targets, args.seed + 5000, args)
    rows = []
    best_q = None
    best_target = None
    best_err = float("inf")

    for idx, target in enumerate(targets, start=1):
        data.qpos[:] = 0.0
        mujoco.mj_forward(model, data)
        reached_step = ""
        for step in range(args.max_steps):
            ee_pos = data.xpos[ee_body_id].copy()
            err_vec = target - ee_pos
            err = float(np.linalg.norm(err_vec))
            if err < args.tol:
                reached_step = step
                break
            obs = np.concatenate([err_vec.astype(np.float32), selected_q(data, joints)])
            with torch.no_grad():
                dq = policy(torch.from_numpy(obs).float().unsqueeze(0)).numpy().squeeze()
            for item, delta in zip(joints, dq):
                data.qpos[item["qadr"]] += args.step_scale * float(delta)
            clamp_qpos(data, joints)
            mujoco.mj_forward(model, data)

        final_pos = data.xpos[ee_body_id].copy()
        final_err = float(np.linalg.norm(target - final_pos))
        success = final_err < args.tol
        if final_err < best_err:
            best_err = final_err
            best_q = data.qpos.copy()
            best_target = target.copy()
        rows.append(
            {
                "index": idx,
                "target_x": target[0],
                "target_y": target[1],
                "target_z": target[2],
                "final_x": final_pos[0],
                "final_y": final_pos[1],
                "final_z": final_pos[2],
                "final_error": final_err,
                "success": success,
                "reached_step": reached_step,
            }
        )
        print(f"eval {idx:03d}/{len(targets)} err={final_err:.4f} success={success}")

    csv_path = Path(args.eval_csv).expanduser().resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    errors = np.array([row["final_error"] for row in rows], dtype=np.float64)
    successes = np.array([row["success"] for row in rows], dtype=np.bool_)
    print(f"eval_csv={csv_path}")
    print(f"closed_loop_success={int(successes.sum())}/{len(successes)} rate={successes.mean():.3f}")
    print(
        f"closed_loop_error_mean={errors.mean():.4f} median={np.median(errors):.4f} "
        f"p90={np.percentile(errors, 90):.4f} max={errors.max():.4f}"
    )
    if best_q is not None:
        data.qpos[:] = best_q
        mujoco.mj_forward(model, data)
        image_path = Path(args.image).expanduser().resolve()
        render_eval(model, data, ee_body_id, best_target, image_path)
        print(f"image={image_path}")


def main():
    args = parse_args()
    model, data, ee_body_id, joints, default_ee_pos = build_model(args)
    print(f"default_ee_pos={np.array2string(default_ee_pos, precision=4)}")
    obs, actions = make_dataset(args, model, data, ee_body_id, joints, default_ee_pos)
    policy = train(args, obs, actions)
    evaluate(args, policy, model, data, ee_body_id, joints, default_ee_pos)


if __name__ == "__main__":
    main()
