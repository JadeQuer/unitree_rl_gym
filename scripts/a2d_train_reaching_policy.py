import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a2d_reaching_ik_teacher import (
    IK_JOINTS,
    add_marker,
    joint_info,
    named_id,
    render_solution,
    sample_targets,
    solve_ik,
)
from load_a2d_urdf import convert_package_mesh_paths


class ReachingPolicy(nn.Module):
    def __init__(self, target_scale, q_mean, q_scale):
        super().__init__()
        self.register_buffer("target_scale", torch.as_tensor(target_scale, dtype=torch.float32))
        self.register_buffer("q_mean", torch.as_tensor(q_mean, dtype=torch.float32))
        self.register_buffer("q_scale", torch.as_tensor(q_scale, dtype=torch.float32))
        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, len(IK_JOINTS)),
        )

    def forward(self, target_offset):
        x = target_offset / self.target_scale
        q_norm = self.net(x)
        return self.q_mean + q_norm * self.q_scale


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate an A2D learned reaching policy from IK teacher data.")
    parser.add_argument("--urdf", default="/root/autodl-tmp/A2D_dh/A2D_dh/A2D.urdf")
    parser.add_argument("--converted", default="reports/a2d_mujoco/A2D_mujoco.urdf")
    parser.add_argument("--ee-body", default="Link7_r")
    parser.add_argument("--train-samples", type=int, default=800)
    parser.add_argument("--eval-samples", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--tol", type=float, default=0.05)
    parser.add_argument("--max-iters", type=int, default=120)
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--damping", type=float, default=0.04)
    parser.add_argument("--step-scale", type=float, default=0.65)
    parser.add_argument("--x-offset", type=float, nargs=2, default=(0.10, 0.38))
    parser.add_argument("--y-offset", type=float, nargs=2, default=(-0.02, 0.22))
    parser.add_argument("--z-offset", type=float, nargs=2, default=(-0.16, 0.16))
    parser.add_argument("--policy", default="logs/a2d_reaching/a2d_reaching_policy.pt")
    parser.add_argument("--dataset", default="reports/a2d_mujoco/a2d_reaching_policy_dataset.csv")
    parser.add_argument("--eval-csv", default="reports/a2d_mujoco/a2d_reaching_policy_eval.csv")
    parser.add_argument("--image", default="reports/a2d_mujoco/a2d_reaching_policy_eval.png")
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


def make_teacher_dataset(args, model, data, ee_body_id, joints, default_ee_pos):
    requested = args.train_samples + args.eval_samples
    targets = sample_targets(default_ee_pos, int(requested * 1.25) + 64, args.seed, args)
    rng = np.random.default_rng(args.seed + 1000)
    rows = []
    inputs = []
    outputs = []

    for idx, target in enumerate(targets, start=1):
        qpos, err, iters, success, _ = solve_ik(model, data, ee_body_id, joints, target, args, rng)
        if not success:
            continue
        q_target = np.array([qpos[item["qadr"]] for item in joints], dtype=np.float32)
        target_offset = (target - default_ee_pos).astype(np.float32)
        inputs.append(target_offset)
        outputs.append(q_target)
        row = {
            "index": len(rows) + 1,
            "target_x": target[0],
            "target_y": target[1],
            "target_z": target[2],
            "offset_x": target_offset[0],
            "offset_y": target_offset[1],
            "offset_z": target_offset[2],
            "teacher_error": err,
            "iterations": iters,
        }
        for name, value in zip(IK_JOINTS, q_target):
            row[name] = float(value)
        rows.append(row)
        if idx % 100 == 0:
            print(f"teacher {idx}/{len(targets)} kept={len(rows)} last_err={err:.4f}")
        if len(rows) >= requested:
            break

    if len(rows) < requested:
        raise RuntimeError(f"Only generated {len(rows)} successful IK samples; reduce requested samples or expand IK settings.")

    dataset_path = Path(args.dataset).expanduser().resolve()
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dataset_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"dataset={dataset_path}")

    return np.stack(inputs), np.stack(outputs)


def train_policy(args, inputs, outputs):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = len(inputs)
    train_n = n - args.eval_samples
    x = torch.from_numpy(inputs).float().to(device)
    y = torch.from_numpy(outputs).float().to(device)

    target_scale = np.array(
        [
            max(abs(args.x_offset[0]), abs(args.x_offset[1])),
            max(abs(args.y_offset[0]), abs(args.y_offset[1])),
            max(abs(args.z_offset[0]), abs(args.z_offset[1])),
        ],
        dtype=np.float32,
    )
    q_mean = outputs[:train_n].mean(axis=0).astype(np.float32)
    q_scale = outputs[:train_n].std(axis=0).astype(np.float32)
    q_scale = np.maximum(q_scale, 0.05)
    policy = ReachingPolicy(target_scale, q_mean, q_scale).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(train_n, device=device)
        losses = []
        for start in range(0, train_n, args.batch_size):
            idx = perm[start : start + args.batch_size]
            pred = policy(x[idx])
            loss = F.mse_loss(pred, y[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            with torch.no_grad():
                val_pred = policy(x[train_n:])
                val_mae = torch.mean(torch.abs(val_pred - y[train_n:])).item()
                val_rmse = torch.sqrt(F.mse_loss(val_pred, y[train_n:])).item()
            print(f"epoch={epoch:03d} train_mse={np.mean(losses):.7f} val_mae={val_mae:.5f} val_rmse={val_rmse:.5f}")

    policy_path = Path(args.policy).expanduser().resolve()
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    scripted = torch.jit.script(policy.cpu())
    scripted.save(str(policy_path))
    print(f"policy={policy_path}")
    return scripted


def closed_loop_eval(args, policy, inputs, model, data, ee_body_id, joints, default_ee_pos):
    eval_inputs = inputs[-args.eval_samples :]
    rows = []
    best_qpos = None
    best_target = None
    best_err = float("inf")

    for idx, target_offset in enumerate(eval_inputs, start=1):
        target = default_ee_pos + target_offset
        with torch.no_grad():
            q_target = policy(torch.from_numpy(target_offset).float().unsqueeze(0)).numpy().squeeze()

        data.qpos[:] = 0.0
        # Simple closed-loop first-order joint target tracking.
        for _ in range(80):
            for item, desired in zip(joints, q_target):
                qadr = item["qadr"]
                desired = np.clip(desired, item["lower"], item["upper"]) if item["limited"] else desired
                data.qpos[qadr] += 0.12 * (desired - data.qpos[qadr])
            mujoco.mj_forward(model, data)

        final_pos = data.xpos[ee_body_id].copy()
        err = float(np.linalg.norm(target - final_pos))
        success = err < args.tol
        if err < best_err:
            best_err = err
            best_qpos = data.qpos.copy()
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
                "final_error": err,
                "success": success,
            }
        )
        print(f"eval {idx:03d}/{len(eval_inputs)} err={err:.4f} success={success}")

    eval_csv = Path(args.eval_csv).expanduser().resolve()
    eval_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    errors = np.array([row["final_error"] for row in rows], dtype=np.float64)
    successes = np.array([row["success"] for row in rows], dtype=np.bool_)
    print(f"eval_csv={eval_csv}")
    print(f"closed_loop_success={int(successes.sum())}/{len(successes)} rate={successes.mean():.3f}")
    print(
        f"closed_loop_error_mean={errors.mean():.4f} median={np.median(errors):.4f} "
        f"p90={np.percentile(errors, 90):.4f} max={errors.max():.4f}"
    )

    if best_qpos is not None:
        data.qpos[:] = best_qpos
        mujoco.mj_forward(model, data)
        image_path = Path(args.image).expanduser().resolve()
        render_solution(model, data, best_target, ee_body_id, image_path)
        print(f"image={image_path}")


def main():
    args = parse_args()
    model, data, ee_body_id, joints, default_ee_pos = build_model(args)
    print(f"default_ee_pos={np.array2string(default_ee_pos, precision=4)}")
    print("joints=" + ", ".join(IK_JOINTS))
    inputs, outputs = make_teacher_dataset(args, model, data, ee_body_id, joints, default_ee_pos)
    policy = train_policy(args, inputs, outputs)
    closed_loop_eval(args, policy, inputs, model, data, ee_body_id, joints, default_ee_pos)


if __name__ == "__main__":
    main()
