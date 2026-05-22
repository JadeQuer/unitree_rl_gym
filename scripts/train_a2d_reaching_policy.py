import argparse
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a2d_reaching_ik_teacher import IK_JOINTS, joint_info, named_id, sample_targets, solve_ik
from load_a2d_urdf import convert_package_mesh_paths


class A2DReachingPolicy(nn.Module):
    def __init__(self, q_lower, q_upper):
        super().__init__()
        self.register_buffer("q_lower", torch.as_tensor(q_lower, dtype=torch.float32))
        self.register_buffer("q_upper", torch.as_tensor(q_upper, dtype=torch.float32))
        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, 9),
        )

    def forward(self, target_offset):
        raw = torch.tanh(self.net(target_offset))
        center = 0.5 * (self.q_lower + self.q_upper)
        half_range = 0.5 * (self.q_upper - self.q_lower)
        return center + raw * half_range


def make_teacher_args(args):
    return SimpleNamespace(
        max_iters=args.ik_max_iters,
        restarts=1,
        tol=args.ik_tol,
        damping=args.damping,
        step_scale=args.step_scale,
        x_offset=args.x_offset,
        y_offset=args.y_offset,
        z_offset=args.z_offset,
    )


def collect_dataset(args, model, data, ee_body_id, joints, default_ee_pos):
    teacher_args = make_teacher_args(args)
    targets = sample_targets(default_ee_pos, args.samples, args.seed, teacher_args)
    rng = np.random.default_rng(args.seed + 2000)
    xs = []
    ys = []
    errors = []
    successes = 0

    for idx, target in enumerate(targets, start=1):
        qpos, final_err, _, success, _ = solve_ik(model, data, ee_body_id, joints, target, teacher_args, rng)
        if success:
            successes += 1
        target_offset = target - default_ee_pos
        q_targets = np.array([qpos[item["qadr"]] for item in joints], dtype=np.float32)
        xs.append(target_offset.astype(np.float32))
        ys.append(q_targets)
        errors.append(final_err)
        if idx % max(1, args.samples // 10) == 0:
            print(f"teacher {idx:04d}/{args.samples} success_rate={successes / idx:.3f} mean_err={np.mean(errors):.4f}")

    return np.stack(xs), np.stack(ys), np.asarray(errors, dtype=np.float32), successes / len(targets)


def evaluate_fk(model, data, ee_body_id, joints, default_ee_pos, policy, targets, device):
    rows = []
    with torch.no_grad():
        target_offsets = torch.as_tensor(targets - default_ee_pos, dtype=torch.float32, device=device)
        q_preds = policy(target_offsets).cpu().numpy()

    for target, q_pred in zip(targets, q_preds):
        data.qpos[:] = 0.0
        for item, value in zip(joints, q_pred):
            data.qpos[item["qadr"]] = value
        mujoco.mj_forward(model, data)
        ee_pos = data.xpos[ee_body_id].copy()
        err = float(np.linalg.norm(target - ee_pos))
        rows.append((target, ee_pos, err, q_pred))
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Train a distilled A2D reaching policy from the IK teacher.")
    parser.add_argument("--urdf", default="/root/autodl-tmp/A2D_dh/A2D_dh/A2D.urdf")
    parser.add_argument("--converted", default="reports/a2d_mujoco/A2D_mujoco.urdf")
    parser.add_argument("--ee-body", default="Link7_r")
    parser.add_argument("--samples", type=int, default=1500)
    parser.add_argument("--val-targets", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--ik-max-iters", type=int, default=120)
    parser.add_argument("--ik-tol", type=float, default=0.05)
    parser.add_argument("--damping", type=float, default=0.04)
    parser.add_argument("--step-scale", type=float, default=0.65)
    parser.add_argument("--x-offset", type=float, nargs=2, default=(0.10, 0.38))
    parser.add_argument("--y-offset", type=float, nargs=2, default=(-0.02, 0.22))
    parser.add_argument("--z-offset", type=float, nargs=2, default=(-0.16, 0.16))
    parser.add_argument("--success-radius", type=float, default=0.05)
    parser.add_argument("--output", default="logs/a2d_reaching/reaching_policy.pt")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    converted = Path(args.converted).expanduser().resolve()
    convert_package_mesh_paths(Path(args.urdf).expanduser().resolve(), converted)
    model = mujoco.MjModel.from_xml_path(str(converted))
    data = mujoco.MjData(model)
    ee_body_id = named_id(model, mujoco.mjtObj.mjOBJ_BODY, args.ee_body)
    joints = joint_info(model, IK_JOINTS)

    mujoco.mj_forward(model, data)
    default_ee_pos = data.xpos[ee_body_id].copy()
    q_lower = np.array([item["lower"] if item["limited"] else -1.0 for item in joints], dtype=np.float32)
    q_upper = np.array([item["upper"] if item["limited"] else 1.0 for item in joints], dtype=np.float32)

    print(f"default_ee_pos={np.array2string(default_ee_pos, precision=4)}")
    print("collecting IK teacher dataset")
    x_np, y_np, ik_errors, ik_success_rate = collect_dataset(args, model, data, ee_body_id, joints, default_ee_pos)
    print(f"teacher_success_rate={ik_success_rate:.3f} teacher_mean_err={ik_errors.mean():.4f}")

    x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
    y = torch.as_tensor(y_np, dtype=torch.float32, device=device)
    split = int(len(x_np) * 0.9)
    perm = torch.randperm(len(x), device=device)
    train_idx = perm[:split]
    val_idx = perm[split:]

    policy = A2DReachingPolicy(q_lower, q_upper).to(device)
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-5)

    for epoch in range(1, args.epochs + 1):
        order = train_idx[torch.randperm(len(train_idx), device=device)]
        train_loss = 0.0
        batches = 0
        for start in range(0, len(order), args.batch_size):
            idx = order[start : start + args.batch_size]
            pred = policy(x[idx])
            loss = F.mse_loss(pred, y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
            batches += 1

        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            with torch.no_grad():
                val_pred = policy(x[val_idx])
                val_mse = F.mse_loss(val_pred, y[val_idx]).item()
                val_mae = torch.mean(torch.abs(val_pred - y[val_idx])).item()
            print(f"epoch={epoch:03d} train_mse={train_loss / batches:.7f} val_mse={val_mse:.7f} val_q_mae={val_mae:.5f}")

    teacher_args = make_teacher_args(args)
    val_targets = sample_targets(default_ee_pos, args.val_targets, args.seed + 999, teacher_args)
    fk_rows = evaluate_fk(model, data, ee_body_id, joints, default_ee_pos, policy, val_targets, device)
    fk_errors = np.array([row[2] for row in fk_rows], dtype=np.float32)
    fk_success = fk_errors < args.success_radius
    print(f"fk_success={int(fk_success.sum())}/{len(fk_success)} rate={fk_success.mean():.3f}")
    print(
        f"fk_error_mean={fk_errors.mean():.4f} fk_error_median={np.median(fk_errors):.4f} "
        f"fk_error_p90={np.percentile(fk_errors, 90):.4f} fk_error_max={fk_errors.max():.4f}"
    )

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    scripted = torch.jit.script(policy.cpu())
    scripted.save(str(output))
    print(f"saved_policy={output}")


if __name__ == "__main__":
    main()
