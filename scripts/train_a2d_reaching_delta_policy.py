import argparse
from pathlib import Path

import mujoco
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a2d_reaching_ik_teacher import IK_JOINTS, joint_info, named_id, sample_targets
from load_a2d_urdf import convert_package_mesh_paths


class DeltaReachingPolicy(nn.Module):
    def __init__(self, action_scale=0.08):
        super().__init__()
        self.action_scale = action_scale
        self.net = nn.Sequential(
            nn.Linear(12, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, 9),
        )

    def forward(self, obs):
        return torch.tanh(self.net(obs)) * self.action_scale


def clamp_q(q, lower, upper):
    return np.minimum(np.maximum(q, lower), upper)


def get_joint_arrays(model, joints):
    qadr = np.array([item["qadr"] for item in joints], dtype=np.int32)
    dadr = np.array([item["dadr"] for item in joints], dtype=np.int32)
    lower = np.array([item["lower"] if item["limited"] else -1.0 for item in joints], dtype=np.float64)
    upper = np.array([item["upper"] if item["limited"] else 1.0 for item in joints], dtype=np.float64)
    center = 0.5 * (lower + upper)
    half = 0.5 * (upper - lower)
    return qadr, dadr, lower, upper, center, half


def teacher_delta(model, data, ee_body_id, dadr, err_vec, damping, max_delta):
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacBody(model, data, jacp, jacr, ee_body_id)
    j_task = jacp[:, dadr]
    lhs = j_task @ j_task.T + damping**2 * np.eye(3)
    dq = j_task.T @ np.linalg.solve(lhs, err_vec)
    return np.clip(dq, -max_delta, max_delta)


def make_obs(q, err_vec, center, half):
    q_norm = (q - center) / np.maximum(half, 1e-6)
    return np.concatenate([q_norm, err_vec], dtype=np.float32)


def collect_dataset(args, model, data, ee_body_id, joints, default_ee_pos):
    qadr, dadr, lower, upper, center, half = get_joint_arrays(model, joints)
    target_args = argparse.Namespace(x_offset=args.x_offset, y_offset=args.y_offset, z_offset=args.z_offset)
    targets = sample_targets(default_ee_pos, args.targets, args.seed, target_args)
    rng = np.random.default_rng(args.seed + 3000)
    obs_rows = []
    act_rows = []

    for idx, target in enumerate(targets, start=1):
        q = np.zeros(len(joints), dtype=np.float64)
        # Small random starts teach recovery, while staying near the nominal pose.
        if args.randomize_q:
            q = clamp_q(rng.normal(0.0, 0.08, size=len(joints)), lower, upper)

        for _ in range(args.steps_per_target):
            data.qpos[:] = 0.0
            data.qpos[qadr] = q
            mujoco.mj_forward(model, data)
            ee_pos = data.xpos[ee_body_id].copy()
            err_vec = target - ee_pos
            dq = teacher_delta(model, data, ee_body_id, dadr, err_vec, args.damping, args.max_delta)
            obs_rows.append(make_obs(q, err_vec, center, half))
            act_rows.append(dq.astype(np.float32))
            q = clamp_q(q + args.step_scale * dq, lower, upper)

        if idx % max(1, args.targets // 10) == 0:
            print(f"dataset {idx:04d}/{args.targets}")

    return np.stack(obs_rows), np.stack(act_rows), (qadr, lower, upper, center, half), targets


def rollout(policy, model, data, ee_body_id, joint_arrays, targets, args, device):
    qadr, lower, upper, center, half = joint_arrays
    errors = []
    successes = []
    with torch.no_grad():
        for target in targets:
            q = np.zeros(len(qadr), dtype=np.float64)
            best_err = float("inf")
            for _ in range(args.rollout_steps):
                data.qpos[:] = 0.0
                data.qpos[qadr] = q
                mujoco.mj_forward(model, data)
                ee_pos = data.xpos[ee_body_id].copy()
                err_vec = target - ee_pos
                best_err = min(best_err, float(np.linalg.norm(err_vec)))
                obs = torch.as_tensor(make_obs(q, err_vec, center, half), dtype=torch.float32, device=device).unsqueeze(0)
                dq = policy(obs).cpu().numpy().squeeze()
                q = clamp_q(q + args.policy_step_scale * dq, lower, upper)
            data.qpos[:] = 0.0
            data.qpos[qadr] = q
            mujoco.mj_forward(model, data)
            final_err = float(np.linalg.norm(target - data.xpos[ee_body_id].copy()))
            err = min(best_err, final_err)
            errors.append(err)
            successes.append(err < args.success_radius)
    return np.array(errors, dtype=np.float32), np.array(successes, dtype=np.bool_)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a closed-loop A2D reaching delta policy from Jacobian IK teacher.")
    parser.add_argument("--urdf", default="/root/autodl-tmp/A2D_dh/A2D_dh/A2D.urdf")
    parser.add_argument("--converted", default="reports/a2d_mujoco/A2D_mujoco.urdf")
    parser.add_argument("--ee-body", default="Link7_r")
    parser.add_argument("--targets", type=int, default=1500)
    parser.add_argument("--steps-per-target", type=int, default=12)
    parser.add_argument("--val-targets", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--damping", type=float, default=0.04)
    parser.add_argument("--max-delta", type=float, default=0.08)
    parser.add_argument("--step-scale", type=float, default=0.8)
    parser.add_argument("--policy-step-scale", type=float, default=0.8)
    parser.add_argument("--rollout-steps", type=int, default=35)
    parser.add_argument("--success-radius", type=float, default=0.05)
    parser.add_argument("--randomize-q", action="store_true")
    parser.add_argument("--x-offset", type=float, nargs=2, default=(0.10, 0.38))
    parser.add_argument("--y-offset", type=float, nargs=2, default=(-0.02, 0.22))
    parser.add_argument("--z-offset", type=float, nargs=2, default=(-0.16, 0.16))
    parser.add_argument("--output", default="logs/a2d_reaching/reaching_delta_policy.pt")
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

    print(f"default_ee_pos={np.array2string(default_ee_pos, precision=4)}")
    x_np, y_np, joint_arrays, _ = collect_dataset(args, model, data, ee_body_id, joints, default_ee_pos)
    x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
    y = torch.as_tensor(y_np, dtype=torch.float32, device=device)
    split = int(0.9 * len(x))
    perm = torch.randperm(len(x), device=device)
    train_idx, val_idx = perm[:split], perm[split:]

    policy = DeltaReachingPolicy(action_scale=args.max_delta).to(device)
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-5)
    for epoch in range(1, args.epochs + 1):
        order = train_idx[torch.randperm(len(train_idx), device=device)]
        loss_sum = 0.0
        batches = 0
        for start in range(0, len(order), args.batch_size):
            idx = order[start : start + args.batch_size]
            pred = policy(x[idx])
            loss = F.mse_loss(pred, y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item()
            batches += 1
        if epoch == 1 or epoch % 20 == 0 or epoch == args.epochs:
            with torch.no_grad():
                val_pred = policy(x[val_idx])
                val_mse = F.mse_loss(val_pred, y[val_idx]).item()
                val_mae = torch.mean(torch.abs(val_pred - y[val_idx])).item()
            print(f"epoch={epoch:03d} train_mse={loss_sum / batches:.8f} val_mse={val_mse:.8f} val_dq_mae={val_mae:.5f}")

    val_args = argparse.Namespace(x_offset=args.x_offset, y_offset=args.y_offset, z_offset=args.z_offset)
    val_targets = sample_targets(default_ee_pos, args.val_targets, args.seed + 999, val_args)
    errors, successes = rollout(policy, model, data, ee_body_id, joint_arrays, val_targets, args, device)
    print(f"rollout_success={int(successes.sum())}/{len(successes)} rate={successes.mean():.3f}")
    print(
        f"rollout_error_mean={errors.mean():.4f} median={np.median(errors):.4f} "
        f"p90={np.percentile(errors, 90):.4f} max={errors.max():.4f}"
    )

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.script(policy.cpu()).save(str(output))
    print(f"saved_policy={output}")


if __name__ == "__main__":
    main()
