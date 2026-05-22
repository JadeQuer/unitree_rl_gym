import argparse
from pathlib import Path

import mujoco
import numpy as np
import torch
import torch.nn.functional as F

from a2d_reaching_ik_teacher import IK_JOINTS, joint_info, named_id, sample_targets
from load_a2d_urdf import convert_package_mesh_paths
from train_a2d_reaching_delta_policy import (
    DeltaReachingPolicy,
    clamp_q,
    get_joint_arrays,
    make_obs,
    teacher_delta,
    rollout,
)


def collect_teacher_dataset(args, model, data, ee_body_id, joint_arrays, default_ee_pos, targets):
    qadr, dadr, lower, upper, center, half = joint_arrays
    xs = []
    ys = []
    for idx, target in enumerate(targets, start=1):
        q = np.zeros(len(qadr), dtype=np.float64)
        for _ in range(args.teacher_steps):
            data.qpos[:] = 0.0
            data.qpos[qadr] = q
            mujoco.mj_forward(model, data)
            err_vec = target - data.xpos[ee_body_id].copy()
            dq = teacher_delta(model, data, ee_body_id, dadr, err_vec, args.damping, args.max_delta)
            xs.append(make_obs(q, err_vec, center, half))
            ys.append(dq.astype(np.float32))
            q = clamp_q(q + args.step_scale * dq, lower, upper)
        if idx % max(1, len(targets) // 5) == 0:
            print(f"base_teacher {idx}/{len(targets)}")
    return xs, ys


def rollout_arrays(joint_arrays):
    qadr, _dadr, lower, upper, center, half = joint_arrays
    return qadr, lower, upper, center, half


def collect_dagger_dataset(args, model, data, ee_body_id, joint_arrays, default_ee_pos, targets, policy, device):
    qadr, dadr, lower, upper, center, half = joint_arrays
    xs = []
    ys = []
    with torch.no_grad():
        for idx, target in enumerate(targets, start=1):
            q = np.zeros(len(qadr), dtype=np.float64)
            for _ in range(args.dagger_rollout_steps):
                data.qpos[:] = 0.0
                data.qpos[qadr] = q
                mujoco.mj_forward(model, data)
                err_vec = target - data.xpos[ee_body_id].copy()

                teacher_dq = teacher_delta(model, data, ee_body_id, dadr, err_vec, args.damping, args.max_delta)
                xs.append(make_obs(q, err_vec, center, half))
                ys.append(teacher_dq.astype(np.float32))

                obs = torch.as_tensor(xs[-1], dtype=torch.float32, device=device).unsqueeze(0)
                policy_dq = policy(obs).cpu().numpy().squeeze()
                q = clamp_q(q + args.policy_step_scale * policy_dq, lower, upper)
            if idx % max(1, len(targets) // 5) == 0:
                print(f"dagger_rollout {idx}/{len(targets)}")
    return xs, ys


def train_policy(policy, x_np, y_np, args, device, epochs, label):
    x = torch.as_tensor(np.stack(x_np), dtype=torch.float32, device=device)
    y = torch.as_tensor(np.stack(y_np), dtype=torch.float32, device=device)
    split = int(0.9 * len(x))
    perm = torch.randperm(len(x), device=device)
    train_idx, val_idx = perm[:split], perm[split:]
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-5)

    for epoch in range(1, epochs + 1):
        order = train_idx[torch.randperm(len(train_idx), device=device)]
        total = 0.0
        batches = 0
        for start in range(0, len(order), args.batch_size):
            idx = order[start : start + args.batch_size]
            pred = policy(x[idx])
            loss = F.mse_loss(pred, y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            batches += 1
        if epoch == 1 or epoch % args.log_interval == 0 or epoch == epochs:
            with torch.no_grad():
                val_pred = policy(x[val_idx])
                val_mse = F.mse_loss(val_pred, y[val_idx]).item()
                val_mae = torch.mean(torch.abs(val_pred - y[val_idx])).item()
            print(f"{label} epoch={epoch:03d} train_mse={total / batches:.8f} val_mse={val_mse:.8f} val_dq_mae={val_mae:.5f}")


def parse_args():
    parser = argparse.ArgumentParser(description="DAgger training for A2D closed-loop reaching policy.")
    parser.add_argument("--urdf", default="/root/autodl-tmp/A2D_dh/A2D_dh/A2D.urdf")
    parser.add_argument("--converted", default="reports/a2d_mujoco/A2D_mujoco.urdf")
    parser.add_argument("--ee-body", default="Link7_r")
    parser.add_argument("--base-targets", type=int, default=4000)
    parser.add_argument("--dagger-targets", type=int, default=2500)
    parser.add_argument("--val-targets", type=int, default=256)
    parser.add_argument("--teacher-steps", type=int, default=18)
    parser.add_argument("--dagger-rollout-steps", type=int, default=45)
    parser.add_argument("--pretrain-epochs", type=int, default=180)
    parser.add_argument("--dagger-epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-interval", type=int, default=40)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--damping", type=float, default=0.04)
    parser.add_argument("--max-delta", type=float, default=0.08)
    parser.add_argument("--step-scale", type=float, default=0.8)
    parser.add_argument("--policy-step-scale", type=float, default=0.8)
    parser.add_argument("--rollout-steps", type=int, default=70)
    parser.add_argument("--success-radius", type=float, default=0.05)
    parser.add_argument("--x-offset", type=float, nargs=2, default=(0.10, 0.38))
    parser.add_argument("--y-offset", type=float, nargs=2, default=(-0.02, 0.22))
    parser.add_argument("--z-offset", type=float, nargs=2, default=(-0.16, 0.16))
    parser.add_argument("--output", default="logs/a2d_reaching/reaching_dagger_policy.pt")
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
    joint_arrays = get_joint_arrays(model, joints)
    mujoco.mj_forward(model, data)
    default_ee_pos = data.xpos[ee_body_id].copy()

    target_args = argparse.Namespace(x_offset=args.x_offset, y_offset=args.y_offset, z_offset=args.z_offset)
    base_targets = sample_targets(default_ee_pos, args.base_targets, args.seed, target_args)
    dagger_targets = sample_targets(default_ee_pos, args.dagger_targets, args.seed + 100, target_args)
    val_targets = sample_targets(default_ee_pos, args.val_targets, args.seed + 999, target_args)

    print(f"device={device}")
    print(f"default_ee_pos={np.array2string(default_ee_pos, precision=4)}")
    xs, ys = collect_teacher_dataset(args, model, data, ee_body_id, joint_arrays, default_ee_pos, base_targets)

    policy = DeltaReachingPolicy(action_scale=args.max_delta).to(device)
    train_policy(policy, xs, ys, args, device, args.pretrain_epochs, "pretrain")
    errors, successes = rollout(policy, model, data, ee_body_id, rollout_arrays(joint_arrays), val_targets, args, device)
    print(f"before_dagger_success={int(successes.sum())}/{len(successes)} rate={successes.mean():.3f}")
    print(f"before_dagger_error_mean={errors.mean():.4f} median={np.median(errors):.4f} p90={np.percentile(errors, 90):.4f} max={errors.max():.4f}")

    dx, dy = collect_dagger_dataset(args, model, data, ee_body_id, joint_arrays, default_ee_pos, dagger_targets, policy, device)
    xs.extend(dx)
    ys.extend(dy)
    print(f"aggregated_samples={len(xs)}")
    train_policy(policy, xs, ys, args, device, args.dagger_epochs, "dagger")

    errors, successes = rollout(policy, model, data, ee_body_id, rollout_arrays(joint_arrays), val_targets, args, device)
    print(f"after_dagger_success={int(successes.sum())}/{len(successes)} rate={successes.mean():.3f}")
    print(f"after_dagger_error_mean={errors.mean():.4f} median={np.median(errors):.4f} p90={np.percentile(errors, 90):.4f} max={errors.max():.4f}")

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.script(policy.cpu()).save(str(output))
    print(f"saved_policy={output}")


if __name__ == "__main__":
    main()
