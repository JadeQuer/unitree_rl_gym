import argparse
from pathlib import Path

import mujoco
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a2d_reaching_ik_teacher import IK_JOINTS, joint_info, named_id, sample_targets
from load_a2d_urdf import convert_package_mesh_paths
from train_a2d_reaching_delta_policy import clamp_q, get_joint_arrays


BASE_NAMES = ["base_x", "base_y", "base_yaw"]


class MobileBasePolicy(nn.Module):
    def __init__(self, action_scale):
        super().__init__()
        self.register_buffer("action_scale", torch.as_tensor(action_scale, dtype=torch.float32))
        self.net = nn.Sequential(
            nn.Linear(15, 160),
            nn.ELU(),
            nn.Linear(160, 160),
            nn.ELU(),
            nn.Linear(160, 12),
        )

    def forward(self, obs):
        return torch.tanh(self.net(obs)) * self.action_scale


def yaw_matrix(yaw):
    c = np.cos(yaw)
    s = np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def apply_state(model, data, ee_body_id, qadr, arm_q, base_state):
    data.qpos[:] = 0.0
    data.qpos[qadr] = arm_q
    mujoco.mj_forward(model, data)
    local_ee = data.xpos[ee_body_id].copy()
    rot = yaw_matrix(base_state[2])
    world_ee = rot @ local_ee + np.array([base_state[0], base_state[1], 0.0])
    return local_ee, world_ee


def make_obs(base_state, arm_q, err_world, center, half):
    q_norm = (arm_q - center) / np.maximum(half, 1e-6)
    base_norm = np.array([base_state[0] / 0.25, base_state[1] / 0.25, base_state[2] / 0.45], dtype=np.float64)
    return np.concatenate([base_norm, q_norm, err_world], dtype=np.float32)


def mobile_teacher_delta(model, data, ee_body_id, qadr, dadr, arm_q, base_state, target, lower, upper, args):
    local_ee, world_ee = apply_state(model, data, ee_body_id, qadr, arm_q, base_state)
    err = target - world_ee
    rot = yaw_matrix(base_state[2])

    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacBody(model, data, jacp, jacr, ee_body_id)
    j_arm = rot @ jacp[:, dadr]

    yaw_axis = np.array([0.0, 0.0, 1.0])
    yaw_col = np.cross(yaw_axis, rot @ local_ee)
    j_base = np.column_stack(
        [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            yaw_col,
        ]
    )
    j = np.concatenate([j_base, j_arm], axis=1)
    lhs = j @ j.T + args.damping**2 * np.eye(3)
    dq_all = j.T @ np.linalg.solve(lhs, err)

    base_delta = np.clip(dq_all[:3], [-args.max_base_xy, -args.max_base_xy, -args.max_base_yaw], [args.max_base_xy, args.max_base_xy, args.max_base_yaw])
    arm_delta = np.clip(dq_all[3:], -args.max_arm_delta, args.max_arm_delta)
    # Keep the virtual base conservative; prefer arm/waist unless base helps.
    base_delta[:2] *= args.base_xy_weight
    base_delta[2] *= args.base_yaw_weight
    return np.concatenate([base_delta, arm_delta]).astype(np.float32), err


def clamp_base(base_state, args):
    base_state[0] = np.clip(base_state[0], -args.base_xy_limit, args.base_xy_limit)
    base_state[1] = np.clip(base_state[1], -args.base_xy_limit, args.base_xy_limit)
    base_state[2] = np.clip(base_state[2], -args.base_yaw_limit, args.base_yaw_limit)
    return base_state


def collect_dataset(args, model, data, ee_body_id, qadr, dadr, lower, upper, center, half, default_ee_pos):
    target_args = argparse.Namespace(x_offset=args.x_offset, y_offset=args.y_offset, z_offset=args.z_offset)
    targets = sample_targets(default_ee_pos, args.targets, args.seed, target_args)
    xs = []
    ys = []
    for idx, target in enumerate(targets, start=1):
        base_state = np.zeros(3, dtype=np.float64)
        arm_q = np.zeros(len(qadr), dtype=np.float64)
        for _ in range(args.steps_per_target):
            action, err = mobile_teacher_delta(model, data, ee_body_id, qadr, dadr, arm_q, base_state, target, lower, upper, args)
            xs.append(make_obs(base_state, arm_q, err, center, half))
            ys.append(action)
            base_state = clamp_base(base_state + args.step_scale * action[:3], args)
            arm_q = clamp_q(arm_q + args.step_scale * action[3:], lower, upper)
        if idx % max(1, args.targets // 10) == 0:
            print(f"mobile_dataset {idx}/{args.targets}")
    return np.stack(xs), np.stack(ys), targets


def rollout(policy, model, data, ee_body_id, qadr, lower, upper, center, half, default_ee_pos, targets, args, device):
    errors = []
    base_norms = []
    successes = []
    with torch.no_grad():
        for target in targets:
            base_state = np.zeros(3, dtype=np.float64)
            arm_q = np.zeros(len(qadr), dtype=np.float64)
            best = float("inf")
            for _ in range(args.rollout_steps):
                _, world_ee = apply_state(model, data, ee_body_id, qadr, arm_q, base_state)
                err = target - world_ee
                best = min(best, float(np.linalg.norm(err)))
                obs = torch.as_tensor(make_obs(base_state, arm_q, err, center, half), dtype=torch.float32, device=device).unsqueeze(0)
                action = policy(obs).cpu().numpy().squeeze()
                base_state = clamp_base(base_state + args.policy_step_scale * action[:3], args)
                arm_q = clamp_q(arm_q + args.policy_step_scale * action[3:], lower, upper)
            _, world_ee = apply_state(model, data, ee_body_id, qadr, arm_q, base_state)
            final = float(np.linalg.norm(target - world_ee))
            err = min(best, final)
            errors.append(err)
            successes.append(err < args.success_radius)
            base_norms.append(float(np.linalg.norm(base_state[:2])))
    return np.array(errors, dtype=np.float32), np.array(successes, dtype=np.bool_), np.array(base_norms, dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="Train A2D reaching policy with virtual mobile-base DOFs.")
    parser.add_argument("--urdf", default="/root/autodl-tmp/A2D_dh/A2D_dh/A2D.urdf")
    parser.add_argument("--converted", default="reports/a2d_mujoco/A2D_mujoco.urdf")
    parser.add_argument("--ee-body", default="Link7_r")
    parser.add_argument("--targets", type=int, default=5000)
    parser.add_argument("--steps-per-target", type=int, default=18)
    parser.add_argument("--val-targets", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=51)
    parser.add_argument("--damping", type=float, default=0.04)
    parser.add_argument("--max-arm-delta", type=float, default=0.08)
    parser.add_argument("--max-base-xy", type=float, default=0.025)
    parser.add_argument("--max-base-yaw", type=float, default=0.035)
    parser.add_argument("--base-xy-weight", type=float, default=0.35)
    parser.add_argument("--base-yaw-weight", type=float, default=0.4)
    parser.add_argument("--base-xy-limit", type=float, default=0.25)
    parser.add_argument("--base-yaw-limit", type=float, default=0.45)
    parser.add_argument("--step-scale", type=float, default=0.8)
    parser.add_argument("--policy-step-scale", type=float, default=0.8)
    parser.add_argument("--rollout-steps", type=int, default=70)
    parser.add_argument("--success-radius", type=float, default=0.05)
    parser.add_argument("--x-offset", type=float, nargs=2, default=(0.10, 0.45))
    parser.add_argument("--y-offset", type=float, nargs=2, default=(-0.08, 0.28))
    parser.add_argument("--z-offset", type=float, nargs=2, default=(-0.18, 0.18))
    parser.add_argument("--output", default="logs/a2d_reaching/reaching_mobile_base_policy.pt")
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
    qadr, dadr, lower, upper, center, half = get_joint_arrays(model, joints)
    mujoco.mj_forward(model, data)
    default_ee_pos = data.xpos[ee_body_id].copy()

    print(f"device={device}")
    print(f"default_ee_pos={np.array2string(default_ee_pos, precision=4)}")
    x_np, y_np, _ = collect_dataset(args, model, data, ee_body_id, qadr, dadr, lower, upper, center, half, default_ee_pos)
    x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
    y = torch.as_tensor(y_np, dtype=torch.float32, device=device)
    split = int(0.9 * len(x))
    perm = torch.randperm(len(x), device=device)
    train_idx, val_idx = perm[:split], perm[split:]

    action_scale = np.array([args.max_base_xy, args.max_base_xy, args.max_base_yaw] + [args.max_arm_delta] * len(qadr), dtype=np.float32)
    policy = MobileBasePolicy(action_scale).to(device)
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-5)
    for epoch in range(1, args.epochs + 1):
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
        if epoch == 1 or epoch % 40 == 0 or epoch == args.epochs:
            with torch.no_grad():
                pred = policy(x[val_idx])
                val_mse = F.mse_loss(pred, y[val_idx]).item()
                val_mae = torch.mean(torch.abs(pred - y[val_idx])).item()
            print(f"epoch={epoch:03d} train_mse={total / batches:.8f} val_mse={val_mse:.8f} val_action_mae={val_mae:.5f}")

    target_args = argparse.Namespace(x_offset=args.x_offset, y_offset=args.y_offset, z_offset=args.z_offset)
    val_targets = sample_targets(default_ee_pos, args.val_targets, args.seed + 999, target_args)
    errors, successes, base_norms = rollout(policy, model, data, ee_body_id, qadr, lower, upper, center, half, default_ee_pos, val_targets, args, device)
    print(f"mobile_success={int(successes.sum())}/{len(successes)} rate={successes.mean():.3f}")
    print(f"mobile_error_mean={errors.mean():.4f} median={np.median(errors):.4f} p90={np.percentile(errors, 90):.4f} max={errors.max():.4f}")
    print(f"base_xy_mean={base_norms.mean():.4f} base_xy_p90={np.percentile(base_norms, 90):.4f} base_xy_max={base_norms.max():.4f}")

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.script(policy.cpu()).save(str(output))
    print(f"saved_policy={output}")


if __name__ == "__main__":
    main()
