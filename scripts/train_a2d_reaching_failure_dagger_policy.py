import argparse
from pathlib import Path

import mujoco
import numpy as np
import torch

from a2d_reaching_ik_teacher import IK_JOINTS, joint_info, named_id, sample_targets
from load_a2d_urdf import convert_package_mesh_paths
from train_a2d_reaching_dagger_policy import train_policy
from train_a2d_reaching_delta_policy import (
    DeltaReachingPolicy,
    clamp_q,
    get_joint_arrays,
    make_obs,
    rollout,
    teacher_delta,
)


def rollout_arrays(joint_arrays):
    qadr, _dadr, lower, upper, center, half = joint_arrays
    return qadr, lower, upper, center, half


def collect_failure_recovery(args, model, data, ee_body_id, joint_arrays, failed_targets, policy, device):
    qadr, dadr, lower, upper, center, half = joint_arrays
    xs = []
    ys = []
    rng = np.random.default_rng(args.seed + 5000)
    with torch.no_grad():
        for repeat in range(args.failure_repeats):
            for idx, target in enumerate(failed_targets, start=1):
                q = rng.normal(0.0, args.q_noise, size=len(qadr))
                q = clamp_q(q, lower, upper)
                for _ in range(args.dagger_rollout_steps):
                    data.qpos[:] = 0.0
                    data.qpos[qadr] = q
                    mujoco.mj_forward(model, data)
                    err_vec = target - data.xpos[ee_body_id].copy()
                    obs_np = make_obs(q, err_vec, center, half)
                    teacher_dq = teacher_delta(model, data, ee_body_id, dadr, err_vec, args.damping, args.max_delta)
                    xs.append(obs_np)
                    ys.append(teacher_dq.astype(np.float32))

                    obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
                    policy_dq = policy(obs).cpu().numpy().squeeze()
                    mix = args.teacher_mix
                    executed_dq = (1.0 - mix) * policy_dq + mix * teacher_dq
                    q = clamp_q(q + args.policy_step_scale * executed_dq, lower, upper)
            print(f"failure_dagger repeat={repeat + 1}/{args.failure_repeats} samples={len(xs)}")
    return xs, ys


def parse_args():
    parser = argparse.ArgumentParser(description="Targeted DAgger around failed A2D reaching targets.")
    parser.add_argument("--urdf", default="/root/autodl-tmp/A2D_dh/A2D_dh/A2D.urdf")
    parser.add_argument("--converted", default="reports/a2d_mujoco/A2D_mujoco.urdf")
    parser.add_argument("--ee-body", default="Link7_r")
    parser.add_argument("--init-policy", default="logs/a2d_reaching/reaching_delta_policy_gpu.pt")
    parser.add_argument("--output", default="logs/a2d_reaching/reaching_failure_dagger_policy.pt")
    parser.add_argument("--val-targets", type=int, default=512)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--damping", type=float, default=0.04)
    parser.add_argument("--max-delta", type=float, default=0.08)
    parser.add_argument("--policy-step-scale", type=float, default=0.8)
    parser.add_argument("--rollout-steps", type=int, default=70)
    parser.add_argument("--success-radius", type=float, default=0.05)
    parser.add_argument("--dagger-rollout-steps", type=int, default=70)
    parser.add_argument("--failure-repeats", type=int, default=12)
    parser.add_argument("--q-noise", type=float, default=0.04)
    parser.add_argument("--teacher-mix", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--log-interval", type=int, default=40)
    parser.add_argument("--x-offset", type=float, nargs=2, default=(0.10, 0.38))
    parser.add_argument("--y-offset", type=float, nargs=2, default=(-0.02, 0.22))
    parser.add_argument("--z-offset", type=float, nargs=2, default=(-0.16, 0.16))
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
    val_targets = sample_targets(default_ee_pos, args.val_targets, args.seed + 999, target_args)

    policy = torch.jit.load(str(Path(args.init_policy).expanduser().resolve()), map_location=device)
    errors, successes = rollout(policy, model, data, ee_body_id, rollout_arrays(joint_arrays), val_targets, args, device)
    failed_targets = val_targets[~successes]
    print(f"device={device}")
    print(f"initial_success={int(successes.sum())}/{len(successes)} rate={successes.mean():.3f}")
    print(f"initial_error_mean={errors.mean():.4f} median={np.median(errors):.4f} p90={np.percentile(errors, 90):.4f} max={errors.max():.4f}")
    print(f"failed_targets={len(failed_targets)}")
    if len(failed_targets) == 0:
        print("No failed targets; saving unchanged policy.")
        torch.jit.save(policy, str(Path(args.output).expanduser().resolve()))
        return

    xs, ys = collect_failure_recovery(args, model, data, ee_body_id, joint_arrays, failed_targets, policy, device)
    student = DeltaReachingPolicy(action_scale=args.max_delta).to(device)
    student.load_state_dict(policy.state_dict())
    train_policy(student, xs, ys, args, device, args.epochs, "failure_dagger")

    errors, successes = rollout(student, model, data, ee_body_id, rollout_arrays(joint_arrays), val_targets, args, device)
    print(f"after_failure_dagger_success={int(successes.sum())}/{len(successes)} rate={successes.mean():.3f}")
    print(f"after_failure_dagger_error_mean={errors.mean():.4f} median={np.median(errors):.4f} p90={np.percentile(errors, 90):.4f} max={errors.max():.4f}")

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.script(student.cpu()).save(str(output))
    print(f"saved_policy={output}")


if __name__ == "__main__":
    main()
