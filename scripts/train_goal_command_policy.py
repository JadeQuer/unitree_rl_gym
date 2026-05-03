import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class GoalCommandPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, 3),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        raw = self.net(obs)
        limits = torch.tensor([0.5, 0.15, 0.5], dtype=raw.dtype, device=raw.device)
        return torch.tanh(raw) * limits


def teacher_command(goal_dist, goal_dir_y, heading_error, success_radius):
    vx = torch.clamp(goal_dist * 0.45, 0.0, 0.5)
    vx = torch.where(goal_dist < success_radius, torch.zeros_like(vx), vx)
    vy = torch.clamp(goal_dir_y * 0.15, -0.15, 0.15)
    yaw_rate = torch.clamp(heading_error * 0.8, -0.5, 0.5)
    return torch.stack([vx, vy, yaw_rate], dim=-1)


def make_dataset(num_samples, success_radius, device):
    goal_x = torch.empty(num_samples, device=device).uniform_(0.0, 4.5)
    goal_y = torch.empty(num_samples, device=device).uniform_(-1.5, 1.5)
    heading = torch.empty(num_samples, device=device).uniform_(-3.14159, 3.14159)

    goal_dist = torch.sqrt(goal_x.square() + goal_y.square()).clamp(min=1e-6)
    goal_heading = torch.atan2(goal_y, goal_x)
    heading_error = (goal_heading - heading + torch.pi) % (2 * torch.pi) - torch.pi

    cos_h = torch.cos(heading)
    sin_h = torch.sin(heading)
    body_x = cos_h * goal_x + sin_h * goal_y
    body_y = -sin_h * goal_x + cos_h * goal_y
    body_dist = torch.sqrt(body_x.square() + body_y.square()).clamp(min=1e-6)
    body_dir_x = body_x / body_dist
    body_dir_y = body_y / body_dist

    obs = torch.stack(
        [
            goal_dist / 4.5,
            body_dir_x,
            body_dir_y,
            heading_error / torch.pi,
        ],
        dim=-1,
    )
    target = teacher_command(goal_dist, body_dir_y, heading_error, success_radius)
    return obs, target


def parse_args():
    parser = argparse.ArgumentParser(description="Distill the hand-written goal controller into a TorchScript policy.")
    parser.add_argument("--output", default="logs/g1_goal/exported/policies/goal_command_policy.pt")
    parser.add_argument("--samples", type=int, default=200000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--success-radius", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    obs, target = make_dataset(args.samples, args.success_radius, device)
    split = int(args.samples * 0.9)
    train_obs, val_obs = obs[:split], obs[split:]
    train_target, val_target = target[:split], target[split:]

    model = GoalCommandPolicy().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(split, device=device)
        train_loss = 0.0
        num_batches = 0
        for start in range(0, split, args.batch_size):
            idx = perm[start : start + args.batch_size]
            pred = model(train_obs[idx])
            loss = F.mse_loss(pred, train_target[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            num_batches += 1

        with torch.no_grad():
            val_pred = model(val_obs)
            val_mse = F.mse_loss(val_pred, val_target).item()
            val_mae = torch.mean(torch.abs(val_pred - val_target), dim=0)

        print(
            f"epoch={epoch:03d} train_mse={train_loss / num_batches:.8f} "
            f"val_mse={val_mse:.8f} "
            f"mae_vx={val_mae[0].item():.5f} mae_vy={val_mae[1].item():.5f} mae_yaw={val_mae[2].item():.5f}"
        )

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scripted = torch.jit.script(model.cpu())
    scripted.save(str(output_path))
    print(f"Saved goal command policy to: {output_path}")


if __name__ == "__main__":
    main()
