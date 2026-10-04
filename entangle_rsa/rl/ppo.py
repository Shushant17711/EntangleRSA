"""PPO (CleanRL-style) for every policy family, with encoder gradient-variance logging.

Identical algorithm, critic and head across families; only the encoder differs. Truncated episodes are
bootstrapped with V(final_obs) instead of being treated as terminal.
"""

from __future__ import annotations

import csv
import json
import time
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from entangle_rsa.config import RunConfig
from entangle_rsa.envs.wrappers import get_task, scale_obs
from entangle_rsa.policies.agent import Agent

METRIC_FIELDS = [
    "update", "global_step", "ep_return_mean", "ep_len_mean", "n_episodes", "pg_loss", "v_loss",
    "entropy", "approx_kl", "clipfrac", "grad_var", "grad_norm", "sps",
]


def make_vec_env(task_name: str, n_envs: int, seed: int) -> gym.vector.VectorEnv:
    task = get_task(task_name)
    envs = gym.vector.SyncVectorEnv(
        [lambda: gym.make(task.gym_id) for _ in range(n_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    envs = gym.wrappers.vector.RecordEpisodeStatistics(envs)
    envs.action_space.seed(seed)
    return envs


def build_optimizer(agent: Agent, cfg: RunConfig) -> torch.optim.Optimizer:
    lr = {"quantum": cfg.ppo.lr_quantum, "actor": cfg.ppo.lr_actor, "critic": cfg.ppo.lr_critic}
    groups = [{"params": ps, "lr": lr[name], "base_lr": lr[name], "name": name} for name, ps in agent.param_groups().items()]
    return torch.optim.Adam(groups, eps=1e-5)


def env_action(task, action: np.ndarray) -> np.ndarray:
    if task.discrete:
        return action
    high = 2.0  # Pendulum torque bound; Gaussian samples are clipped only when sent to the env
    return np.clip(action, -high, high)


def evaluate(agent: Agent, cfg: RunConfig, n_episodes: int = 20, seed: int = 10_000) -> dict:
    task = get_task(cfg.task)
    env = gym.make(task.gym_id)
    returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, total = False, 0.0
        while not done:
            with torch.no_grad():
                d = agent.dist(torch.from_numpy(scale_obs(task, obs)).unsqueeze(0))
                a = d.probs.argmax(-1) if task.discrete else d.mean
            a = a.squeeze(0).numpy()
            obs, r, term, trunc, _ = env.step(env_action(task, a))
            total += float(r)
            done = term or trunc
        returns.append(total)
    return {"eval_return_mean": float(np.mean(returns)), "eval_return_std": float(np.std(returns)), "returns": returns}


def train(cfg: RunConfig, root: Path | str = "runs", log_every: int = 1, verbose: bool = False) -> Path:
    out = cfg.run_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(cfg.to_json())

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    task = get_task(cfg.task)
    p = cfg.ppo
    agent = Agent(cfg)
    opt = build_optimizer(agent, cfg)
    envs = make_vec_env(cfg.task, p.n_envs, cfg.seed)

    act_shape = () if task.discrete else (task.n_actions,)
    obs_buf = torch.zeros(p.n_steps, p.n_envs, task.obs_dim)
    act_buf = torch.zeros((p.n_steps, p.n_envs) + act_shape)
    logp_buf = torch.zeros(p.n_steps, p.n_envs)
    rew_buf = torch.zeros(p.n_steps, p.n_envs)
    done_buf = torch.zeros(p.n_steps, p.n_envs)
    val_buf = torch.zeros(p.n_steps, p.n_envs)

    batch = p.n_envs * p.n_steps
    mb_size = batch // p.n_minibatches
    n_updates = max(1, p.total_steps // batch)
    enc_params = agent.encoder_params()

    raw_obs, _ = envs.reset(seed=cfg.seed)
    next_obs = torch.from_numpy(scale_obs(task, raw_obs))
    next_done = torch.zeros(p.n_envs)
    recent = deque(maxlen=20)
    recent_len = deque(maxlen=20)
    n_eps, global_step, t0 = 0, 0, time.time()

    with open(out / "metrics.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        for update in range(1, n_updates + 1):
            if p.anneal_lr:
                frac = 1.0 - (update - 1) / n_updates
                for g in opt.param_groups:
                    g["lr"] = frac * g["base_lr"]

            for t in range(p.n_steps):
                global_step += p.n_envs
                obs_buf[t], done_buf[t] = next_obs, next_done
                with torch.no_grad():
                    d = agent.dist(next_obs)
                    a = d.sample()
                    logp = d.log_prob(a) if task.discrete else d.log_prob(a).sum(-1)
                    val_buf[t] = agent.value(next_obs)
                act_buf[t], logp_buf[t] = a, logp
                raw_obs, r, term, trunc, info = envs.step(env_action(task, a.numpy()))
                r = torch.as_tensor(r, dtype=torch.float32)
                if "_final_obs" in info:
                    # bootstrap truncated (not terminated) episodes with the critic's value of the final state
                    idx = np.flatnonzero(info["_final_obs"] & trunc & ~term)
                    if len(idx):
                        fin = torch.from_numpy(scale_obs(task, np.stack(info["final_obs"][idx])))
                        with torch.no_grad():
                            r[idx] += p.gamma * agent.value(fin)
                if "_episode" in info:
                    for i in np.flatnonzero(info["_episode"]):
                        recent.append(float(info["episode"]["r"][i]))
                        recent_len.append(float(info["episode"]["l"][i]))
                        n_eps += 1
                rew_buf[t] = r
                next_obs = torch.from_numpy(scale_obs(task, raw_obs))
                next_done = torch.as_tensor(np.logical_or(term, trunc), dtype=torch.float32)

            with torch.no_grad():
                next_val = agent.value(next_obs)
                adv = torch.zeros_like(rew_buf)
                last = torch.zeros(p.n_envs)
                for t in reversed(range(p.n_steps)):
                    nonterm = 1.0 - (next_done if t == p.n_steps - 1 else done_buf[t + 1])
                    nv = next_val if t == p.n_steps - 1 else val_buf[t + 1]
                    delta = rew_buf[t] + p.gamma * nv * nonterm - val_buf[t]
                    last = delta + p.gamma * p.gae_lambda * nonterm * last
                    adv[t] = last
                ret = adv + val_buf

            b_obs = obs_buf.reshape(batch, task.obs_dim)
            b_act = act_buf.reshape((batch,) + act_shape)
            b_logp, b_adv, b_ret = logp_buf.reshape(-1), adv.reshape(-1), ret.reshape(-1)

            enc_grads, stats = [], {"pg_loss": [], "v_loss": [], "entropy": [], "approx_kl": [], "clipfrac": [], "grad_norm": []}
            for _ in range(p.n_epochs):
                perm = torch.randperm(batch)
                for start in range(0, batch, mb_size):
                    mb = perm[start : start + mb_size]
                    d = agent.dist(b_obs[mb])
                    if task.discrete:
                        new_logp, ent = d.log_prob(b_act[mb]), d.entropy()
                    else:
                        new_logp, ent = d.log_prob(b_act[mb]).sum(-1), d.entropy().sum(-1)
                    ratio = (new_logp - b_logp[mb]).exp()
                    mb_adv = b_adv[mb]
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                    pg_loss = torch.max(-mb_adv * ratio, -mb_adv * ratio.clamp(1 - p.clip, 1 + p.clip)).mean()
                    v_loss = 0.5 * ((agent.value(b_obs[mb]) - b_ret[mb]) ** 2).mean()
                    loss = pg_loss - p.ent_coef * ent.mean() + p.vf_coef * v_loss
                    opt.zero_grad()
                    loss.backward()
                    # encoder gradients (critic does not touch the encoder) before clipping
                    enc_grads.append(torch.cat([q.grad.detach().flatten() for q in enc_params if q.grad is not None]))
                    gn = torch.nn.utils.clip_grad_norm_(agent.parameters(), p.max_grad_norm)
                    opt.step()
                    with torch.no_grad():
                        logr = new_logp - b_logp[mb]
                        stats["approx_kl"].append(((logr.exp() - 1) - logr).mean().item())
                        stats["clipfrac"].append(((ratio - 1).abs() > p.clip).float().mean().item())
                    stats["pg_loss"].append(pg_loss.item())
                    stats["v_loss"].append(v_loss.item())
                    stats["entropy"].append(ent.mean().item())
                    stats["grad_norm"].append(float(gn))

            g = torch.stack(enc_grads)
            row = {k: float(np.mean(v)) for k, v in stats.items()}
            row.update(
                update=update,
                global_step=global_step,
                ep_return_mean=float(np.mean(recent)) if recent else float("nan"),
                ep_len_mean=float(np.mean(recent_len)) if recent_len else float("nan"),
                n_episodes=n_eps,
                grad_var=float(g.var(0).mean()),
                sps=int(global_step / (time.time() - t0)),
            )
            if update % log_every == 0 or update == n_updates:
                writer.writerow(row)
                fh.flush()
            if verbose and update % 10 == 0:
                print(f"[{cfg.variant} s{cfg.seed}] upd {update}/{n_updates} step {global_step} ret {row['ep_return_mean']:.1f} sps {row['sps']}", flush=True)

    envs.close()
    torch.save({"state_dict": agent.state_dict(), "config": cfg.to_dict()}, out / "model.pt")
    ev = evaluate(agent, cfg)
    ev.update(train_seconds=time.time() - t0, total_steps=global_step)
    (out / "eval.json").write_text(json.dumps(ev, indent=2))
    (out / "done").write_text("ok\n")
    return out


def load_agent(run_dir: Path | str) -> Agent:
    ckpt = torch.load(Path(run_dir) / "model.pt", weights_only=False)
    agent = Agent(RunConfig.from_dict(ckpt["config"]))
    agent.load_state_dict(ckpt["state_dict"])
    agent.eval()
    return agent
