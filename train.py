import torch
import random
from collections import deque
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy


def train(env, model, train_steps=1e5, replay_size=1e4, train_start=1000, target_update=1000, max_steps=None,
          batch_size=8, eps=1, eps_decay=.9999, eps_min=.1, eval_freq=1, smooth=1000, seed=42):
    random.seed(seed)
    if max_steps is None:
        max_steps = len(env.QUESTIONS) * env.NUM_EFFECTS * 2 + 1

    optim = torch.optim.Adam(model.monster_encoder.parameters(), lr=1e-4)
    target_model = deepcopy(model) if env.feature in ["qa", "truth"] else None

    returns = []
    eval_returns = []
    replay = deque(maxlen=int(replay_size))
    for train_step in tqdm(range(int(train_steps))):
        obs = env.reset()
        
        for _ in range(max_steps):
            if random.random() < eps:
                action = random.randint(0, model.out_size - 1)
            else:
                prepared_obs = prepare_obs(obs)
                model.eval()
                qs = model(prepared_obs).squeeze(0)
                action = torch.argmax(qs).item()

            next_obs, reward, done, _ = env.step(action)

            replay.append((obs, action, reward, done, next_obs))
            obs = next_obs

            if done:
                break

        returns.append(reward)

        if train_step % eval_freq == 0:
            obs = env.reset(eval=True)
            for _ in range(max_steps):
                prepared_obs = prepare_obs(obs)
                model.eval()
                qs = model(prepared_obs).squeeze(0)
                action = torch.argmax(qs).item()
                obs, reward, done, _ = env.step(action)
                if done:
                    break
            eval_returns.append(reward)

        if eps > eps_min:
            eps *= eps_decay

        if len(replay) > train_start:
            batch = random.sample(replay, batch_size)
            observations, actions, rewards, dones, next_observations = prepare_batch(batch)
            model.train()
            loss = calculate_loss(model, target_model, observations, actions, rewards, dones, next_observations)
            optim.zero_grad()
            loss.backward()
            optim.step()

        if train_step % target_update == 0 and env.feature in ["qa", "truth"]:
            target_model.load_state_dict(model.state_dict())

        if train_step % smooth == 0:
            if len(returns) > smooth:
                graph("returns_{}_{}.png".format(env.feature, seed), returns, smooth)
                log("returns_{}_{}.txt".format(env.feature, seed), returns)
            if len(eval_returns) > smooth:
                graph("eval_returns_{}_{}.png".format(env.feature, seed), eval_returns, smooth)
                log("eval_returns_{}_{}.txt".format(env.feature, seed), eval_returns)


def calculate_loss(model, target_model, obs, action, reward, dones, next_observations):
    action = action.to(model.device).long()
    reward = reward.to(model.device).float()
    dones = dones.to(model.device).float()

    qs = model(obs)
    qs = torch.gather(qs, 1, action.unsqueeze(0))
    if target_model is None:
        target = reward
    else:
        next_qs = target_model(next_observations).detach()
        target = reward + .99 * torch.max(next_qs, dim=1)[0] * (1 - dones)

    return torch.mean((qs - target) ** 2)


def prepare_batch(batch):
    observations = []
    actions = []
    rewards = []
    dones = []
    next_observations = []
    for o, a, r, d, n in batch:
        observations.append(o)
        actions.append(a)
        rewards.append(r)
        dones.append(d)
        next_observations.append(n)
    observations = prepare_obs(observations)
    next_observations = prepare_obs(next_observations)
    return observations, torch.tensor(actions), torch.tensor(rewards), torch.tensor(dones), next_observations


def prepare_obs(obs):
    prepared_obs = dict()
    if isinstance(obs, list):
        for o in obs:
            for key in o:
                if key not in prepared_obs:
                    prepared_obs[key] = []
                prepared_obs[key].append(o[key])
    else:
        for key in obs:
            if key not in prepared_obs:
                prepared_obs[key] = []
            prepared_obs[key].append(obs[key])
    return prepared_obs


def graph(path, data, smooth):
    plt.clf()
    data = [sum(data[d:d+smooth]) / smooth for d in range(0, len(data) - smooth, smooth)]
    plt.plot(list(range(smooth, (len(data)+1)*smooth, smooth)), data)
    plt.savefig(path)


def log(path, data):
    with open(path, "w") as f:
        for d in data:
            f.write(str(d) + "\n")
