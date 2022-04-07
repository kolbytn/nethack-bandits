import torch
import torch.nn.functional as F
import random
from collections import deque
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy


def train(env, model, train_steps=1e5, replay_size=1e4, train_start=1000, target_update=1000, max_steps=None,
          batch_size=8, eps=1, eps_decay=.9999, eps_min=.1, eval_freq=1, smooth=1000, seed=42, bmi_freq=10000):
    random.seed(seed)
    if max_steps is None:
        max_steps = len(env.QUESTIONS) * env.NUM_EFFECTS * 2 + 1

    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    returns = []
    eval_returns = []
    replay = deque(maxlen=int(replay_size))
    for train_step in tqdm(range(int(train_steps))):
        obs = env.reset()
        
        if random.random() < eps:
            action = random.randint(0, model.out_size - 1)
        else:
            prepared_obs = prepare_obs(obs)
            model.eval()
            qs = model(prepared_obs).squeeze(0)
            action = torch.argmax(qs).item()

        _, reward, done, _ = env.step(action)

        replay.append((obs, action, reward, done))
        returns.append(reward)

        if train_step % eval_freq == 0:
            obs = env.reset(eval=True)
            prepared_obs = prepare_obs(obs)
            model.eval()
            qs = model(prepared_obs).squeeze(0)
            action = torch.argmax(qs).item()
            obs, reward, done, _ = env.step(action)
            eval_returns.append(reward)

        if eps > eps_min:
            eps *= eps_decay

        if len(replay) > train_start:
            batch = random.sample(replay, batch_size)
            observations, actions, rewards, dones = prepare_batch(batch)
            model.train()
            loss = calculate_loss(model, observations, actions, rewards, dones)
            optim.zero_grad()
            loss.backward()
            optim.step()

        if train_step % smooth == 0:
            if len(returns) > smooth:
                # graph("returns_{}_{}.png".format(env.feature, seed), returns, smooth)
                log("2_28b_results/returns_{}_{}.txt".format(env.feature, seed), returns)
            if len(eval_returns) > smooth:
                # graph("eval_returns_{}_{}.png".format(env.feature, seed), eval_returns, smooth)
                log("2_28b_results/eval_returns_{}_{}.txt".format(env.feature, seed), eval_returns)


def calculate_loss(model, obs, action, reward, dones):
    action = action.to(model.device).long()
    reward = reward.to(model.device).float()
    dones = dones.to(model.device).float()

    qs = model(obs)
    qs = torch.gather(qs, 1, action.unsqueeze(0))

    return torch.mean((qs - reward) ** 2)


def prepare_batch(batch):
    observations = []
    actions = []
    rewards = []
    dones = []
    for o, a, r, d, n in batch:
        observations.append(o)
        actions.append(a)
        rewards.append(r)
        dones.append(d)
    observations = prepare_obs(observations)
    return observations, torch.tensor(actions), torch.tensor(rewards), torch.tensor(dones)


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
    if smooth is not None:
        data = [sum(data[d:d+smooth]) / smooth for d in range(0, len(data) - smooth, smooth)]
        plt.plot(list(range(smooth, (len(data)+1)*smooth, smooth)), data)
    else:
        plt.plot(data)
    plt.savefig(path)


def log(path, data):
    with open(path, "w") as f:
        for d in data:
            f.write(str(d) + "\n")
