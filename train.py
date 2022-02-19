import torch
import random
from collections import deque
from tqdm import tqdm
import matplotlib.pyplot as plt


def train(env, model, train_steps=1e5, replay_size=1e4, train_start=1000, 
          batch_size=8, eps=1, esp_decay=.9999, eps_min=.1, smooth=100):
    optim = torch.optim.Adam(model.monster_encoder.parameters(), lr=1e-4)

    returns = []
    replay = deque(maxlen=int(replay_size))
    for train_step in tqdm(range(int(train_steps))):
        obs = env.reset()
        
        if random.random() < eps:
            action = random.randint(0, 1)
        else:
            prepared_obs = prepare_obs(obs)
            model.eval()
            qs = model(prepared_obs)
            action = torch.argmax(qs).item()

        _, reward, _, _ = env.step(action)

        returns.append(reward)
        replay.append((obs, action, reward))

        if eps > eps_min:
            eps *= esp_decay

        if len(replay) > train_start:
            batch = random.sample(replay, batch_size)
            observations, actions, rewards = prepare_batch(batch)
            model.train()
            loss = calculate_loss(model, observations, actions, rewards)
            optim.zero_grad()
            loss.backward()
            optim.step()

        if train_step % smooth == 0:
            if len(returns) > smooth:
                graph("returns.png", returns, smooth)


def calculate_loss(model, obs, action, reward):
    action = action.to(model.device).long()
    reward = reward.to(model.device).float()

    qs = model(obs)
    qs = torch.gather(qs, 1, action.unsqueeze(0))

    return torch.mean((qs - reward) ** 2)


def prepare_batch(batch):
    observations = []
    actions = []
    rewards = []
    for o, a, r in batch:
        observations.append(o)
        actions.append(a)
        rewards.append(r)
    observations = prepare_obs(observations)
    return observations, torch.tensor(actions), torch.tensor(rewards)


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
    plt.plot(data)
    plt.savefig(path)
