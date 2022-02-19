import torch
import random
from collections import deque
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy


def train(env, model, train_steps=1e5, replay_size=1e4, train_start=1000, target_update=100,
          batch_size=8, eps=1, esp_decay=.9999, eps_min=.1, smooth=100):

    optim = torch.optim.Adam(model.monster_encoder.parameters(), lr=1e-4)
    target_model = deepcopy(model) if env.feature == "qa" else None

    returns = []
    replay = deque(maxlen=int(replay_size))
    for train_step in tqdm(range(int(train_steps))):
        obs = env.reset()
        
        done = False
        while not done:
            if random.random() < eps:
                action = random.randint(0, model.out_size - 1)
            else:
                prepared_obs = prepare_obs(obs)
                model.eval()
                qs = model(prepared_obs)
                action = torch.argmax(qs).item()

            next_obs, reward, done, _ = env.step(action)

            replay.append((obs, action, reward, done, next_obs))
            obs = next_obs

        if eps > eps_min:
            eps *= esp_decay

        if len(replay) > train_start:
            batch = random.sample(replay, batch_size)
            observations, actions, rewards, dones, next_observations = prepare_batch(batch)
            model.train()
            loss = calculate_loss(model, target_model, observations, actions, rewards, dones, next_observations)
            optim.zero_grad()
            loss.backward()
            optim.step()
            
        if train_step % target_update == 0 and env.feature == "qa":
            target_model = deepcopy(model)

        returns.append(reward)
        if train_step % smooth == 0:
            if len(returns) > smooth:
                graph("returns.png", returns, smooth)


def calculate_loss(model, target_model, obs, action, reward, dones, next_observations):
    action = action.to(model.device).long()
    reward = reward.to(model.device).float()
    dones = dones.to(model.device).float()

    qs = model(obs)
    qs = torch.gather(qs, 1, action.unsqueeze(0))
    if target_model is None:
        target = reward
    else:
        next_qs = target_model(next_observations)
        target = reward + .9 * torch.max(next_qs, dim=1)[0]

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
    plt.plot(data)
    plt.savefig(path)
