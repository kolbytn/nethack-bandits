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
    target_model = deepcopy(model) if env.feature in ["qa", "truth"] else None

    returns = []
    eval_returns = []
    train_bmis = []
    eval_bmis = []
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

        if train_step % bmi_freq == 0:  # Only setup for non QA methods
            train_obs, train_act = env.get_policy()
            train_bmis.append(calculate_bmi(train_obs, train_act, model))
            eval_obs, eval_act = env.get_policy(eval=True)
            eval_bmis.append(calculate_bmi(eval_obs, eval_act, model))

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
                # graph("returns_{}_{}.png".format(env.feature, seed), returns, smooth)
                log("2_28b_results/returns_{}_{}.txt".format(env.feature, seed), returns)
            if len(eval_returns) > smooth:
                # graph("eval_returns_{}_{}.png".format(env.feature, seed), eval_returns, smooth)
                log("2_28b_results/eval_returns_{}_{}.txt".format(env.feature, seed), eval_returns)
            # graph("bmi_{}_{}.png".format(env.feature, seed), bmis, None)
            log("2_28b_results/bmi_{}_{}.txt".format(env.feature, seed), train_bmis)
            log("2_28b_results/eval_bmi_{}_{}.txt".format(env.feature, seed), eval_bmis)


def calculate_bmi(obs, act, model):
    model.eval()
    with torch.no_grad():
        batch_size = 32768
        num_batches = int(len(obs) / batch_size) + 1
        qs = []
        for b in range(num_batches):
            if b == num_batches - 1:
                batch = obs[b*batch_size:]
            else:
                batch = obs[b*batch_size:(b+1)*batch_size]
            prepared_obs = prepare_obs(batch)
            qs.append(model(prepared_obs))
        qs = torch.cat(qs, dim=0)
        model_probs = F.softmax(qs, dim=1)
        optim_probs = F.one_hot(torch.tensor(act), num_classes=2).float().to(model.device)
        mean_model_probs = torch.mean(model_probs, dim=0).unsqueeze(0)
        mean_optim_probs = torch.mean(optim_probs, dim=0).unsqueeze(0)
        cross_means = F.cross_entropy(mean_model_probs, mean_optim_probs)
        mean_cross = F.cross_entropy(model_probs, optim_probs)
        bmi = (cross_means - mean_cross).item()
    return bmi


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
