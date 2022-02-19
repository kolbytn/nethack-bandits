from env import MonsterEnv
from model import *
from train import train
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--feature", type=str)
args = parser.parse_args()


if __name__ == "__main__":
    env = MonsterEnv(feature=args.feature)
    if args.feature == "bert":
        model = BERTQ(len(env.monsters), len(env.effects), env.model.config.hidden_size)
    else:
        model = OneHotQ(len(env.monsters), len(env.effects))
    train(env, model)
