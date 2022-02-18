from env import MonsterEnv
from model import *
from train import train


if __name__ == "__main__":
    env = MonsterEnv()
    model = BERTQ(len(env.effects))
    train(env, model)