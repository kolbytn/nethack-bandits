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
        model = BERTQ(env.model.config.hidden_size, len(env.effects))
    elif args.feature in ["qa", "truth"]:
        model = QuestionAnswerQ(len(env.effects))
    elif args.feature in ["full", "full_truth"]:
        model = QuestionAnswerQ(len(env.effects), out_size=2)
    else:
        model = OneHotQ(len(env.monsters), len(env.effects))
    train(env, model)
