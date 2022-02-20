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
        model = BERTQ(env.model.config.hidden_size, env.NUM_EFFECTS)
    elif args.feature in ["qa", "truth"]:
        model = QuestionAnswerQ(env.NUM_EFFECTS)
    elif args.feature in ["full", "full_truth"]:
        model = QuestionAnswerQ(env.NUM_EFFECTS, out_size=2)
    else:
        model = OneHotQ(env.NUM_MONSTERS, env.NUM_EFFECTS)
    train(env, model)
