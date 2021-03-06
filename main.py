from env import MonsterEnv
from model import *
from train import train
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--feature", type=str)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()


if __name__ == "__main__":
    env = MonsterEnv(feature=args.feature, seed=args.seed)

    if args.feature in ["bert", "t5"]:
        model = LMQ(env.model.config.hidden_size, env.NUM_EFFECTS, seed=args.seed)
    elif args.feature == "rnn":
        model = RNNQ(env.NUM_EFFECTS, seed=args.seed)
    elif args.feature == "finetune":
        model = FinetuneQ(env.NUM_EFFECTS, seed=args.seed)
    elif args.feature in ["qa", "truth"]:
        model = QuestionAnswerQ(env.NUM_EFFECTS, q_size=len(env.QUESTIONS), seed=args.seed)
    elif args.feature in ["full", "full_truth"]:
        model = QuestionAnswerQ(env.NUM_EFFECTS, out_size=2, seed=args.seed)
    else:
        model = OneHotQ(env.NUM_MONSTERS, env.NUM_EFFECTS, seed=args.seed)

    if args.feature in ["qa", "truth"]:
        train(env, model, seed=args.seed, eps_decay=.99999, train_steps=1e6)
    else:
        train(env, model, seed=args.seed)
