import json
from functools import lru_cache
import numpy as np
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Model


class MonsterEnv:
    EFFECT2STR = {
        "FIRE_RES": "fire",
        "SHOCK_RES": "shock",
        "SLEEP_RES": "sleep",
        "POISON_RES": "poison",
        "COLD_RES": "cold",
        "DISINT_RES": "disintegrat",
        "POISONOUS": "poisonous",
        "ACIDIC": "acid",
        "INT": "intellig",
        "TELEPAT": "telepathy",
        "TELEPORT_CONTROL": "teleport",
        "PETRIFY": "petrif",
        "FAST": "fast",
        "STUNNED": "stun",
        "INVIS": "invisib",
        "FOOD_POISONING": "food poison",
        "HALLUC": "halluc",
        "STR": "strength",
        "POLYMORPH": "polymorph",
    }
    QUESTIONS = [
        "What resistance does eating a [monster] corpse confer?",
        "What does eating a [monster] corpse confer?",
        "What does a [monster] confer?",
        "What happens when you eat a [monster] corpse?",
        "What resistance does eating a [monster] confer?"
    ]
    NUM_MONSTERS = 388
    NUM_EFFECTS = 5

    def __init__(self, device="cuda", feature="bert", seed=42):
        np.random.seed(seed)

        with open("corpses.json", "r") as f:
            self.all_monsters = json.load(f)
        monster_names = list(self.all_monsters.keys())
        np.random.shuffle(monster_names)

        self.train_monsters = {k: v for k, v in self.all_monsters.items() if k in monster_names[:int(self.NUM_MONSTERS * .8)]}
        self.eval_monsters = {k: v for k, v in self.all_monsters.items() if k in monster_names[int(self.NUM_MONSTERS * .8):]}
        self.train_effects = self.get_effects(self.train_monsters)
        self.eval_effects = self.get_effects(self.eval_monsters)
        self.all_effects = self.get_effects(self.all_monsters)
        
        with open("wiki.json", "r") as f:
            self.wiki = json.load(f)

        self.device = device
        self.feature = feature

        if self.feature == "t5":
            self.tokenizer = T5Tokenizer.from_pretrained("t5-3b")
            self.model = T5Model.from_pretrained("t5-3b", return_dict=True).to(device)
            self.model.requires_grad = False
        elif self.feature == "qa":
            self.tokenizer = T5Tokenizer.from_pretrained("allenai/unifiedqa-t5-3b")
            self.model = T5ForConditionalGeneration.from_pretrained("allenai/unifiedqa-t5-3b").to(device)
            self.model.requires_grad = False
        else:
            self.tokenizer = None
            self.model = None

        self.monster_choices = None
        self.goal = None

        self.queries = self.get_queries()

    def reset(self, eval=False):
        if eval:
            curr_monsters = self.eval_monsters
            curr_effects = self.eval_effects
        else:
            curr_monsters = self.train_monsters
            curr_effects = self.train_effects

        self.goal = np.random.choice(list(curr_effects.keys()))
        self.monster_choices = []
        self.monster_choices.append(
            np.random.choice(list(curr_effects[self.goal].keys()))  # Add monster with effect
        )
        self.monster_choices.append(
            np.random.choice([m for m in curr_monsters.keys() if m not in list(curr_effects[self.goal].keys())])
        )
        np.random.shuffle(self.monster_choices)

        obs = self.prepare_obs(self.goal, self.monster_choices[0], self.monster_choices[1])         

        return obs

    def step(self, action):
        reward = int(self.goal in self.all_monsters[self.monster_choices[action]])
        done = True
        obs = self.prepare_obs(self.goal, self.monster_choices[0], self.monster_choices[1])
        info = {
            "goal": self.goal,
            "monster_choices": self.monster_choices,
        }

        return obs, reward, done, info

    def get_effects(self, monsters):
        effects = dict()
        for m in monsters:
            for e in monsters[m]:
                if "_RES" in e and e != "DISINT_RES":  # Only using resistance effects
                    if e not in effects:
                        effects[e] = dict()
                    if monsters[m][e] > 0:
                        effects[e][m] = monsters[m][e]
        return effects

    def prepare_obs(self, goal, monster1, monster2):
        obs = {
            "goal": list(self.all_effects.keys()).index(goal),
        }   
        
        if self.feature == "t5":
            obs["monster1"] = self.run_t5_model(monster1)
            obs["monster2"] = self.run_t5_model(monster2)
        elif self.feature == "rnn":
            obs["monster1"] = monster1 + "\n" + self.wiki[monster1]
            obs["monster2"] = monster2 + "\n" + self.wiki[monster2]
        elif self.feature == "qa":
            obs["monster1"] = self.get_monster_qa(monster1)
            obs["monster2"] = self.get_monster_qa(monster1)
        elif self.feature == "truth":
            obs["monster1"] = self.get_monster_truth(monster1)
            obs["monster2"] = self.get_monster_truth(monster1)
        else:
            obs["monster1"] = list(self.all_monsters.keys()).index(monster1)
            obs["monster2"] = list(self.all_monsters.keys()).index(monster2)

        return obs

    @lru_cache(maxsize=NUM_MONSTERS)
    def get_monster_truth(self, monster):
        queries = np.zeros((self.NUM_EFFECTS, len(self.QUESTIONS)))
        for e in range(self.NUM_EFFECTS):
            for q in range(len(self.QUESTIONS)):
                effect = list(self.all_effects.keys())[e]
                queries[e][q] = int(effect in list(self.all_monsters[monster].keys()))
        return queries

    @lru_cache(maxsize=NUM_MONSTERS)
    def get_monster_qa(self, monster):
        queries = np.zeros((self.NUM_EFFECTS, len(self.QUESTIONS)))
        for q in range(len(self.QUESTIONS)):
            qry = self.QUESTIONS[q].replace("[monster]", monster) + \
                " \\n " + self.wiki[monster]
            a = self.run_qa_model(qry)
            for e in range(self.NUM_EFFECTS):
                queries[e][q] = int(self.EFFECT2STR[list(self.all_effects.keys())[e]] in a)
        return queries

    @lru_cache(maxsize=NUM_MONSTERS*5)
    def run_qa_model(self, q):
        input_ids = self.tokenizer.encode(q, return_tensors="pt").to(self.device)
        res = self.model.generate(input_ids)
        a = self.tokenizer.batch_decode(res, skip_special_tokens=True)
        return a

    @lru_cache(maxsize=NUM_MONSTERS)
    def run_t5_model(self, monster):
        with torch.no_grad():
            monster_info = self.tokenizer(monster + "\n" + self.wiki[monster], return_tensors="pt").input_ids
            monster_info = self.model(input_ids=monster_info.to(self.device), decoder_input_ids=monster_info.to(self.device))
            return monster_info.last_hidden_state[0, 0].cpu().detach().numpy()
