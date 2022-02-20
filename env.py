import json
from functools import lru_cache
import numpy as np
import torch
from transformers import BertTokenizer, BertModel, T5Tokenizer, T5ForConditionalGeneration


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
    NUM_MONSTERS = 388
    NUM_EFFECTS = 5

    def __init__(self, device="cuda", feature="bert"):

        with open("corpses.json", "r") as f:
            monster_data = json.load(f)
        monster_names = list(monster_data.keys())
        np.random.shuffle(monster_names)

        self.train_monsters = {k: v for k, v in monster_data.items() if k in monster_names[:int(self.NUM_MONSTERS * .8)]}
        self.eval_monsters = {k: v for k, v in monster_data.items() if k in monster_names[int(self.NUM_MONSTERS * .8):]}
        self.train_effects = self.get_effects(self.train_monsters)
        self.eval_effects = self.get_effects(self.eval_monsters)
        
        with open("wiki.json", "r") as f:
            self.wiki = json.load(f)

        self.device = device
        self.feature = feature

        if self.feature == "bert":
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertModel.from_pretrained('bert-base-uncased', return_dict=True).to(device)
            self.model.requires_grad = False
        elif self.feature == "qa" or self.feature == "full":
            self.tokenizer = T5Tokenizer.from_pretrained("allenai/unifiedqa-t5-base")
            self.model = T5ForConditionalGeneration.from_pretrained("allenai/unifiedqa-t5-base").to(device)
            self.model.requires_grad = False
        else:
            self.tokenizer = None
            self.model = None

        self.monster_choices = None
        self.goal = None
        self.curr_monsters = self.train_monsters
        self.curr_effects = self.train_effects
        self.queries = np.zeros((2, self.NUM_EFFECTS))
        self.queried = np.zeros((2, self.NUM_EFFECTS))

    def reset(self, eval=False):
        if eval:
            self.curr_monsters = self.eval_monsters
            self.curr_effects = self.eval_effects
        else:
            self.curr_monsters = self.train_monsters
            self.curr_effects = self.train_effects

        self.goal = np.random.choice(list(self.curr_effects.keys()))
        self.monster_choices = []
        self.monster_choices.append(
            np.random.choice(list(self.curr_effects[self.goal].keys()))  # Add monster with effect
        )
        self.monster_choices.append(
            np.random.choice([m for m in self.curr_monsters.keys() if m not in list(self.curr_effects[self.goal].keys())])
        )
        np.random.shuffle(self.monster_choices)

        self.queries = np.zeros((2, self.NUM_EFFECTS))
        self.queried = np.zeros((2, self.NUM_EFFECTS))
        if self.feature == "full" or self.feature == "full_truth":
            for m in range(2):
                for e in range(self.NUM_EFFECTS):
                    self.query(m, e)

        obs = self.prepare_obs()         

        return obs

    def step(self, action):
        if action < 2:
            reward = int(self.goal in self.curr_monsters[self.monster_choices[action]])
            done = True

        else:
            q = action - 2
            m = int(q >= len(self.curr_effects))
            e = q - len(self.curr_effects) if q >= len(self.curr_effects) else q
            self.query(m, e)
            
            reward = 0
            done = False

        obs = self.prepare_obs()
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

    def prepare_obs(self):
        obs = {
            "goal": list(self.curr_effects.keys()).index(self.goal),
        }   
        
        if self.feature == "bert":
            obs["monster1_info"] = self.run_bert_model(self.monster_choices[0])
            obs["monster2_info"] = self.run_bert_model(self.monster_choices[1])
        elif self.feature in ["qa", "full", "truth", "full_truth"]:
            obs["monster1_qa"] = np.concatenate((self.queried[0], self.queries[0]))
            obs["monster2_qa"] = np.concatenate((self.queried[1], self.queries[1]))
        else:
            obs["monster1_id"] = list(self.curr_monsters.keys()).index(self.monster_choices[0])
            obs["monster2_id"] = list(self.curr_monsters.keys()).index(self.monster_choices[1])

        return obs

    def query(self, m, e):
        self.queried[m][e] = 1
        if self.feature == "truth" or self.feature == "full_truth":
            monster = self.monster_choices[m]
            effect = list(self.curr_effects.keys())[e]
            self.queries[m][e] = int(effect in list(self.curr_monsters[monster].keys()))
        else:
            q = "What resistance does eating a {} corpse confer? \\n {}".format(
                self.monster_choices[m],
                self.wiki[self.monster_choices[m]]
            )        
            a = self.run_qa_model(q)
            self.queries[m][e] = int(self.EFFECT2STR[list(self.curr_effects.keys())[e]] in a)
        

    @lru_cache(maxsize=NUM_MONSTERS)
    def run_qa_model(self, q):
        input_ids = self.tokenizer.encode(q, return_tensors="pt").to(self.device)
        res = self.model.generate(input_ids)
        a = self.tokenizer.batch_decode(res, skip_special_tokens=True)
        return a

    @lru_cache(maxsize=NUM_MONSTERS)
    def run_bert_model(self, monster):
        with torch.no_grad():
            monster1_info = self.tokenizer([
                monster + "\n" + self.wiki[monster]
            ], return_tensors="pt", padding=True, truncation=True)
            return self.model(**monster1_info.to(self.device)).last_hidden_state[0, 0].cpu().detach().numpy()
