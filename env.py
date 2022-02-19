import numpy as np
import json
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

    def __init__(self, device="cuda", feature="bert"):

        with open("corpses.json", "r") as f:
            self.monsters = json.load(f)
        with open("wiki.json", "r") as f:
            self.wiki = json.load(f)

        self.monsters = {key: self.monsters[key] for key in list(self.monsters)}

        self.effects = dict()
        for m in self.monsters:
            for e in self.monsters[m]:
                if "_RES" in e and e != "DISINT_RES":  # Only using resistance effects
                    if e not in self.effects:
                        self.effects[e] = dict()
                    if self.monsters[m][e] > 0:
                        self.effects[e][m] = self.monsters[m][e]

        self.device = device
        self.feature = feature

        if self.feature == "bert":
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertModel.from_pretrained('bert-base-uncased', return_dict=True).to(device)
            self.model.requires_grad = False
        elif self.feature == "qa":
            self.tokenizer = T5Tokenizer.from_pretrained("allenai/unifiedqa-t5-base")
            self.model = T5ForConditionalGeneration.from_pretrained("allenai/unifiedqa-t5-base").to(device)
            self.model.requires_grad = False
        else:
            self.tokenizer = None
            self.model = None

        self.monster_choices = None
        self.goal = None
        self.queries = np.zeros((2, len(self.effects)))

    def reset(self):
        self.goal = np.random.choice(list(self.effects.keys()))

        self.monster_choices = []
        self.monster_choices.append(
            np.random.choice(list(self.effects[self.goal].keys()))  # Add monster with effect
        )
        self.monster_choices.append(
            np.random.choice([m for m in self.monsters.keys() if m not in self.monster_choices])  # Add a different monster
        )
        np.random.shuffle(self.monster_choices)

        self.queries = np.zeros((2, len(self.effects)))

        obs = self.prepare_obs()         

        return obs

    def step(self, action):
        if action < 2:
            selected_monster = self.monsters[self.monster_choices[action]]
            selected_prob = selected_monster[self.goal] if self.goal in selected_monster else 0
            reward = int(np.random.random() < selected_prob)

            done = True

        else:
            q = action - 2
            m = int(q >= len(self.effects))
            e = q - len(self.effects) if q >= len(self.effects) else q
            self.query(m, e)
            
            reward = 0
            done = False

        obs = self.prepare_obs()
        info = {
            "goal": self.goal,
            "monster_choices": self.monster_choices,
        }

        return obs, reward, done, info

    def prepare_obs(self):
        obs = {
            "goal": list(self.effects.keys()).index(self.goal),
        }   
        
        if self.feature == "bert":
            with torch.no_grad():
                monster1_info = self.tokenizer([
                    self.monster_choices[0] + "\n" + self.wiki[self.monster_choices[0]]
                ], return_tensors="pt", padding=True, truncation=True)
                obs["monster1_info"] = self.model(**monster1_info.to(self.device)).last_hidden_state[0, 0].cpu().detach().numpy()
                
                monster2_info = self.tokenizer([
                    self.monster_choices[1] + "\n" + self.wiki[self.monster_choices[1]]
                ], return_tensors="pt", padding=True, truncation=True)
                obs["monster2_info"] = self.model(**monster2_info.to(self.device)).last_hidden_state[0, 0].cpu().detach().numpy()
        elif self.feature == "qa":
            obs["monster1_qa"] = self.queries[0]
            obs["monster2_qa"] = self.queries[1]
        else:
            obs["monster1_id"] = list(self.monsters.keys()).index(self.monster_choices[0])
            obs["monster2_id"] = list(self.monsters.keys()).index(self.monster_choices[1])

        return obs

    def query(self, m, e):
        q = "What resistance does eating a {} corpse confer? \\n {}".format(
            self.monster_choices[m],
            self.wiki[self.monster_choices[m]]
        )        

        input_ids = self.tokenizer.encode(q, return_tensors="pt").to(self.device)
        res = self.model.generate(input_ids)
        a = self.tokenizer.batch_decode(res, skip_special_tokens=True)

        self.queries[m][e] = int(self.EFFECT2STR[list(self.effects.keys())[e]] in a)
