import numpy as np
import json
import torch
from transformers import BertTokenizer, BertModel


class MonsterEnv:
    def __init__(self, device="cuda", feature="bert"):

        with open("corpses.json", "r") as f:
            self.monsters = json.load(f)
        with open("wiki.json", "r") as f:
            self.wiki = json.load(f)

        self.monsters = {key: self.monsters[key] for key in list(self.monsters)}

        self.effects = dict()
        for m in self.monsters:
            for e in self.monsters[m]:
                if "_RES" in e:  # Only using resistance effects
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
        else:
            self.tokenizer = None
            self.model = None

        self.monster_choices = None
        self.goal = None

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

        obs = self.prepare_obs()         

        return obs

    def step(self, action):
        obs = self.prepare_obs()    
        
        selected_monster = self.monsters[self.monster_choices[action]]
        selected_prob = selected_monster[self.goal] if self.goal in selected_monster else 0

        reward = int(np.random.random() < selected_prob)

        done = True

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
        else:
            obs["monster1_id"] = list(self.monsters.keys()).index(self.monster_choices[0])
            obs["monster2_id"] = list(self.monsters.keys()).index(self.monster_choices[1])

        return obs
