from multiprocessing.spawn import prepare
import numpy as np
import json


class MonsterEnv:
    def __init__(self):

        with open("corpses.json", "r") as f:
            self.monsters = json.load(f)
        with open("wiki.json", "r") as f:
            self.wiki = json.load(f)

        self.effects = dict()
        for m in self.monsters:
            for e in self.monsters[m]:
                if e not in self.effects:
                    self.effects[e] = dict()
                if self.monsters[m][e] > 0:
                    self.effects[e][m] = self.monsters[m][e]

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

        obs = self.prepare_obs()         

        return obs

    def step(self, action):
        obs = self.prepare_obs()         
        
        selected_monster = self.monsters[self.monster_choices[action]]
        other_monster = self.monsters[self.monster_choices[int(not action)]]

        selected_prob = selected_monster[self.goal] if self.goal in selected_monster else 0
        other_prob = other_monster[self.goal] if self.goal in other_monster else 0

        reward = int(selected_prob >= other_prob)

        done = True

        info = {
            "goal": self.goal,
            "monster_choices": self.monster_choices,
        }

        return obs, reward, done, info

    def prepare_obs(self):
        return {
            "goal": list(self.effects.keys()).index(self.goal),
            "monster1": self.wiki[self.monster_choices[0]],
            "monster2": self.wiki[self.monster_choices[1]]
        }   
