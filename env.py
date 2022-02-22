import json
from functools import lru_cache
import numpy as np
import torch
from transformers import BertTokenizer, BertModel, T5Tokenizer, T5ForConditionalGeneration, T5Model


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

        if self.feature == "bert":
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertModel.from_pretrained('bert-base-uncased', return_dict=True).to(device)
            self.model.requires_grad = False
        elif self.feature == "t5":
            self.tokenizer = T5Tokenizer.from_pretrained("t5-3b")
            self.model = T5Model.from_pretrained("t5-3b", return_dict=True).to(device)
            self.model.requires_grad = False
        elif self.feature == "qa" or self.feature == "full":
            self.tokenizer = T5Tokenizer.from_pretrained("allenai/unifiedqa-t5-3b")
            self.model = T5ForConditionalGeneration.from_pretrained("allenai/unifiedqa-t5-3b").to(device)
            self.model.requires_grad = False
        else:
            self.tokenizer = None
            self.model = None

        self.monster_choices = None
        self.goal = None
        self.queries = np.zeros((2, self.NUM_EFFECTS, len(self.QUESTIONS)))
        self.queried = np.zeros((2, self.NUM_EFFECTS, len(self.QUESTIONS)))

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

        self.queries = np.zeros((2, self.NUM_EFFECTS, len(self.QUESTIONS)))
        self.queried = np.zeros((2, self.NUM_EFFECTS, len(self.QUESTIONS)))
        if self.feature == "full" or self.feature == "full_truth":
            for m in range(2):
                for e in range(self.NUM_EFFECTS):
                    for q in range(len(self.QUESTIONS)):
                        self.query(m, e, q)

        obs = self.prepare_obs()         

        return obs

    def step(self, action):
        if action < 2:
            reward = int(self.goal in self.all_monsters[self.monster_choices[action]])
            done = True

        else:
            q = action - 2
            idx = np.zeros_like(self.queries).flatten()
            idx[q] = 1
            idx = np.reshape(idx, self.queries.shape)
            idx = np.where(idx)
            self.query(idx[0].item(), idx[1].item(), idx[2].item())
            
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
            "goal": list(self.all_effects.keys()).index(self.goal),
        }   
        
        if self.feature == "bert":
            obs["monster1"] = self.run_bert_model(self.monster_choices[0])
            obs["monster2"] = self.run_bert_model(self.monster_choices[1])
        elif self.feature == "t5":
            obs["monster1"] = self.run_t5_model(self.monster_choices[0])
            obs["monster2"] = self.run_t5_model(self.monster_choices[1])
        elif self.feature in ["rnn", "finetune"]:
            obs["monster1"] = self.monster_choices[0] + "\n" + self.wiki[self.monster_choices[0]]
            obs["monster2"] = self.monster_choices[1] + "\n" + self.wiki[self.monster_choices[1]]
        elif self.feature in ["qa", "full", "truth", "full_truth"]:
            obs["monster1"] = np.concatenate((self.queried[0].flatten(), self.queries[0].flatten()))
            obs["monster2"] = np.concatenate((self.queried[1].flatten(), self.queries[1].flatten()))
        else:
            obs["monster1"] = list(self.all_monsters.keys()).index(self.monster_choices[0])
            obs["monster2"] = list(self.all_monsters.keys()).index(self.monster_choices[1])

        return obs

    def query(self, m, e, q):
        self.queried[m][e][q] = 1
        if self.feature == "truth" or self.feature == "full_truth":
            monster = self.monster_choices[m]
            effect = list(self.all_effects.keys())[e]
            self.queries[m][e][q] = int(effect in list(self.all_monsters[monster].keys()))
        else:
            qry = self.QUESTIONS[q].replace("[monster]", self.monster_choices[m]) + \
                " \\n " + self.wiki[self.monster_choices[m]]
            a = self.run_qa_model(qry)
            self.queries[m][e][q] = int(self.EFFECT2STR[list(self.all_effects.keys())[e]] in a)

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

    @lru_cache(maxsize=NUM_MONSTERS)
    def run_bert_model(self, monster):
        with torch.no_grad():
            monster1_info = self.tokenizer([
                monster + "\n" + self.wiki[monster]
            ], return_tensors="pt", padding=True, truncation=True)
            return self.model(**monster1_info.to(self.device)).last_hidden_state[0, 0].cpu().detach().numpy()
