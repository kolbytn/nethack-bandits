import json
from functools import lru_cache
import numpy as np
import torch
from transformers import BertTokenizer, BertModel, T5Tokenizer, T5ForConditionalGeneration, T5Model
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from collections import defaultdict
from DPR import get_most_similar

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
        "What resistance does eating a [monster] corpse confer?"
        ,"What does eating a [monster] corpse confer?",
        "What does a [monster] confer?",
        "What happens when you eat a [monster] corpse?",
        "What resistance does eating a [monster] confer?"
    ]
    NUM_MONSTERS = 388
    NUM_EFFECTS = 5

    def __init__(self, device="cuda", feature="bert", seed=42):
        np.random.seed(seed)

        with open("No-Queries-Nethack-bandit/nethack-bandits/corpses.json", "r") as f:
            self.all_present_monsters = json.load(f)

        with open("No-Queries-Nethack-bandit/nethack-bandits/wiki.json", "r") as f:
            self.wiki = json.load(f)

        monstor_wiki = set(list(self.wiki.keys()))
        monster_names = list()
        corpse_monster_names = list(self.all_present_monsters.keys())
        self.all_monsters = dict()
        for name in corpse_monster_names:
            if name in monstor_wiki:
                self.all_monsters[name] = self.all_present_monsters[name]

        monster_names = list(self.all_monsters.keys())
        self.NUM_MONSTERS = len(monster_names)
        np.random.shuffle(monster_names)
        self.train_monsters = {k: v for k, v in self.all_monsters.items() if k in monster_names[:int(self.NUM_MONSTERS * .8)]}
        self.eval_monsters = {k: v for k, v in self.all_monsters.items() if k in monster_names[int(self.NUM_MONSTERS * .8):]}
        self.train_effects = self.get_effects(self.train_monsters)
        self.eval_effects = self.get_effects(self.eval_monsters)
        self.all_effects = self.get_effects(self.all_monsters)
        
        self.device = device
        self.feature = feature

        self.IR_data = {}
        self.IR_all_Q_data = defaultdict(list)

        if self.feature == "ir":
            for question in self.QUESTIONS:
                for monstor in monster_names:
                    query_formed = question.replace("[monster]", monstor)
                    self.IR_all_Q_data[monstor].append(get_most_similar(query_formed, self.wiki[monstor], self.wiki['corpse'],monstor)[1])
                    self.IR_data[query_formed] = get_most_similar(query_formed, self.wiki[monstor], self.wiki['corpse'],monstor)
                    
            print("Done creating IR dict")

        if self.feature == "bert":
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertModel.from_pretrained('bert-base-uncased', return_dict=True).to(device)
            self.model.requires_grad = False
        elif self.feature == "t5"  or self.feature == "ir":
            self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
            self.model = T5Model.from_pretrained("t5-small", return_dict=True).to(device)
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
        
        if self.feature == "bert":
            obs["monster1"] = self.run_bert_model(self.monster_choices[0])
            obs["monster2"] = self.run_bert_model(self.monster_choices[1])
        elif self.feature == "t5":
            obs["monster1"] = self.run_t5_model(self.monster_choices[0])
            obs["monster2"] = self.run_t5_model(self.monster_choices[1])
        elif self.feature in ["ir"]:
            obs["monster1"] = self.run_t5_model_ir(self.monster_choices[0])
            obs["monster2"] = self.run_t5_model_ir(self.monster_choices[1])
        elif self.feature in ["rnn", "finetune"]:
            obs["monster1"] = self.monster_choices[0] + "\n" + self.wiki[self.monster_choices[0]]
            obs["monster2"] = self.monster_choices[1] + "\n" + self.wiki[self.monster_choices[1]]
        elif self.feature in ["qa", "full", "truth", "full_truth"]:
            obs["monster1"] = np.concatenate((self.queried[0].flatten(), self.queries[0].flatten()))
            obs["monster2"] = np.concatenate((self.queried[1].flatten(), self.queries[1].flatten()))
        else:
            obs["monster1"] = list(self.all_monsters.keys()).index(monster1)
            obs["monster2"] = list(self.all_monsters.keys()).index(monster2)

        return obs

    def query(self, m, e, q):
        self.queried[m][e][q] = 1
        if self.feature == "truth" or self.feature == "full_truth":
            monster = self.monster_choices[m]
            effect = list(self.all_effects.keys())[e]
            self.queries[m][e][q] = int(effect in list(self.all_monsters[monster].keys()))
        elif self.feature == 'ir':
            question = self.QUESTIONS[q].replace("[monster]", self.monster_choices[m])
            doc_id = self.IR_data[question][0]
            Ir_result = self.IR_data[question][1]
            #update the condition and check
            qry = question + " \\n " + Ir_result
            a = self.run_qa_model(qry)
            monstor = self.monster_choices[m]
            if monstor in docid:
                mon = True
            else:
                mon = monstor in Ir_result
            val = self.EFFECT2STR[list(self.all_effects.keys())[e]] in Ir_result and mon
            self.queries[m][e][q] = int(val)
        else:
            qry = self.QUESTIONS[q].replace("[monster]", self.monster_choices[m]) + \
                " \\n " + self.wiki[self.monster_choices[m]]
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

    @lru_cache(maxsize=NUM_MONSTERS)
    def run_bert_model(self, monster):
        with torch.no_grad():
            monster1_info = self.tokenizer([
                monster + "\n" + self.wiki[monster]
            ], return_tensors="pt", padding=True, truncation=True)
            return self.model(**monster1_info.to(self.device)).last_hidden_state[0, 0].cpu().detach().numpy()

    @lru_cache(maxsize=NUM_MONSTERS)
    def run_t5_model_ir(self, monster):
        with torch.no_grad():
            if len(self.IR_all_Q_data[monster]) == 0:
                monster_info = self.tokenizer(monster, return_tensors="pt").input_ids
            else:
                monster_info = self.tokenizer(monster + "\n" + " ".join(self.IR_all_Q_data[monster]), return_tensors="pt").input_ids
            monster_info = self.model(input_ids=monster_info.to(self.device), decoder_input_ids=monster_info.to(self.device))
            return monster_info.last_hidden_state[0, 0].cpu().detach().numpy()

    def get_IR_part(self,query, monstor): 
        
        paragraph_relevant = self.get_sub_sections_in_document(query, self.wiki[monstor], ".")
        return paragraph_relevant

    def get_sub_sections_in_document(self,q, document, seperator):
    
        paragraphs = document.split(seperator)
        if len(paragraphs) == 1:
            return paragraphs[0]

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(paragraphs)
        X = X.T.toarray()
        df = pd.DataFrame(X, index=vectorizer.get_feature_names_out())

        q = [q]
        q_vec = vectorizer.transform(q).toarray().reshape(df.shape[0],)
        sim = {}

        # Calculate the similarity
        for i in range(len(paragraphs)):
            if np.linalg.norm(df.loc[:, i]) * np.linalg.norm(q_vec) == 0:
                sim[i] = 0
            else:
                sim[i] = np.dot(df.loc[:, i].values, q_vec) / np.linalg.norm(df.loc[:, i]) * np.linalg.norm(q_vec)

        sim_sorted = sorted(sim.items(), key=lambda x: x[1], reverse=True)[0:1]
        res = ""

        for k, v in sim_sorted:
            if v != 0.0:
                res += paragraphs[k]
        return res


