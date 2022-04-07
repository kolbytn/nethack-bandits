import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import T5Tokenizer


class RNNQ(nn.Module):
    def __init__(self, goal_size, device="cuda", seed=42):
        super().__init__()
        torch.random.manual_seed(seed)

        self.goal_size = goal_size
        self.out_size = 2
        self.device = device
        hidden_size = 32

        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
        self.embed = nn.Embedding(self.tokenizer.vocab_size, hidden_size).to(device)
        self.model = nn.LSTM(hidden_size, hidden_size, batch_first=True).to(device)
        self.monster_encoder = nn.Sequential(
            nn.Linear(self.goal_size + hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.out_size)
        ).to(device)

    def forward(self, obs):
        goal = F.one_hot(torch.tensor(obs["goal"]).long(), num_classes=self.goal_size).to(self.device)
        
        monster1 = self.tokenizer(obs["monster1"], return_tensors="pt", padding=True, truncation=True).input_ids.to(self.device)
        monster1 = self.embed(monster1)
        monster1 = self.model(monster1)[0][:, -1]
        
        monster2 = self.tokenizer(obs["monster2"], return_tensors="pt", padding=True, truncation=True).input_ids.to(self.device)
        monster2 = self.embed(monster2)
        monster2 = self.model(monster2)[0][:, -1]
        
        return self.monster_encoder(torch.cat((goal, monster1, monster2), dim=-1).float())


class LMQ(nn.Module):
    def __init__(self, monster_size, goal_size, device="cuda", seed=42):
        super().__init__()
        torch.random.manual_seed(seed)

        self.monster_size = monster_size
        self.goal_size = goal_size
        self.out_size = 2
        self.device = device
        hidden_size = 32

        self.monster_encoder = nn.Sequential(
            nn.Linear(self.goal_size + self.monster_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.out_size)
        ).to(device)

    def forward(self, obs):
        goal = F.one_hot(torch.tensor(obs["goal"]).long(), num_classes=self.goal_size).to(self.device)
        
        monster1_info = torch.tensor(np.array(obs["monster1"])).to(self.device)
        monster2_info = torch.tensor(np.array(obs["monster2"])).to(self.device)

        return self.monster_encoder(torch.cat((goal, monster1_info, monster2_info), dim=-1).float())
        

class QuestionAnswerQ(nn.Module):
    def __init__(self, goal_size, q_size, device="cuda", out_size=None, seed=42):
        super().__init__()
        torch.random.manual_seed(seed)

        self.goal_size = goal_size
        self.out_size = 2 + q_size * self.goal_size if out_size is None else out_size
        self.device = device
        hidden_size = 32

        self.monster_encoder = nn.Sequential(
            nn.Linear(self.goal_size * q_size * 4 + self.goal_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.out_size)
        ).to(device)

    def forward(self, obs):
        goal = F.one_hot(torch.tensor(obs["goal"]).long(), num_classes=self.goal_size).to(self.device)
        
        monster1_qa = torch.tensor(np.array(obs["monster1"])).to(self.device)
        monster2_qa = torch.tensor(np.array(obs["monster2"])).to(self.device)

        return self.monster_encoder(torch.cat((goal, monster1_qa, monster2_qa), dim=-1).float())
