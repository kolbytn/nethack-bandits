import torch
from torchtext.data.utils import get_tokenizer
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class OneHotQ(nn.Module):
    def __init__(self, monster_size, goal_size, device="cuda"):
        super().__init__()

        self.monster_size = monster_size
        self.goal_size = goal_size
        self.out_size = 2
        self.device = device
        hidden_size = 100

        self.monster_encoder = nn.Sequential(
            nn.Linear(self.goal_size + self.monster_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        ).to(device)

    def forward(self, obs):
        goal = F.one_hot(torch.tensor(obs["goal"]).long(), num_classes=self.goal_size).to(self.device)
        monster1_id = F.one_hot(torch.tensor(obs["monster1_id"]).long(), num_classes=self.monster_size).to(self.device)
        monster2_id = F.one_hot(torch.tensor(obs["monster2_id"]).long(), num_classes=self.monster_size).to(self.device)
        
        monster1 = self.monster_encoder(torch.cat((goal, monster1_id), dim=-1).float())
        monster2 = self.monster_encoder(torch.cat((goal, monster2_id), dim=-1).float())

        return torch.cat((monster1, monster2), dim=-1)


class BERTQ(nn.Module):
    def __init__(self, monster_size, goal_size, info_size, device="cuda"):
        super().__init__()

        self.monster_size = monster_size
        self.goal_size = goal_size
        self.info_size = info_size
        self.out_size = 2
        self.device = device
        hidden_size = 100

        self.monster_encoder = nn.Sequential(
            nn.Linear(self.goal_size + self.info_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        ).to(device)

    def forward(self, obs):
        goal = F.one_hot(torch.tensor(obs["goal"]).long(), num_classes=self.goal_size).to(self.device)
        
        monster1_info = torch.tensor(np.array(obs["monster1_info"])).to(self.device)
        monster2_info = torch.tensor(np.array(obs["monster2_info"])).to(self.device)

        monster1 = self.monster_encoder(torch.cat((goal, monster1_info), dim=-1).float())
        monster2 = self.monster_encoder(torch.cat((goal, monster2_info), dim=-1).float())

        return torch.cat((monster1, monster2), dim=-1)
        

class QuestionAnswerQ(nn.Module):
    def __init__(self, goal_size, device="cuda"):
        super().__init__()

        self.goal_size = goal_size
        self.out_size = 2 + 2 * self.goal_size
        self.device = device
        hidden_size = 100

        self.monster_encoder = nn.Sequential(
            nn.Linear(self.goal_size * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.out_size)
        ).to(device)

    def forward(self, obs):
        goal = F.one_hot(torch.tensor(obs["goal"]).long(), num_classes=self.goal_size).to(self.device)
        
        monster1_qa = torch.tensor(np.array(obs["monster1_qa"])).to(self.device)
        monster2_qa = torch.tensor(np.array(obs["monster2_qa"])).to(self.device)

        return self.monster_encoder(torch.cat((goal, monster1_qa, monster2_qa), dim=-1).float())
