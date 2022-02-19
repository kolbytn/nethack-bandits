import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, T5Tokenizer, T5ForConditionalGeneration


class BERTQ(nn.Module):
    def __init__(self, monster_size, goal_size, device="cuda"):
        super().__init__()

        self.monster_size = monster_size
        self.goal_size = goal_size
        self.device = device
        hidden_size = 100

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased', return_dict=True).to(device)
        self.bert.requires_grad = False

        self.monster_encoder = nn.Sequential(
            nn.Linear(self.goal_size + self.bert.config.hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        ).to(device)

    def forward(self, obs):
        goal = F.one_hot(torch.tensor(obs["goal"]).long(), num_classes=self.goal_size).to(self.device)
        
        monster1_wiki = self.tokenizer(obs["monster1_wiki"], return_tensors="pt", padding=True, truncation=True).to(self.device)
        monster1_wiki = self.bert(**monster1_wiki).last_hidden_state[:, 0]
        
        monster1_name = self.tokenizer(obs["monster1_name"], return_tensors="pt", padding=True, truncation=True).to(self.device)
        monster1_name = self.bert(**monster1_name).last_hidden_state[:, 0]
        
        monster2_wiki = self.tokenizer(obs["monster2_wiki"], return_tensors="pt", padding=True, truncation=True).to(self.device)
        monster2_wiki = self.bert(**monster2_wiki).last_hidden_state[:, 0]
        
        monster2_name = self.tokenizer(obs["monster2_name"], return_tensors="pt", padding=True, truncation=True).to(self.device)
        monster2_name = self.bert(**monster2_name).last_hidden_state[:, 0]

        monster1 = self.monster_encoder(torch.cat((goal, monster1_name, monster1_wiki), dim=-1))
        monster2 = self.monster_encoder(torch.cat((goal, monster2_name, monster2_wiki), dim=-1))

        return torch.cat((monster1, monster2), dim=-1)
