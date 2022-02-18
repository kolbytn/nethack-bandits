import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel


class BERTQ(nn.Module):
    def __init__(self, goal_size, device="cuda"):
        super().__init__()

        self.goal_size = goal_size
        self.device = device

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased', return_dict=True).to(device)
        self.bert.requires_grad = False

        self.q = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size * 2 + self.goal_size, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        ).to(device)

    def forward(self, obs):
        goal = F.one_hot(torch.tensor(obs["goal"]).long(), num_classes=self.goal_size).to(self.device)
        
        monster1 = self.tokenizer(obs["monster1"], return_tensors="pt", padding=True, truncation=True).to(self.device)
        monster1 = self.bert(**monster1).last_hidden_state[:, 0]
        
        monster2 = self.tokenizer(obs["monster2"], return_tensors="pt", padding=True, truncation=True).to(self.device)
        monster2 = self.bert(**monster2).last_hidden_state[:, 0]

        x = torch.cat((goal, monster1, monster2), dim=-1)

        return self.q(x)
        