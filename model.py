from torch import nn

class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 64, 3), # [N, 64, 26]
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, 3), # [N, 32, 24]
            nn.LeakyReLU(),
            nn.Conv2d(32, 16, 3), # [N, 16, 22]
            nn.LeakyReLU(),
            nn.Conv2d(16, 8, 3), # [N, 8, 20]
            nn.LeakyReLU()
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 20 * 20, 128),
            nn.Dropout(),
            nn.Linear(128, 10)
        )
        
    def forward(self, x):
        return self.classifier(self.backbone(x))

''' from torch import nn
import torch.nn.functional as F
import torch

class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output '''
