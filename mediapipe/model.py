import torch
import torch.nn as nn
# 定义 MLP 网络
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()

        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        self.hidden_layers = nn.Sequential(*layers)

        self.fc = nn.Linear(prev_size, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.hidden_layers(x)
        x = self.fc(x)
        return x