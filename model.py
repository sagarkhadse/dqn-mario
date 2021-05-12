import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, n_inputs, n_actions):
        super(DQN, self).__init__()

        # Convolutional Layers =====================================================================
        self.conv1 = nn.Conv2d(in_channels=n_inputs, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        # Fully Connected Layers ===================================================================
        self.fc1 = nn.Linear(in_features=7*7*64, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=n_actions)
        # Activation ===============================================================================
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x