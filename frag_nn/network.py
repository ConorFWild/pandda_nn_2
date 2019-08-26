# Get imports
import torch.nn as nn
import torch.nn.functional as F


# Define Network

class FragmentNet(nn.module):
    """
    CNN for recognising interesting ligands
    """

    def __init__(self):
        # Instatiate Network layers

        super(FragmentNet, self).__init__()

        self.conv1 = nn.Conv3d(1, 10, kernel_size=3)
        self.drop1 = nn.Dropout3d()

        self.conv1 = nn.Conv3d(1, 10, kernel_size=3)
        self.drop1 = nn.Dropout3d()

        self.conv1 = nn.Conv3d(1, 10, kernel_size=3)
        self.drop1 = nn.Dropout3d()

        self.fc1 = nn.Linear()

        self.fc2 = nn.Linear()


def forward(self, x):

        x = F.relu(F.max_pool3d(self.drop1(self.conv1(x)), 2))

        x = F.relu(F.max_pool3d(self.drop1(self.conv1(x)), 2))

        x = F.relu(F.max_pool3d(self.drop1(self.conv1(x)), 2))

        x = x.view(-1, 320)

        x = F.relu(self.fc1(x))

        x = F.Dropout(x, training=self.training)

        x = F.relu(self.fc2(x))

        return F.log_softmax(x, dim=1)