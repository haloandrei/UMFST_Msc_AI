import torch
from torch import nn
from torch.nn import functional as F


class SimpleCNN(nn.Module):
    def __init__(
        self,
        num_classes,
        embedding_dim,
        dropout,
        activation,
        image_size,
    ):
        super().__init__()
        self.activation = activation

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(p=dropout)
        feat_size = self._feature_size(image_size, pools=4)
        self.fc_embed = nn.Linear(256 * feat_size * feat_size, embedding_dim)
        self.fc_out = nn.Linear(embedding_dim, num_classes)

    def set_dropout(self, p):
        self.dropout.p = p

    def set_activation(self, activation):
        self.activation = activation

    def _act(self, x):
        if self.activation == "leaky_relu":
            return F.leaky_relu(x, negative_slope=0.1)
        return F.relu(x)

    def forward(self, x):
        x = self.pool(self._act(self.bn1(self.conv1(x))))
        x = self.pool(self._act(self.bn2(self.conv2(x))))
        x = self.pool(self._act(self.bn3(self.conv3(x))))
        x = self.pool(self._act(self.bn4(self.conv4(x))))

        x = x.view(x.size(0), -1)
        embed = self.fc_embed(self.dropout(x))
        logits = self.fc_out(self._act(embed))
        return logits, embed

    def conv1_features(self, x):
        return self.bn1(self.conv1(x))

    @staticmethod
    def _feature_size(image_size, pools):
        size = image_size
        for _ in range(pools):
            size //= 2
        return size
