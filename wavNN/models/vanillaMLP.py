import torch
import torch.nn as nn


class VanillaMLP(nn.Module):
    def __init__(self, in_channels, hidden_size, out_channels, tail=False):
        super().__init__()

        self.flatten_input = nn.Flatten()
        self.hidden_layer = nn.Linear(in_channels**2, hidden_size)
        self.output_layer = nn.Linear(hidden_size, out_channels)

        if tail:
            self.tail = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.flatten_input(x)
        x = self.hidden_layer(x)
        x = self.output_layer(x)

        if hasattr(self, "tail"):
            x = self.tail(x)

        return x


class BananaSplitMLP(nn.Module):
    def __init__(self, in_channels, hidden_size, out_channels, tail=False):
        super().__init__()

        self.flatten_input = nn.Flatten()

        self.hidden_layer_1 = nn.Linear(in_channels**2, hidden_size)
        self.hidden_layer_2 = nn.Linear(in_channels**2, hidden_size)
        self.hidden_layer_3 = nn.Linear(in_channels**2, hidden_size)

        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

        # Output of those tied 3 channel layers, and the flattened concat of those
        self.output = nn.Linear(hidden_size * 3, out_channels)

        if tail:
            self.tail = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.flatten_input(x)

        channel_1 = self.hidden_layer_1(x)
        channel_2 = self.hidden_layer_2(x)
        channel_3 = self.hidden_layer_3(x)

        concat = torch.stack([channel_1, channel_2, channel_3], dim=1)
        # Flatten for the output dense

        x = self.flatten(concat)
        x = self.output(x)

        if hasattr(self, "tail"):
            x = self.tail(x)

        return x
