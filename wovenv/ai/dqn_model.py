import torch
import random
from wovenv.venv.snapshot import *

class DQN(torch.nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(DQN, self).__init__()

        self.feature = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=5,
                out_channels=5,
                padding=1,
                kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=5,
                out_channels=5,
                padding=1,
                kernel_size=3
            ),
            torch.nn.ReLU()
        )

        self.advantage = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(num_inputs, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 1)
        )

        self.value = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(num_inputs, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, num_actions)
        )

    def forward(self, x):
        x = self.feature(x)

        advantage = self.advantage(x)
        value = self.value(x)

        return advantage + value

    def act(self, _state: SnapShot):
        state   = torch.autograd.Variable(_state.to_tensor().unsqueeze(0), volatile=True)
        q_values = self.forward(state)
        legal_actions = _state.get_legal_actions()
        opt_val = (-10000000, None)
        for action in legal_actions:
            if q_values[0][action.to_index()] > opt_val[0]:
                opt_val = (q_values[0][action.to_index()], action)
        return opt_val