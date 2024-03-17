import torch
import random
from wovenv.venv.snapshot import *
from wovenv.venv.utils import write_error

class DQN(torch.nn.Module):
    def __init__(self, num_inputs, num_actions):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        super(DQN, self).__init__()
        
        self.feature = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=6,
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
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=5,
                out_channels=5,
                padding=1,
                kernel_size=3
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=5,
                out_channels=1,
                padding=1,
                kernel_size=3
            ),
            torch.nn.ReLU(),
            torch.nn.Flatten()
        ).to(device)

        self.advantage = torch.nn.Sequential(
            torch.nn.Linear(num_inputs, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 1)
        ).to(device)

        self.value = torch.nn.Sequential(
            torch.nn.Linear(num_inputs, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, num_actions)
        ).to(device)

    def forward(self, x, xp):
        x = torch.cat([self.feature(x), xp], dim=1)
        advantage = self.advantage(x)
        value = self.value(x)

        return advantage + value

    def act(self, _state: SnapShot):
        state, statep = form_states([_state])
        q_values = self(state, statep).detach()
        legal_actions = _state.get_legal_actions()
        opt_val = (-10000000, None)
        for action in legal_actions:
            if q_values[0][action.to_index()] > opt_val[0]:
                opt_val = (q_values[0][action.to_index()], action)
        return opt_val
