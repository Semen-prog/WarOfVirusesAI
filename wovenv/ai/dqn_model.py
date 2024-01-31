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
        )

        self.advantage = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(num_inputs, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 1)
        )

        self.value = torch.nn.Sequential(
            torch.nn.ReLU(),
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

    def act(self, _state: SnapShot, epsilon):
        state   = torch.autograd.Variable(_state.to_tensor().unsqueeze(0), volatile=True)
        q_value = self.forward(state)
        action  = form_action(q_value.max(1)[1].item())
        legal_actions = _state.get_legal_actions();
        if random.random() > epsilon or (action not in legal_actions):
            action = legal_actions[random.randrange(len(legal_actions))]
        return action