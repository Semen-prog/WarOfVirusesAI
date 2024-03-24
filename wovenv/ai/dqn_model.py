import torch

class DQN(torch.nn.Module):
    def __init__(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        super(DQN, self).__init__()
        
        self.feature = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=7,
                out_channels=7,
                padding=1,
                kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=7,
                out_channels=7,
                padding=1,
                kernel_size=3
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=7,
                out_channels=1,
                padding=1,
                kernel_size=3
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=1,
                padding=1,
                kernel_size=3
            ),
            torch.nn.ReLU(),
            torch.nn.Flatten()
        ).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return self.feature(x.to(device))

    def act(self, state: torch.Tensor) -> tuple[float, int]:
        state = state.unsqueeze(0)
        q_values = self.forward(state)
        q_values[~state[:, -1].flatten(1).bool()] = -float('inf')
        return q_values[0].detach().argmax().item()
