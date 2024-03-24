import torch
from wovenv import N, M, SAMPLE_SIZE, MAX_TURN
from wovenv.venv.replay import PrioritizedExperienceReplay
from .dqn_model import DQN


class Engine:
    def __init__(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = DQN().to(device)
        self.optim = torch.optim.Adam(
                self.model.parameters())

class PolicyNet:

    def __init__(self, discount=1) -> None:
        self.engine   = Engine()
        self.discount = discount
        
    def get_action(self, s: torch.Tensor) -> int:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return self.engine.model.to(device).act(s)
    
    def compute_td_loss(self, states: torch.Tensor, action: torch.Tensor, next_states: torch.Tensor, reward: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
        q_values         = self.engine.model.forward(states)
        q_value          = q_values.gather(1, action.unsqueeze(1))

        next_q_values  = self.engine.model.forward(next_states)
        next_q_values[~next_states[:, -1].flatten(1).bool()] = -float('inf')

        next_q_value, _ = next_q_values.max(1)
        next_q_value[done] = 0
        
        expected_q_value = reward + self.discount * next_q_value
        return (q_value.squeeze(1) - expected_q_value.detach())
    
    def update_batch(self, rep: PrioritizedExperienceReplay):
        if rep.len() == 0: return torch.tensor([float('inf')])
        s, a, ns, r, d, index, weight = rep.sample(SAMPLE_SIZE)
        losses = self.compute_td_loss(s, a, ns, r, d)
        rep.update_priorities(index, losses.abs())
        self.engine.optim.zero_grad()
        loss = ((losses * weight.detach()) ** 2).mean()
        with torch.autograd.set_detect_anomaly(True):
            loss.backward()
        self.engine.optim.step()
        return (losses ** 2).mean().detach()

