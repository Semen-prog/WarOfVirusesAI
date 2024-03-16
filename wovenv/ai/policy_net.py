import torch
from wovenv.venv.utils import write_error
from wovenv import N, M, SAMPLE_SIZE
from wovenv.venv.snapshot import SnapShot, Action
from wovenv.venv.replay import Replay
from .dqn_model import DQN

def form_data(data: list[SnapShot]) -> torch.Tensor:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    res = torch.stack([s.to_tensor().to(device) for s in data])
    return res

class Engine:
    def __init__(self, learning_rate):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = DQN(N * M, N * M).to(device)
        self.optim = torch.optim.Adam(
                self.model.parameters(),
                lr=learning_rate)

class PolicyNet:

    def __init__(self, learning_rate=0.5, discount=1) -> None:
        self.engine   = Engine(learning_rate)
        self.discount = discount
        
    def get_action(self, s: SnapShot) -> tuple[float, Action]:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return self.engine.model.to(device).act(s)
    
    def compute_td_loss(self, state: list[SnapShot], action: list[Action], next_state: list[SnapShot], reward: list[int], done: list[bool]) -> torch.Tensor:

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        state      = torch.autograd.Variable(form_data(state).to(device))
        action     = torch.autograd.Variable(torch.LongTensor(list(map(lambda a: a.to_index(), action))).to(device))
        reward     = torch.autograd.Variable(torch.FloatTensor(reward).to(device)).unsqueeze(1).to(device)
        done       = torch.autograd.Variable(torch.FloatTensor(done).to(device)).unsqueeze(1).to(device)

        q_values      = self.engine.model(state)

        q_value          = q_values.gather(1, action.unsqueeze(1))
        next_q_value     = torch.tensor([[self.get_action(ns)[0]] for ns in next_state]).to(device)
        expected_q_value = reward + self.discount * next_q_value * (1 - done)

        loss = (q_value - expected_q_value.detach()) ** 2

        return loss
    
    def update_batch(self, rep: Replay):
        if rep.len() == 0: return torch.tensor([float('inf')])
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        sample, idx = rep.sample(SAMPLE_SIZE)
        s, a, ns, r, d = zip(*sample)
        self.engine.optim.zero_grad()
        loss = self.compute_td_loss(s, a, ns, r, d).to(device)
        rep.update_priority(loss, idx)
        loss = loss.mean()
        loss.backward()
        self.engine.optim.step()
        return loss.detach().to(device)

