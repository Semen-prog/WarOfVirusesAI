import torch
from wovenv import N, M
from wovenv.venv.snapshot import SnapShot, Action
from wovenv.venv.replay import Replay
from .dqn_model import DQN

def form_data(data: list[SnapShot]) -> torch.Tensor:
    res = torch.stack([s.to_tensor() for s in data])
    return res

class Engine:
    def __init__(self, learning_rate):
        self.model = DQN(N * M * 5, N * M * 2)
        self.optim = torch.optim.SGD(
                self.model.parameters(),
                lr=learning_rate)

class PolicyNet:

    def __init__(self, learning_rate=0.5, discount=1) -> None:
        self.engine   = Engine(learning_rate)
        self.discount = discount
        
    def get_action(self, s: SnapShot, eps: float) -> tuple[float, Action]:
        return self.engine.model.act(s, eps)
    
    def compute_td_loss(self, state: list[SnapShot], action: list[Action], next_state: list[SnapShot], reward: list[int], done: list[bool]) -> torch.Tensor:

        state      = torch.autograd.Variable(form_data(state))
        next_state = torch.autograd.Variable(form_data(next_state), volatile=True)
        action     = torch.autograd.Variable(torch.LongTensor(list(map(lambda a: a.to_index(), action))))
        reward     = torch.autograd.Variable(torch.FloatTensor(reward))
        done       = torch.autograd.Variable(torch.FloatTensor(done))

        q_values      = self.engine.model(state)
        next_q_values = self.engine.model(next_state)

        q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value     = next_q_values.max(1)[0]
        expected_q_value = reward + self.discount * next_q_value * (1 - done)

        loss = (q_value - torch.autograd.Variable(expected_q_value.data)).pow(2).mean()

        return loss
    
    def update_batch(self, rep: Replay):
        s, a, ns, r, d = zip(*rep.rl)
        self.engine.optim.zero_grad()
        loss = self.compute_td_loss(s, a, ns, r, d)
        loss.backward()
        self.engine.optim.step()
        return loss.detach()
