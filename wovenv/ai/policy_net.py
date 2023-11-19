import torch
import numpy as np
from wovenv.venv.state import N, M
from wovenv.venv.snapshot import SnapShot, Action

def form_data(data: list[SnapShot]) -> torch.Tensor:
    res = []
    for s in data:
        res.append(np.hstack([np.array([[float(s.table[i][j].value) for j in range(M)] for i in range(N)]).reshape(N * M), np.array([s.turn])]))
    return torch.tensor(np.vstack(res), dtype=torch.float32)

def form_index(a: Action):
    return (a.i * M + a.j) * 2 + int(a.change)

inp = N * M + 1
out = N * M * 2

class PolicyNet:

    def __init__(self, learning_rate=0.5, discount=1) -> None:
        self.model = torch.nn.Sequential(
            torch.nn.Linear(inp, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, out),
            torch.nn.ReLU())
        self.opt = torch.optim.SGD(
            self.model.parameters(),
            lr=learning_rate)
        self.discount = discount
        
    def get_action(self, s: SnapShot) -> tuple[float, Action]:

        X = form_data([s])
        q_vals_pred = self.model(X)[0]

        actions_list = s.get_legal_actions()

        if len(actions_list) == 0: return (0., None)

        opt_q_action = (q_vals_pred[form_index(actions_list[0])], actions_list[0])
        for a in actions_list:
            if opt_q_action[0] < q_vals_pred[form_index(a)]:
                opt_q_action = (q_vals_pred[form_index(a)], a)

        return opt_q_action
    
    def compute_td_loss(self, ss: list[SnapShot], acs: list[Action], nss: list[SnapShot], rs: list[int], ds: list[bool]) -> torch.Tensor:

        states = torch.tensor(form_data(ss), dtype=torch.float32)
        actions = torch.tensor([form_index(a) for a in acs], dtype=torch.long)
        rewards = torch.tensor([rs], dtype=torch.float32).T
        dones = torch.tensor([ds], dtype=torch.bool).T

        pred_q_values = self.model(states)
        pred_q_values_for_actions = pred_q_values[:, actions]

        next_q_values = torch.tensor([[self.get_action(s)[0]] for s in nss])

        new_q_values = rewards + self.discount * next_q_values
        loss = torch.mean((pred_q_values_for_actions - new_q_values.detach()) ** 2)

        return loss
    
    def update_batch(self, batch: list[tuple[SnapShot, Action, SnapShot, int, bool]]):

        ss, acs, nss, rs, ds = list(zip(*batch))
        self.opt.zero_grad()
        #print(self.get_action(batch[0][0]))
        self.compute_td_loss(ss, acs, nss, rs, ds).backward()
        self.opt.step()
        #print(self.get_action(batch[0][0]))