import torch
import numpy as np
from wovenv.venv.state import N, M, MAX_TURN
from wovenv.venv.snapshot import SnapShot, Action

def form_data(data: SnapShot) -> torch.Tensor:
    res = torch.zeros(1, 5, N, M)
    for i in range(N):
        for j in range(M):
            res[0][data.table[i][j].value][i][j] = 1.
    return res

def form_index(a: Action):
    return (a.i * M + a.j) * 2 + int(a.change)

inp = N * M * 5
out = N * M * 2

class PolicyNet:

    def __init__(self, learning_rate=0.5, discount=1) -> None:

        self.models = [[None, None] for _ in range(MAX_TURN)]
        for i in range(MAX_TURN):
            self.models[i][0] = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=5,
                    out_channels=5,
                    padding=1,
                    kernel_size=3),
                torch.nn.ReLU(),
                torch.nn.Flatten(),
                torch.nn.Linear(inp, 100),
                torch.nn.ReLU(),
                torch.nn.Linear(100, out))
            self.models[i][1] = torch.optim.Adam(
                self.models[i][0].parameters(),
                lr=learning_rate)

        self.discount = discount
        
    def get_action(self, s: SnapShot) -> tuple[float, Action]:

        X = form_data(s)
        q_vals_pred = self.models[s.turn - 1][0](X)[0]

        actions_list = s.get_legal_actions()

        if len(actions_list) == 0: return (0., None)

        opt_q_action = (q_vals_pred[form_index(actions_list[0])], actions_list[0])
        for a in actions_list:
            if opt_q_action[0] < q_vals_pred[form_index(a)]:
                opt_q_action = (q_vals_pred[form_index(a)], a)

        return opt_q_action
    
    def compute_td_loss(self, s: SnapShot, a: Action, ns: SnapShot, r: int, d: bool) -> torch.Tensor:

        states = form_data(s)
        actions = torch.tensor([form_index(a)], dtype=torch.long)
        rewards = torch.tensor([[r]], dtype=torch.float32)

        pred_q_values = self.models[s.turn - 1][0](states)
        pred_q_values_for_actions = pred_q_values[:, actions]

        next_q_values = torch.tensor([self.get_action(ns)[0]])

        new_q_values = rewards + self.discount * next_q_values
        loss = torch.mean((pred_q_values_for_actions - new_q_values.detach()) ** 2)

        return loss
    
    def update_batch(self, s: SnapShot, a: Action, ns: SnapShot, r: int, d: bool):

        self.models[s.turn - 1][1].zero_grad()
        #print(self.get_action(batch[0][0]))
        self.compute_td_loss(s, a, ns, r, d).backward()
        self.models[s.turn - 1][1].step()
        #print(self.get_action(batch[0][0]))