import torch
from wovenv import N, M
from wovenv.venv.snapshot import SnapShot, Action
from wovenv.venv.replay import Replay

def form_data(data: list[SnapShot]) -> torch.Tensor:
    res = torch.zeros(len(data), 5, N, M)
    for k in range(len(data)):
        for i in range(N):
            for j in range(M):
                res[k][data[k].table[i][j].value][i][j] = 1.
    return res

def form_index(a: Action):
    return (a.i * M + a.j) * 2 + int(a.change)

class Engine:
    def __init__(self, learning_rate):
        self.model = torch.nn.Sequential(
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
                    kernel_size=3),
                torch.nn.ReLU(),
                torch.nn.Flatten(),
                torch.nn.Linear(inp, 100),
                torch.nn.ReLU(),
                torch.nn.Linear(100, out))
        self.optim = torch.optim.Adam(
                self.model.parameters(),
                lr=learning_rate)

inp = N * M * 5
out = N * M * 2

class PolicyNet:

    def __init__(self, learning_rate=0.5, discount=1) -> None:
        self.engine = Engine(learning_rate)
        self.discount = discount
        
    def get_action(self, s: SnapShot) -> tuple[float, Action]:

        X = form_data([s])
        q_vals_pred = self.engine.model(X)[0]

        actions_list = s.get_legal_actions()

        if len(actions_list) == 0: return (0., None)

        opt_q_action = (q_vals_pred[form_index(actions_list[0])], actions_list[0])
        for a in actions_list:
            if opt_q_action[0] < q_vals_pred[form_index(a)]:
                opt_q_action = (q_vals_pred[form_index(a)], a)

        return opt_q_action
    
    def compute_td_loss(self, s: list[SnapShot], a: list[Action], ns: list[SnapShot], r: list[int], d: list[bool]) -> torch.Tensor:

        states = form_data(s)
        actions = torch.tensor(list(map(form_index, a)), dtype=torch.long)
        rewards = torch.tensor([[ri] for ri in r], dtype=torch.float32)

        pred_q_values = self.engine.model(states)
        pred_q_values_for_actions = pred_q_values[:, actions]

        next_q_values = torch.tensor([self.get_action(ss)[0] for ss in ns])

        new_q_values = rewards + self.discount * next_q_values
        loss = torch.mean((pred_q_values_for_actions - new_q_values.detach()) ** 2)

        return loss
    
    def update_batch(self, rep: Replay):
        s, a, ns, r, d = zip(*rep.rl)
        self.engine.optim.zero_grad()
        self.compute_td_loss(s, a, ns, r, d).backward()
        self.engine.optim.step()