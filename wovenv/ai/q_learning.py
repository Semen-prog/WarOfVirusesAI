import torch, random
from wovenv.ai.policy_net import PolicyNet
from wovenv.venv.replay import PrioritizedExperienceReplay
from wovenv import MAX_TURN, BATCH_SIZE, N, M

class QLearningAgent():
  
  def __init__(self,epsilon=1,discount=1):
    
    self.network = PolicyNet(discount=discount)
    self.replay = PrioritizedExperienceReplay(capacity=BATCH_SIZE)
    self.epsilon = epsilon

  def _flip_coin(self, prob):
    return random.uniform(0, 1) < prob

  def get_action(self, state: torch.Tensor) -> int:
    if self._flip_coin(self.epsilon):
      return random.choice(torch.tensor(range(N * M))[state[-1].bool().flatten()]).item()
    return self.network.get_action(state)
  
  def update_batch(self):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return self.network.update_batch(self.replay).to(device).item()

  def add(self, s: torch.Tensor, a: int, ns: torch.Tensor, r: float, d: bool):
    self.replay.add_experience(s, a, ns, r, d)
