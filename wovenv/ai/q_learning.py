import numpy as np
import torch
from wovenv.ai.policy_net import PolicyNet
from wovenv.venv.snapshot import SnapShot, Action
from wovenv.venv.replay import Replay
from wovenv import MAX_TURN, BATCH_SIZE

class QLearningAgent():
  
  def __init__(self,alpha=0.5,epsilon=1,discount=1):
    
    self.network = PolicyNet(learning_rate=alpha, discount=discount)
    self.replay = Replay(size=BATCH_SIZE)
    self.epsilon = epsilon

  def _flip_coin(self, prob):
    return np.random.uniform() < prob

  def get_action(self, state: SnapShot):
    if self._flip_coin(self.epsilon):
      actions = state.get_legal_actions()
      return actions[np.random.randint(len(actions))]
    return self.network.get_action(state)[1]
  
  def update_batch(self):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return self.network.update_batch(self.replay).to(device).item()
  
  def set_lr(self, lr):
    self.network.engine.optim.param_groups[0]['lr'] = lr

  def add(self, s: SnapShot, a: Action, ns: SnapShot, r: int, d: bool):
    self.replay.add(s, a, ns, r, d)
