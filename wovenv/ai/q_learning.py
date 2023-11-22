import numpy as np
from wovenv.ai.policy_net import PolicyNet
from wovenv.venv.snapshot import SnapShot, Action

class QLearningAgent():
  
  def __init__(self,alpha=0.5,epsilon=1,discount=1):
    
    self.network = PolicyNet(learning_rate=alpha, discount=discount)
    self.epsilon = epsilon

  def _flip_coin(self, prob):
    return np.random.uniform() < prob

  def get_action(self, state: SnapShot):
    
    if self._flip_coin(self.epsilon):
      return np.random.choice(state.get_legal_actions())
    return self.network.get_action(state)[1]

  def update(self, s, a, ns, r, d):
    
    self.network.update_batch(s, a, ns, r, d)