import numpy as np
from wovenv.ai.policy_net import PolicyNet
from wovenv.venv.snapshot import SnapShot, Action
from wovenv.venv.replay import Replay
from wovenv import MAX_TURN, BATCH_SIZE

class QLearningAgent():
  
  def __init__(self,alpha=0.5,epsilon=1,discount=1):
    
    self.networks = [PolicyNet(learning_rate=alpha, discount=discount) for _ in range(MAX_TURN)]
    self.replays = [Replay(size=BATCH_SIZE) for _ in range(MAX_TURN)]
    self.epsilon = epsilon

  def _flip_coin(self, prob):
    return np.random.uniform() < prob

  def get_action(self, state: SnapShot):
    if self._flip_coin(self.epsilon):
      actions = state.get_legal_actions()
      return actions[np.random.randint(len(actions))]
    return self.networks[state.turn - 1].get_action(state)[1]
  
  def update_batch(self):
    sm_loss = 0
    for i in range(MAX_TURN):
      sm_loss += self.networks[i].update_batch(self.replays[i]).item()
    return sm_loss / 3

  def update(self, s: SnapShot, a: Action, ns: SnapShot, r: int, d: bool):
    self.replays[s.turn - 1].add(s, a, ns, r, d)