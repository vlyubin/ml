import gym
import numpy as np

env = gym.make("CartPole-v0")
#env = gym.wrappers.Monitor(env, 'CP', force=True)

class VovaEstimator:
    def __init__(self):
      self.weights = np.random.random(14)
      self.gamma = 0.9 

    def featurize(self, s, a):
      if a == 0:
        a = -1
      return np.array([s[0], s[1], s[2], s[3],
        abs(s[0]), s[1] * s[1], s[2] * s[2], s[3] * s[3],
        a, a * s[0], a * s[1], a * s[2], a * s[3], 1])

    def predict(self, s, a):
      return self.Q(s, a, self.weights)

    def update(self, data):
      y = []
      X = []
      for d in data:
        y.append(d[2] + max(self.Q(d[3], 0, self.weights), self.Q(d[3], 1, self.weights)) * self.gamma)
        X.append(self.featurize(d[0], d[1]))
      rv = np.linalg.lstsq(X, y)
      sol = rv[0]
      self.weights = sol

    def Q(self, s, a, wei):
      return wei.dot(self.featurize(s, a).T)

    
def epsilon_greedy_policy(estimator, epsilon, actions):
    """ Q is a numpy array, epsilon between 0,1 
    and a list of actions"""
    
    def policy_fn(state):
        if np.random.rand() > epsilon:
            action = np.argmax(np.array([estimator.predict(state, 0), estimator.predict(state, 1)]))
        else:
            action = np.random.choice(actions)
        return action
    return policy_fn


estimator = VovaEstimator()

n_episodes = 10000

actions = range(env.action_space.n)
data = []

for j in range(n_episodes):
    #env.render()
    done = False
    state = env.reset()

    if j % 10 == 0 and j > 0:
      data = data[-500:]
      batch = []
      for bindex in range(min(128, len(data))):
        batch.append(data[np.random.randint(len(data))])
      estimator.update(batch)

    policy = epsilon_greedy_policy(estimator, epsilon=0.1, actions = actions)
    total_score = 0

    ### Generate sample episode
    while not done:
      action = policy(state)
      new_state, reward, done, _ =  env.step(action)
      total_score += reward
      data.append((state, action, reward, new_state))
      state = new_state

      if done:
        print("\rEpisode {} / {}. Score: {}".format(
                j,n_episodes,total_score), end="")
        break

env.close()
#gym.upload('CP', api_key='sk_4jIycd3IT1SyJLj3d2mFxw')
