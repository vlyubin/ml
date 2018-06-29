# This variant of solution uses minibatch, where at each time we use the last
# 10000 data points (random works just as well).
# Once we consistently solve the problem, updates are stopped.
# This is not necessary, and other approaches
# solve the problem as well, but with random minibatch your solution is jiggling
# between 100 and 200 (typically around 180).
#
# Example launches:
# https://gym.openai.com/evaluations/eval_s0aRbyjNQSieBMUcMtZNw
# https://gym.openai.com/evaluations/eval_j6s9BMASRF6ov8nhu9MRlQ

import gym
import numpy as np

env = gym.make("CartPole-v0")
env = gym.wrappers.Monitor(env, 'CP', force=True)

class VovaEstimator:
    def __init__(self):
      self.weights = np.ones(14)
      self.gamma = 0.9 

    def featurize(self, s, a):
      # This is a fairly important hack - with a = 0, all terms that are multiplied by
      # a become 0! Otherwise, we get some meaningful results from it.
      if a == 0:
        a = -1
      # This array contains combined information from states and action
      return np.array([s[0], s[1], s[2], s[3],
        abs(s[0]), s[1] * s[1], s[2] * s[2], s[3] * s[3],
        a, a * s[0], a * s[1], a * s[2], a * s[3], 1])

    def predict(self, s, a):
      return self.Q(s, a, self.weights)

    def update(self, data):
      # np.linalg.lstsq solves the equation w * X = Y for w so that squared error is minimized,
      # which is exactly what we need.
      y = []
      X = []
      for d in data:
        y.append(d[2] + max(self.Q(d[3], 0, self.weights), self.Q(d[3], 1, self.weights)) * self.gamma)
        X.append(self.featurize(d[0], d[1]))
      rv = np.linalg.lstsq(X, y)
      sol = rv[0]
      self.weights = sol

    def Q(self, s, a, wei):
      # Basically, Q(s, a) tells us how good is it to take action a in state s
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

n_episodes = 1000
update_frequency = 10

actions = range(env.action_space.n)
data = []
batch_score = 0
solved = False

for j in range(n_episodes):
    #env.render()
    done = False
    state = env.reset()

    # Once we solved the problem for the entire batch, stop the updateds (we converged)
    if batch_score >= update_frequency * 199:
      solved = True

    if j % update_frequency == 0 and j > 0 and not solved:
      estimator.update(data)
      batch_score = 0

    policy = epsilon_greedy_policy(estimator, epsilon=0.1, actions = actions)
    total_score = 0

    ### Generate sample episode
    while not done:
      action = policy(state)
      new_state, reward, done, _ =  env.step(action)
      if done:
        reward = 0
      total_score += reward
      data.append((state, action, reward, new_state))
      state = new_state

      if done:
        print("\rEpisode {} / {}. Score: {}".format(
                j,n_episodes,total_score), end="")

    batch_score += total_score

env.close()
gym.upload('CP', api_key='sk_4jIycd3IT1SyJLj3d2mFxw')
