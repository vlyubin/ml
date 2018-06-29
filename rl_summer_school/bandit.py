import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class BernoulliArm():

  def __init__(self, p):
    self.p = p

  def step(self):
    if np.random.random() > self.p:
      return 1
    else:
      return 0

class EpsilonGreedyBandit():

  def __init__(self, epsilon, k):
    self.epsilon = epsilon
    self.counts = [0] * k
    self.exp = [0.0] * k
    self.k = k

  def choose_action(self):
    if np.random.random() > self.epsilon:
      return np.random.randint(self.k)
    else:
      return np.argmax(self.exp)

  def update(self, idx, reward):
    self.exp[idx] = (self.exp[idx] * self.counts[idx] + reward) / (self.counts[idx] + 1)
    self.counts[idx] += 1

  def reset(self):
    self.counts = [0] * self.k
    self.exp = [0.0] * self.k

def test_bandit(bandit, arms, n_episodes, n_steps, *args, **kwargs):    
    chosen_arms = np.zeros((n_episodes, n_steps))
    avg_reward = np.zeros(n_steps)
    cumulative_reward = np.zeros(n_steps)
    for episode in range(n_episodes):
        ep_reward = 0
        bandit.reset()
        for step in range(n_steps):
            chosen_arm = bandit.choose_action()
            chosen_arms[episode,step] = chosen_arm
            reward = arms[chosen_arm].step()
            ep_reward += reward
            avg_reward[step] += reward
            bandit.update(chosen_arm, reward)
        if episode % 1 == 0:            
            print("Reward in episode {}: {}".format(episode,ep_reward))
    print("Test 1: How often does our algorithm pick the best action?")
    
    #sns.heatmap(chosen_arms)        
    print("Test 2: TODO - Track the average reward over time")
    cumulative_reward[0] = avg_reward[0]
    for i in range(1, n_steps):
      cumulative_reward[i] += cumulative_reward[i - 1] + avg_reward[i]
    avg_reward /= n_episodes
    cumulative_reward /= n_episodes
    print("Test 3: TODO - Track cumulative reward")
    return chosen_arms, avg_reward, cumulative_reward
    
    
    test_bandit(bandit = EpsilonGreedyBandit(0.1,2), arms = [BernoulliArm(p=0.1), 
                BernoulliArm(p=0.9)], n_episodes=10, n_steps=50)

