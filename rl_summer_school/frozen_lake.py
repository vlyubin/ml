import gym
import numpy as np

env = gym.make("FrozenLake-v0")
env = gym.wrappers.Monitor(env, 'first_visit', force=True)

def eps_greedy_policy(Q, eps, actions):

  def policy_f(state):
    if np.random.rand() > eps:
      action = np.argmax(Q[state,:])
    else:
      action = np.random.choice(actions)
    return action

  return policy_f

Q = np.random.rand(env.observation_space.n, env.action_space.n)

actions = range(env.action_space.n)

n_episodes = 4000
alpha = 0.4
gamma = 0.9

for episode_idx in range(n_episodes):
  s = env.reset()
  done = False
  policy = eps_greedy_policy(Q, max(1 - episode_idx * 0.01, 0), actions)

  while True:
    a = policy(s)
    s_star, reward, done, _ = env.step(a)
    if done:
      Q[s, a] += alpha * (reward - Q[s, a])
      break
    Q[s, a] += alpha * (reward + gamma * np.max(Q[s_star, :]) - Q[s, a])
    s = s_star

env.close()
gym.upload('first_visit', api_key='sk_4jIycd3IT1SyJLj3d2mFxw')
