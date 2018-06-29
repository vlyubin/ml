# NES that uses "featurized" version of state
#
# Example launches:
# https://gym.openai.com/evaluations/eval_G2PSMkFmQI6So4LpAtKhpg
# https://gym.openai.com/evaluations/eval_2WgXC1grRUO7xyleREoXRw

import gym
import numpy as np

env = gym.make("CartPole-v0")
env = gym.wrappers.Monitor(env, 'CP', force=True, video_callable=False)

def featurize(s):
  # Get some nonlinear features
  return np.array([s[0], s[1], s[2], s[3],
        s[0] * s[0], s[1] * s[1], s[2] * s[2], s[3] * s[3],
        s[0] * s[1], s[0] * s[2], s[0] * s[3], s[1] * s[2], s[1] * s[3],
        s[2] * s[3], 1])

def decide(state, theta):
  if theta.T.dot(featurize(state)) <= 0:
    return 1
  return 0

# Returns the score for the given theta
def F(theta):
  score = 0
  done = False
  state = env.reset()

  while True:
    action = decide(state, theta)
    new_state, reward, done, _ =  env.step(action)
    if done:
      return score
    score += reward
    state = new_state

num_iter = 2000
num_samp = 20
len_theta = 15
sigma = 0.1
alpha = 0.01

# Find a good starting theta - using NES for score around 20 takes
# ages to converge
theta = 2 * np.random.normal(0, 1, len_theta)
bst_score = F(theta)

for _ in range(100):
  new_theta = 2 * np.random.normal(0, 1, len_theta)
  new_score = F(new_theta)
  if new_score > bst_score:
    bst_score = new_score
    theta = new_theta

solved = False

for i in range(num_iter):
  eps_sum = np.zeros(len_theta)
  epses = []
  vals = []
  for j in range(num_samp):
      # Generate a random vector
      eps = np.random.normal(0, sigma, size=len_theta)
      epses.append(eps)
      vals.append(F(theta + eps * sigma))

  # This is extremely important, as without this line multiplier for each
  # results is very similar. So we just subtract means.
  val_mean = np.mean(vals)
  for j in range(num_samp):
    eps_sum += (F(theta + epses[j] * sigma) - val_mean) * epses[j]
  eps_sum = eps_sum * alpha / num_samp / sigma
  theta += eps_sum

  # Check theta performance, and if it's good, break
  cur_perf = 0
  for _ in range(10):
    cur_perf += F(theta)
  if i % 10 == 0:
    print(cur_perf / 10.0)
  if cur_perf >= 10 * 199:
    print('Solved!')
    solved = True
    break

# Just do 300 perfect iterations to "solve" it in openAI
if solved:
  for _ in range(300):
    F(theta)

env.close()
gym.upload('CP', api_key='sk_4jIycd3IT1SyJLj3d2mFxw')
