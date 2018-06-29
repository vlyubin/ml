import gym
import numpy as np
import time

env = gym.make("Pong-v0")
#env = gym.wrappers.Monitor(env, 'CP', force=True, video_callable=False)

# Actions:
# 0 : "NOOP",
# 2 : "UP",
# 5 : "DOWN",

# Colors: {(92, 186, 92), (213, 130, 74), (0, 0, 0), (144, 72, 17), (236, 236, 236)}

def parse_image(state):
  # Return (his upper left, his bottom right, my upper left, my bottom right, ball upper left,
  # ball bottom right)

  # Crop the image between the lines :)
  #state = state[34:194]

  his_up_left = (-1, -1)
  his_bot_right = (-1, -1)

  my_up_left = (-1, -1)
  my_bot_right = (-1, -1)

  ball_up_left = (-1, -1)
  ball_bot_right = (-1, -1)

  # 109, 118, 43 is clay
  my_color = [53, 95, 24]
  his_color = [101, 111, 228]
  ball_color = [236, 236, 236]

  un = set()

  for i in range(len(state)):
    for j in range(len(state[0])):
      un.add(tuple(state[i][j].tolist()))
      if state[i][j].tolist() == my_color:
        my_bot_right = (i, j)
        if my_up_left == (-1, -1):
          my_up_left = (i, j)
      if state[i][j].tolist() == his_color:
        his_bot_right = (i, j)
        if his_up_left == (-1, -1):
          his_up_left = (i, j)
      if state[i][j].tolist() == ball_color:
        ball_bot_right = (i, j)
        if ball_up_left == (-1, -1):
          ball_up_left = (i, j)

  print(un)
  rv = (his_up_left, his_bot_right, my_up_left, my_bot_right, ball_up_left, ball_bot_right)
  print(rv)
  return rv
state = env.reset()

def featurize(parsed, prev_ball_ul, prev_ball_br):
  return np.concatenate([np.array(parsed).flatten(), np.array(prev_ball_ul), np.array(prev_ball_br)])

def decide(parsed, prev_ball_ul, prev_ball_br, theta):
  if theta.T.dot(state) <= 0:
    return 1
  return 0

# Returns the score for the given theta
def F(theta):
  score = 0
  done = False
  state = env.reset()
  prev_ball_ul = (-1, -1)
  prev_ball_br = (-1, -1)

  while True:
    parsed = parse_image(state)
    print(parsed)
    if parsed[5] == (-1, -1):
      action = 0
    else:
      action = decide(parsed, prev_ball_ul, prev_ball_br, theta)
    env.render()
    new_state, reward, done, _ =  env.step(action)
    if parsed[5] == (-1, -1):
      continue
    score += 0.01 # For playing longer
    score += reward
    if done:
      if np.random.random() < 0.01:
        print(score)
      return score
    prev_ball_ul = parsed[4]
    prev_ball_br = parsed[5]
    state = new_state

num_iter = 2000
num_samp = 20
len_theta = 4
sigma = 0.1
alpha = 0.01

theta = np.zeros(len_theta)
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
    eps_sum += (vals[j] - val_mean) * epses[j]
  eps_sum = eps_sum * alpha / num_samp / sigma
  theta += eps_sum

env.close()
#gym.upload('CP', api_key='sk_4jIycd3IT1SyJLj3d2mFxw')
