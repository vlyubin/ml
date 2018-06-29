import numpy as np
np.random.seed(0)

def f(w):
  center = np.array([0.5, 0.1, -0.3])
  return -np.sum(np.square(center - w))

def nes(npop, n_iter, sigma, alpha, f, w0):
  w = w0
  for i in range(n_iter):
    eps_sum = 0
    for j in range(npop):
      eps = np.random.normal(0, 1)
      eps_sum += f(w + eps * sigma) * eps
    eps_sum = eps_sum * alpha / npop / sigma
    w += eps_sum
  return w

print(nes(10, 300000, 1, 0.01, f, 1))
