import numpy as np
import gym

class RandomEstimator:
    def __init__(self):
        np.random.seed(1)
        self.w = np.ones((16,4))
        self.alpha = 0.01
        
    def featurize(self, s):
        x = np.zeros(16)
        x[s] = 1
        return x

    def predict(self,s):
        return np.matmul(self.featurize(s),self.w)
    
    def update(self,s,a, td_target):
        error = td_target-self.predict(s)[a]
        self.w[s,a] += self.alpha*error
        
    
def epsilon_greedy_policy(estimator, epsilon, actions):
    """ Q is a numpy array, epsilon between 0,1 
    and a list of actions"""
    
    def policy_fn(state):
        if np.random.rand()>epsilon:
            action = np.argmax(estimator.predict(state))
        else:
            action = np.random.choice(actions)
        return action
    return policy_fn


estimator = RandomEstimator()

env = gym.make("FrozenLake-v0")
env = gym.wrappers.Monitor(env,'sarsa', force=True)


gamma = 0.9 

n_episodes = 10000


actions = range(env.action_space.n)
score = []    
for j in range(n_episodes):
    done = False
    state = env.reset()
    policy = epsilon_greedy_policy(estimator,epsilon=1./(j+1), actions = actions )
    
    
    ### Generate sample episode
    while not done:
        
        action = policy(state)
        new_state, reward, done, _ =  env.step(action)
        new_action = policy(new_state)
        
        # Oleg trick
        if reward < 1 and done:
            reward = -1
               
        #Calculate the td_target
        if done:
            td_target = reward
        else:
            new_q_val = estimator.predict(new_state)[new_action]
            td_target = reward + gamma * new_q_val
        
        estimator.update(state,action, td_target)    
        state, action = new_state, new_action
            
        if done:
            if len(score) < 100:
                score.append(reward)
            else:
                score[j % 100] = reward
            print("\rEpisode {} / {}. Avg score: {}".format(
                    j,n_episodes,np.mean(score)), end="")
            
            break

    
env.close()
gym.upload('sarsa', api_key='sk_4jIycd3IT1SyJLj3d2mFxw')
