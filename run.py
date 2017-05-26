import gym 
import random
import numpy as np
from statistics import mean , median
from collections import Counter
# import keras
from keras.models import Sequential , load_model
from keras.layers import Dense , Dropout 


model = load_model('model.h5')
model.load_weights('model_weights.h5')

LR = 1e-3

env = gym.make('CartPole-v0')
env.reset()
goal_steps = 900
score_requirement = 100
initial_games = 25000




scores = []
choices = []
for each_game in range(10):
    score = 0
    game_memory = []
    prev_obs = []
    env.reset()
    for _ in range(goal_steps):
        env.render()

        if len(prev_obs)==0:
            action = random.randrange(0,2)
        else:
            # action = np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0])
            action = np.argmax(model.predict(prev_obs.reshape(1,len(prev_obs))))
            # print 'prev_obs' , prev_obs.shape
            # action = np.argmax(model.predict(prev_obs.reshape(4,)))

        choices.append(action)     
        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        game_memory.append([new_observation, action])
        score+=reward
        if done: 
            print _
            break

    scores.append(score)

print('Average Score:',sum(scores)/len(scores))
print('choice 1:{}  choice 0:{}'.format(float(choices.count(1))/len(choices),float(choices.count(0))/len(choices)))
print(score_requirement)