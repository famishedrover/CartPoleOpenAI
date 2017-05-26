import gym 
import random
import numpy as np
from statistics import mean , median
from collections import Counter
# import keras
from keras.models import Sequential
from keras.layers import Dense , Dropout 


LR = 1e-3

env = gym.make('CartPole-v0')
env.reset()
goal_steps = 500
score_requirement = 50
initial_games = 10000

def some_random_games():
	for episode in range(5) :
		env.reset()
		for t in range(goal_steps) :
			env.render()
			action = env.action_space.sample()
			observation , reward , done , info = env.step(action)
			if done :
				break

# some_random_games()

def initial_population() :
	training_data = []
	scores = []
	accepted_scores = []
	for _ in range(initial_games) :
		score = 0 
		game_memory = []
		prev_observation = []

		for _ in range(goal_steps) :
			action = random.randrange(0,2)
			observation , reward , done , info = env.step(action)
			if (len(prev_observation) > 0) :
				game_memory.append([prev_observation , action])
			prev_observation = observation
			score += reward
			if done :
				break
		if score >= score_requirement :
			accepted_scores.append(score)
			for data in game_memory :
				if data[1] == 1 :
					output = [0,1]
				elif data[1] == 0 :
					output = [1,0]
				training_data.append([data[0],output])
		env.reset()
		scores.append(score)
	training_data_save = np.array(training_data)
	np.save('saved.npy',training_data_save)
	print 'Average accepted score:',mean(accepted_scores)
	print 'Median score for accepted scores:',median(accepted_scores) 
	print Counter(accepted_scores) 
	return training_data




x = initial_population()
X = np.array([i[0] for i in x])
print X[0].shape
print X.shape
# print x[0]
# print (x[0][0])
Y = np.array([i[1] for i in x])
print Y.shape

input_shape = len(X[0])
model = Sequential()
model.add(Dense(128 , input_shape = X[0].shape))
model.add(Dense(256 , activation = 'relu'))
model.add(Dense(512 , activation = 'relu'))
model.add(Dense(256 , activation = 'relu'))
model.add(Dense(128 , activation = 'relu'))
model.add(Dense(2, activation = 'softmax'))
model.summary()
model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'])
hist = model.fit(X , Y ,
			batch_size = 50 ,
			nb_epoch = 5 ,
			)

model.save('model.h5')
model.save_weights('model_weights.h5')

# initial_population()
