"""
Project: Sonic the Hedgehog 2 NEAT AI Playback Neural Network
Created by: John McMeen
Purpose: Render neural network from pkl file, and create bk2 playback file
Notes:
  This will create a bk2 file from the neural network playback
  Render bk2 file to mp4 requires fmpeg: python -m retro.scripts.playback_movie winner.bk2

  This playback demonstrates that if you stopped the simulation manually for some condition
  other than "done" returned from env.step, you will need to do that as well in the neural network
  playback, otherwise the simulation will continue and the neural network will keep taking actions.
"""

import retro        # pip install gym-retro
import numpy as np  # pip install numpy
import cv2          # pip install opencv-python
import neat         # pip install neat-python
import pickle       # pip install cloudpickle
import os

current_path = os.path.dirname(__file__)

# create retro environment: game, state, scenario (defines rewards)
environment = retro.make('SonicTheHedgehog2-Genesis', 'AquaticRuinZone.Act1', scenario='xpos',  record='.')

# reset environment to initial state
observation = environment.reset()

# configuration for playback from pkl must be the same as execution
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')

# NEAT setup
population = neat.Population(config)
population.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
population.add_reporter(stats)

# Open pkl file, binary serialization of neural network
with open('winner.pkl', 'rb') as input_file:
    genome = pickle.load(input_file)

# shape/resolution of image created by emulator
inx, iny, inc = environment.observation_space.shape

# scale down observation
inx = int(inx/8)
iny = int(iny/8)

# create NEAT network
network = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

# create an alternative type of neural network with NEAT
# network = neat.nn.FeedForwardNetwork.create(genome, config)

# set up some variables to track fitness
current_max_fitness = 0
fitness = 0
counter = 0

# set up loop
finished = False
while not finished:
    # render the game
    environment.render()

    # resize and reshape the observation image
    observation = cv2.resize(observation, (inx, iny))
    observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
    observation = np.reshape(observation, (inx, iny))

    # create a single array from 2d pixel data
    img_array = np.ndarray.flatten(observation)

    # create actions from input
    actions = network.activate(img_array)

    # map relu activation output to 0 or 1
    actions = np.where(np.array(actions) <= 0.0, 0.0, 1.0).tolist()

    # increment the emulator state
    observation, reward, done, info = environment.step(actions)

    # update fitness with reward from environment
    fitness += reward

    # give it 250 steps without improvement to improve fitness or restart
    if fitness > current_max_fitness:
        current_max_fitness = fitness
        counter = 0
    else:
        counter += 1

    if done or counter == 250:
        finished = True

    print(current_max_fitness)

# reset here forces Gym Retro to create bk2
environment.reset()
# exit forces windows to close
exit()
