"""
Project: GymRetro + NEAT
Purpose: A generic parallel loader for GymRetro with NEAT
Input files: 
    config-retro - Simulation configuration file
    config-neat - NEAT configuration file
Notes:
  This will create a bk2 file from the neural network playback
  Render bk2 file to mp4 requires fmpeg: python -m retro.scripts.playback_movie winner.bk2

  This playback demonstrates that if you stopped the simulation manually for some condition
  other than "done" returned from env.step, you will need to do that as well in the neural network
  playback, otherwise the simulation will continue and the neural network will keep taking actions.
"""

import retro
import numpy as np
import cv2
import neat
import pickle
import os
import configparser

#simulation configuration file
config_file = 'config-sonic'
pkl_file = 'SonicTheHedgehog-Genesis-GreenHillZone.Act1-contest-1.pkl'

# create path to current directory
current_path = os.path.dirname(__file__)

config = configparser.ConfigParser()
config.read(os.path.join(current_path, config_file))

game = config['retro']['game']
state = config['retro']['state']
scenario = config['retro']['scenario']
num_generations = int(config['simulation']['num_generations'])
network_type = config['neat']['network_type']
steps_to_kill = config.getint('simulation', 'steps_to_kill')

# create retro environment: game, state, scenario (defines rewards)
environment = retro.make(game=game, state=state, record='.')

# reset environment to initial state
observation = environment.reset()

# configuration for playback from pkl must be the same as execution
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        os.path.join(current_path, 'config-neat'))


# NEAT setup
population = neat.Population(config)
population.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
population.add_reporter(stats)

# Open pkl file, binary serialization of neural network
with open(pkl_file, 'rb') as input_file:
    genome = pickle.load(input_file)

# shape/resolution of image created by emulator
inx, iny, inc = environment.observation_space.shape

# scale down observation
inx = int(inx/8)
iny = int(iny/8)

if network_type == "recurrent":
    network = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
elif network_type == "feedforward":
    network = neat.nn.FeedForwardNetwork.create(genome, config)
else:
    raise ValueError("network_type must be 'recurrent' or 'feedforward'")

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

    # give it x steps without improvement to improve fitness or restart
    if fitness > current_max_fitness:
        current_max_fitness = fitness
        counter = 0
    else:
        counter += 1

    if done or counter == steps_to_kill:
        finished = True


# reset here forces Gym Retro to create bk2
environment.reset()
# exit forces windows to close
exit()
