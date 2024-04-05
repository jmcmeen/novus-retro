"""
Project: Playing Sonic the Hedgehog with Gym Retro and NEAT
Purpose: Evolve a neural network to play Sonic
Created by: John McMeen
Adapted from:
  Code: https://gitlab.com/lucasrthompson/Sonic-Bot-In-OpenAI-and-NEAT
  Video: https://www.youtube.com/playlist?list=PLTWFMbPFsvz3CeozHfeuJIXWAJMkPtAdS
Notes:
  You will need to import you own ROMs: python -m retro.import roms/sega_classics
"""

import retro  # pip install gym-retro
import numpy as np  # pip install numpy
import cv2  # pip install opencv-python
import neat  # pip install neat-python
import pickle  # pip install cloudpickle
import os
import configparser

current_path = os.path.dirname(__file__)

config = configparser.ConfigParser()
config.read(os.path.join(current_path, 'config-retro'))

game = config['retro']['game']
state = config['retro']['state']
scenario = config['retro']['scenario']

# create retro environment: game, state, scenario (defines rewards)
environment = retro.make(game=game, state=state, scenario=scenario)

# define an evaluation function for each game "play through" genome
def eval_genomes(genomes, config):
    # for each genome in the population
    for genome_id, genome in genomes:
        # reset environment to initial state
        observation = environment.reset()

        # shape/resolution of image created by emulator
        inx, iny, inc = environment.observation_space.shape

        # scale down observation
        inx = int(inx / 8)
        iny = int(iny / 8)

        # create NEAT network
        network = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

        # create an alternative type of neural network with NEAT - requires config-neat file update
        # network = neat.nn.FeedForwardNetwork.create(genome, config)

        # set up some variables to track fitness
        current_max_fitness = 0
        fitness = 0
        counter = 0

        # optionally create another window for the "neural network's vision"
        # cv2.namedWindow("main", cv2.WINDOW_NORMAL) #-------------------- computer vision optional

        finished = False
        while not finished:
            # render the game
            environment.render()

            # resize and reshape the observation image
            observation = cv2.resize(observation, (inx, iny))
            #observation = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB) #color
            observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
            observation = np.reshape(observation, (inx, iny))
            # print(observation)

            # optional update "neural network's vision" or observation created from
            # cv2.imshow('main', observation)  #-------------------- computer vision optional
            # cv2.waitKey(1)  #-------------------- computer vision optional

            # create a single array from 2d pixel data
            observation = np.ndarray.flatten(observation)

            # create controller actions from input
            actions = network.activate(observation)

            # take a peek at controller actions before translation
            # print(actions)

            # map relu activation output to 0 or 1
            actions = np.where(np.array(actions) <= 0.0, 0.0, 1.0).tolist()

            # take a peek at controller actions before translation
            # print(actions)

            # increment the emulator state
            observation, reward, done, info = environment.step(actions)

            # update fitness with reward from environment
            fitness += reward

            # give it X steps without improvement to improve fitness or restart
            if fitness > current_max_fitness:
                current_max_fitness = fitness
                counter = 0
            else:
                counter += 1

            if done or counter == 1000:
                finished = True
                print(genome_id, current_max_fitness)

        # set the fitness for this genome
        genome.fitness = current_max_fitness


# NEAT configuration, all defaults except a config file is provided
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     os.path.join(current_path, 'config-feedforward'))

# NEAT output
population = neat.Population(config)
# population = neat.Checkpointer.restore_checkpoint('neat-checkpoint-12') #code to reload from checkpoint
population.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
population.add_reporter(stats)
population.add_reporter(neat.Checkpointer(generation_interval=1))

# the winning network, run for x generations
winner = population.run(eval_genomes, 500)

# save the winning network to a binary file to reload later
with open('winner-act1.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)

exit()
