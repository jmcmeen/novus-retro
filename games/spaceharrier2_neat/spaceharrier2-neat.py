"""
Project: Playing Space Harrier 2 with Gym Retro and NEAT
Purpose: Evolve a neural network to play Space Harrier 2
Created by: John McMeen
Notes:
  You will need to import you own ROMs: python -m retro.import roms/sega_classics
"""

import retro  # pip install gym-retro
import numpy as np  # pip install numpy
import cv2  # pip install opencv-python
import neat  # pip install neat-python
import pickle  # pip install cloudpickle

# create retro environment: game, state, scenario
environment = retro.make('SpaceHarrierII-Genesis', 'Level1', scenario='lives3.json')


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

        # create an alternative type of neural network with NEAT
        # network = neat.nn.FeedForwardNetwork.create(genome, config)

        # set up some variables to track fitness
        fitness = 0

        # optionally create another window for the "neural network's vision"
        # cv2.namedWindow("main", cv2.WINDOW_NORMAL)

        finished = False
        while not finished:
            # render the game
            environment.render()

            # resize and reshape the observation image
            observation = cv2.resize(observation, (inx, iny))
            # observation = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB) #alt view
            observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
            observation = np.reshape(observation, (inx, iny))

            # optional update "neural network's vision"
            # cv2.imshow('main', observation)
            # cv2.waitKey(1)

            # create a single array from 2d pixel data
            img_array = np.ndarray.flatten(observation)

            # create controller actions from input
            actions = network.activate(img_array)

            # take a peek at controller actions before translation
            # print(actions)

            # map relu activation output to 0 or 1
            actions = np.where(np.array(actions) <= 0.0, 0.0, 1.0).tolist()

            # take a peek at controller actions before translation
            # print(actions)

            # increment the emulator state
            observation, reward, done, info = environment.step(actions)

            # claim the reward
            fitness += reward

            if done:
                finished = True
                print(genome_id, fitness)

        # set the fitness for this genome
        genome.fitness = fitness


# NEAT configuration, all defaults except a config file is provided
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config')

# NEAT output
population = neat.Population(config)

population.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
population.add_reporter(stats)
population.add_reporter(neat.Checkpointer(1))

# the winning network, run for x generations
winner = population.run(eval_genomes, 1000)

# save the winning network to a binary file to reload later
with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)

exit()
