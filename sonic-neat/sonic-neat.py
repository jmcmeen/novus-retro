"""
Project: Playing Sonic the Hedgehog with Gym Retro and NEAT
Purpose: Evolve a neural network to play Sonic
Created by: John McMeen
Adapted from LucasThompson
  Code: https://gitlab.com/lucasrthompson/Sonic-Bot-In-OpenAI-and-NEAT
  Video: https://www.youtube.com/playlist?list=PLTWFMbPFsvz3CeozHfeuJIXWAJMkPtAdS
Helpful commands:
  Import Roms: python -m retro.import roms/sega_classics
"""

import retro  # pip install gym-retro
import numpy as np  # pip install numpy
import cv2  # pip install opencv-python
import neat  # pip install neat-python
import pickle  # pip install cloudpickle

# create retro environment: game, state, scenario (defines rewards)
environment = retro.make('SonicTheHedgehog-Genesis', 'GreenHillZone.Act1', scenario='xpos')


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

        # create NEAT network alt
        # network = neat.nn.FeedForwardNetwork.create(genome, config)

        # create NEAT network
        network = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

        # set up some variables to track fitness
        current_max_fitness = 0
        fitness = 0
        counter = 0

        cv2.namedWindow("main", cv2.WINDOW_NORMAL)

        done = False
        while not done:
            # render the game
            environment.render()

            # resize and reshape the observation image
            observation = cv2.resize(observation, (inx, iny))
            # observation = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB)
            observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
            observation = np.reshape(observation, (inx, iny))

            cv2.imshow('main', observation)
            cv2.waitKey(1)

            # create a single array from 2d pixel data
            img_array = np.ndarray.flatten(observation)

            # create actions from input
            actions = network.activate(img_array)

            # take a peek at actions before translation
            # print(actions)

            # map relu activation output to 0 or 1
            actions = np.where(np.array(actions) <= 0.0, 1.0, 0.0).tolist()

            # take a peek at actions before translation
            # print(actions)

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
                done = True
                print(genome_id, fitness)

            # set the fitness for this genome
            genome.fitness = fitness


# NEAT configuration, all defaults except a config file is provided
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')

# NEAT output
population = neat.Population(config)
# population = neat.Checkpointer.restore_checkpoint('neat-checkpoint-13')
population.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
population.add_reporter(stats)
population.add_reporter(neat.Checkpointer(10))

# the winning network
winner = population.run(eval_genomes)

# save the winning network to a binary file to reload later
with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)
