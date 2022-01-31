"""
Project: Space Harrier 2 NEAT AI Playback Neural Network
Created by: John McMeen
Purpose: Render neural network from pkl file, and create bk2 playback file
Notes:
  This will create a bk2 file from the neural network playback
  Render bk2 file to mp4 requires fmpeg: python -m retro.scripts.playback_movie winner.bk2
"""

import retro  # pip install gym-retro
import numpy as np  # pip install numpy
import neat  # pip install neat-python
import pickle  # pip install cloudpickle
import glob
import time
import os
import cv2

def playback(cur_network, fitness ):
    # make network
    print("loading network")
    # create retro environment: game, state, scenario (defines rewards)
    environment = retro.make('SpaceHarrierII-Genesis', 'Level1', scenario='lives3.json', record='.')

    # reset environment to initial state
    observation = environment.reset()

    # configuration for playback from pkl must be the same as execution
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'config')

    # NEAT setup
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    # Open pkl file, binary serialization of neural network
    with open(cur_network, 'rb') as input_file:
        genome = pickle.load(input_file)

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

    # set up loop
    finished = False
    while not finished:
        # render the game
        # environment.render()

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

        if done:
            finished = True

    print("renaming pkl file")
    os.rename(cur_network, cur_network + "p")


def rendermp4(filename, fitness):
    newfilename = str(fitness) + ".bk2"
    os.rename(filename, newfilename)

    os.system("python -m retro.scripts.playback_movie " + newfilename)

    oldfilename = newfilename
    newfilename = oldfilename + "p"

    os.rename(oldfilename, newfilename)


if __name__ == '__main__':
    while True:
        cur_networks = glob.glob("*.pkl")
        time.sleep(5)
        print("checking for new best genome")

        if cur_networks:
            print("new best genome found")
            cur_network = cur_networks[0]
            fitness = cur_network[0:-4]
            playback(cur_network, fitness)

        cur_files = glob.glob("*.bk2")

        if cur_files:
            print("converting bk2 to mp4")
            rendermp4(cur_files[0], fitness)
