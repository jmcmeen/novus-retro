"""
Project: GymRetro + NEAT
Purpose: A generic parallel loader for GymRetro with NEAT
Input files: 
    config-retro - Simulation configuration file
    config-neat - NEAT configuration file
"""

import stable_retro as retro
import numpy as np
import cv2
import neat
import pickle
import multiprocessing
import os
from retro_config_parser import RetroConfigParser

config_file = os.path.join(os.path.dirname(__file__), 'config-sonic')

rcp = RetroConfigParser(config_file)

# eval genome takes a genome, evaluates its fitness, and returns it
def eval_genome(genome, config):
    environment = retro.make(game=rcp.game, state=rcp.state, scenario=rcp.scenario)

    # reset environment to initial state
    observation = environment.reset()

    # shape/resolution of image created by emulator
    inx, iny, inc = environment.observation_space.shape

    # scale down observation
    inx = int(inx / 8)
    iny = int(iny / 8)

    # create NEAT network
    if rcp.network_type == "recurrent":
        network = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
    elif rcp.network_type == "feedforward":
        network = neat.nn.FeedForwardNetwork.create(genome, config)
    else:
        raise ValueError("network_type must be 'recurrent' or 'feedforward'")

    # set up some variables to track fitness
    current_max_fitness = 0
    fitness = 0
    counter = 0

    finished = False
    while not finished:
        # resize and reshape the observation image
        observation = cv2.resize(observation[0], (inx, iny))
        if len(observation.shape) == 3:
            observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        observation = np.reshape(observation, (inx, iny))

        # create a single array from 2d pixel data
        img_array = np.ndarray.flatten(observation)

        # create actions from input
        actions = network.activate(img_array)

        # map relu activation output to 0 or 1
        actions = np.where(np.array(actions) <= 0.0, 0.0, 1.0).tolist()

        # increment the emulator state
        observation, reward, done, _, _ = environment.step(actions)

        # update fitness with reward from environment
        fitness += reward

        # give it x steps without improvement to improve fitness or restart
        if fitness > current_max_fitness:
            current_max_fitness = fitness
            counter = 0
        else:
            counter += 1

        if done or counter == rcp.steps_to_kill:
            finished = True

    return current_max_fitness


if __name__ == '__main__':
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'config-neat')

    # NEAT output
    population = neat.Population(config)

    if rcp.checkpoint > 0:
        #check if file exists
        if os.path.isfile(f'{rcp.game}-{rcp.state}-{rcp.scenario}-{rcp.checkpoint}'):
            population = neat.Checkpointer.restore_checkpoint(f'{rcp.game}-{rcp.state}-{rcp.scenario}-{rcp.checkpoint}')

    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(neat.Checkpointer(generation_interval=rcp.checkpoint_interval,
                                              filename_prefix=f'{rcp.game}-{rcp.state}-{rcp.scenario}-'))

    # create a parallel evaluator that will spawn workers
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)

    # the winning network up to x generations
    winner = population.run(pe.evaluate, rcp.num_generations)

    # save the winning network to a binary file to reload later
    with open(f'{rcp.game}-{rcp.state}-{rcp.scenario}-{rcp.num_generations}.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)

    exit()
