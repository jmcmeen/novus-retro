"""
Project: Playing Sonic the Hedgehog with Gym Retro and NEAT threaded
Purpose: Evolve a neural network to play Sonic
Created by: NovusRetro
Adapted from LucasThompson
  Code: https://gitlab.com/lucasrthompson/Sonic-Bot-In-OpenAI-and-NEAT
  Video: https://www.youtube.com/playlist?list=PLTWFMbPFsvz3CeozHfeuJIXWAJMkPtAdS
Helpful commands
  Import Roms: python -m retro.import roms/sega_classics
"""

import retro  # pip install gym-retro
import numpy as np  # pip install numpy
import cv2  # pip install opencv-python
import neat  # pip install neat-python
import pickle  # pip install cloudpickle
import multiprocessing


# eval genome takes a genome, evaluates its fitness, and returns it
def eval_genome(genome, config):
    environment = retro.make('SonicTheHedgehog-Genesis', 'GreenHillZone.Act3', scenario='contest')

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

    finished = False
    while not finished:
        # render the game (be careful!)
        # env.render()

        # resize and reshape the observation image
        observation = cv2.resize(observation, (inx, iny))
        # observation = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB) #alt view
        observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        observation = np.reshape(observation, (inx, iny))

        # create a single array from 2d pixel data
        img_array = np.ndarray.flatten(observation)

        # create actions from input
        actions = network.activate(img_array)

        # take a peek at actions before translation
        # print(actions)

        # map relu activation output to 0 or 1
        actions = np.where(np.array(actions) <= 0.0, 0.0, 1.0).tolist()

        # take a peek at actions before translation
        # print(actions)

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

        if done or counter == 1000:
            finished = True

    return current_max_fitness


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    # local_dir = os.path.dirname(__file__)
    # config_path = os.path.join(local_dir, 'config-neat')
    # NEAT configuration, all defaults except a config file is provided
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'config-neat')

    # NEAT output
    population = neat.Population(config)
    # population = neat.Checkpointer.restore_checkpoint('neat-checkpoint-44')
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(neat.Checkpointer(generation_interval=1))

    # create a parallel evaluator that will spawn workers
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)

    # the winning network up to x generations
    winner = population.run(pe.evaluate, 20)

    # save the winning network to a binary file to reload later
    with open('winner-act1.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)

    exit()
