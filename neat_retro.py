"""
Project: GymRetro + NEAT
Purpose: A generic loader for GymRetro with NEAT
Input files: 
    config-retro - Simulation configuration file
    config-neat - NEAT configuration file
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

# create path to current directory
current_path = os.path.dirname(__file__)

# read configuration file
config = configparser.ConfigParser()
config.read(os.path.join(current_path, config_file))

game = config['retro']['game']
state = config['retro']['state']
scenario = config['retro']['scenario']
num_generations = int(config['simulation']['num_generations'])
network_type = config['neat']['network_type']
show_computer_view = config.getboolean('interactive', 'show_computer_view')
show_controls = config.getboolean('interactive', 'show_controls')
checkpoint = config.getint('neat', 'checkpoint')
steps_to_kill = config.getint('simulation', 'steps_to_kill')
checkpoint_interval = config.getint('neat', 'checkpoint_interval')

# create retro environment: game, state, scenario (defines rewards)
environment = retro.make(game=game, state=state, scenario=scenario)

# eval genomes takes several genomes, evaluates its fitness, and returns it
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

        if network_type == "recurrent":
            network = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
        elif network_type == "feedforward":
            network = neat.nn.FeedForwardNetwork.create(genome, config)
        else:
            raise ValueError("network_type must be 'recurrent' or 'feedforward'")

        # set up some variables to track fitness
        fitness = 0
        current_max_fitness = 0
        counter = 0

        # optionally create another window for the "neural network's vision"
        if show_computer_view:
            cv2.namedWindow("main", cv2.WINDOW_NORMAL)

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
            if show_computer_view:
                cv2.imshow('main', observation)
                cv2.waitKey(1)

            # create a single array from 2d pixel data
            img_array = np.ndarray.flatten(observation)

            # create controller actions from input
            actions = network.activate(img_array)

            # take a peek at controller actions before translation
            if show_controls:
                print(actions)

            # map relu activation output to 0 or 1
            actions = np.where(np.array(actions) <= 0.0, 0.0, 1.0).tolist()

            # take a peek at controller actions after translation
            if show_controls: 
                print(actions)

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

            if done or counter == steps_to_kill:
                finished = True
                print(genome_id, current_max_fitness)

        # set the fitness for this genome
        genome.fitness = fitness

#if main
if __name__ == '__main__':
    # NEAT configuration, all defaults except a config file is provided
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        os.path.join(current_path, 'config-neat'))

    # NEAT output
    population = neat.Population(config)

    if checkpoint > 0:
        #check if file exists
        if os.path.isfile(f'{game}-{state}-{scenario}-{checkpoint}'):
            population = neat.Checkpointer.restore_checkpoint(f'{game}-{state}-{scenario}-{checkpoint}')

    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(neat.Checkpointer(generation_interval=checkpoint_interval, file_prefix=f'{game}-{state}-{scenario}-'))

    # the winning network, run for x generations
    winner = population.run(eval_genomes, num_generations)

    # save the winning network to a binary file to reload later
    with open(f'{game}-{state}-{scenario}-{num_generations}.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)

    exit()
